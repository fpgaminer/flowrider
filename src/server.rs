use anyhow::{Context, bail, ensure};
use byteorder::{ReadBytesExt, WriteBytesExt};
use pyo3::{Python, sync::MutexExt};
use rand::distr::SampleString;
use s3::Bucket;
use std::{
	env,
	io::Write,
	os::{
		linux::net::SocketAddrExt,
		unix::net::{SocketAddr as StdSocketAddr, UnixListener as StdUnixListener, UnixStream as StdUnixStream},
	},
	path::{Path, PathBuf},
	sync::Arc,
};
use tempfile::TempPath;
use tokio::{
	fs,
	io::{AsyncReadExt, AsyncWriteExt},
	net::{UnixListener, UnixStream},
	runtime,
	sync::Semaphore,
};
use tracing::Instrument;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{Layer, fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt};
use url::Url;

use crate::{OptionPythonExt, cache::ShardCache, std_sleep_allow_threads};


pub fn start_server(socket_name: &str, cache_limit: u64, max_downloads: usize, cache_dir: String, worker_threads: usize, trace_path: Option<PathBuf>) {
	let tracing_guard = init_tracing(trace_path.as_deref());

	let socket_addr =
		StdSocketAddr::from_abstract_name(socket_name.as_bytes()).unwrap_or_else(|_| panic!("Failed to create abstract socket address: {socket_name}"));

	let rt = runtime::Builder::new_multi_thread()
		.worker_threads(worker_threads)
		.enable_all()
		.build()
		.expect("Failed to build Tokio runtime");

	tracing::info!("Starting Flowrider cache server...");
	rt.block_on(async {
		if let Err(e) = server(socket_addr, cache_limit, max_downloads, cache_dir).await {
			tracing::error!(?e, "Error in server");
		}
	});

	if let Some(guard) = tracing_guard {
		tracing::info!("Shutting down tracing");
		drop(guard); // Explicitly drop the guard to flush logs
	}
}


/// Initializes tracing with optional file logging.
/// The returned guard should be held onto until the end of the program to ensure all logs are flushed.
fn init_tracing(file_path: Option<&Path>) -> Option<WorkerGuard> {
	let stderr_layer = tracing_subscriber::fmt::layer()
		.pretty()
		.with_writer(std::io::stderr)
		.with_timer(tracing_subscriber::fmt::time::UtcTime::rfc_3339())
		.with_filter(tracing_subscriber::filter::LevelFilter::WARN);

	let (file_layer, guard) = if let Some(file_path) = file_path {
		let file_appender = tracing_appender::rolling::never(file_path.parent().unwrap_or_else(|| Path::new(".")), file_path.file_name().unwrap());
		let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

		let file_layer = tracing_subscriber::fmt::layer()
			.with_writer(file_writer)
			.with_timer(tracing_subscriber::fmt::time::UtcTime::rfc_3339())
			.with_span_events(FmtSpan::CLOSE)
			.with_ansi(false)
			.with_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")));

		(Some(file_layer), Some(guard))
	} else {
		(None, None)
	};

	tracing_subscriber::registry().with(stderr_layer).with(file_layer).init();

	if let Some(file_path) = file_path {
		tracing::info!("Tracing to {}", file_path.display());
	}

	guard
}


// TODO: Timeout
pub async fn download_file<P: AsRef<Path>>(url: &Url, dest_path: P, semaphore: &Semaphore) -> anyhow::Result<()> {
	let dest_path = dest_path.as_ref();
	let dest_parent = dest_path
		.parent()
		.ok_or_else(|| anyhow::anyhow!("Destination path must have a parent directory"))?;
	let dest_filename = dest_path.file_name().ok_or_else(|| anyhow::anyhow!("Destination path must have a file name"))?;

	// Acquire a permit from the semaphore to limit concurrent downloads
	let _download_permit = semaphore.acquire().await.expect("Failed to acquire semaphore for download");

	// If the file already exists, we can skip downloading it
	if tokio::fs::try_exists(dest_path).await.unwrap_or(false) {
		return Ok(());
	}

	tracing::info!(%url, dest_path=%dest_path.display(), "Downloading file");

	// create destination directory if it doesn't exist
	fs::create_dir_all(dest_parent)
		.await
		.context(format!("Failed to create destination directory: {}", dest_parent.display()))?;

	loop {
		// random temporary file path, created in same directory as the destination to ensure renaming is atomic
		let tmp_path = TempPath::from_path(
			dest_parent
				.join(rand::distr::Alphanumeric.sample_string(&mut rand::rng(), 16))
				.with_extension("tmp"),
		);

		// Download the file based on the URL scheme
		match url.scheme() {
			"file" => {
				// for file URLs, we just symlink directly onto the destination
				// but first, ensure destination and source are not the same
				let src_path = url.to_file_path().map_err(|_| anyhow::anyhow!("Invalid file URL: {}", url))?;
				let canonical_source = src_path
					.canonicalize()
					.context(format!("Failed to canonicalize source path: {}", src_path.display()))?;
				let canonical_dest = dest_parent
					.canonicalize()
					.context(format!("Failed to canonicalize destination path: {}", dest_parent.display()))?
					.join(dest_filename);

				ensure!(
					canonical_source != canonical_dest,
					"Source and destination paths must not be the same: {}",
					src_path.display()
				);

				// now we can create the symlink
				ensure!(src_path.exists(), "Source file does not exist: {}", src_path.display());
				fs::symlink(&canonical_source, &tmp_path).await.context(format!(
					"Failed to create symlink from {} to {}",
					canonical_source.display(),
					tmp_path.display()
				))?;
			},
			"s3" => {
				// TODO: Configurable timeout
				let download_future = s3_download(url.clone(), &tmp_path);
				match tokio::time::timeout(std::time::Duration::from_secs(60), download_future).await {
					Ok(Ok(())) => {},
					Ok(Err(e)) => {
						tracing::warn!(%url, ?e, "Failed to download S3 object. Will retry.");
						tokio::time::sleep(std::time::Duration::from_secs(1)).await;
						continue;
					},
					Err(_) => {
						tracing::warn!(%url, "S3 download timed out. Will retry.");
						tokio::time::sleep(std::time::Duration::from_secs(1)).await;
						continue;
					},
				}
			},
			_ => bail!("Unsupported URL scheme: {}", url.scheme()),
		}

		// Verify the hash of the downloaded file
		// Does not apply to the index.json file
		if dest_path
			.extension()
			.and_then(|s| s.to_str())
			.is_none_or(|ext| !ext.eq_ignore_ascii_case("json"))
		{
			let mut file = fs::File::open(&tmp_path).await.context("Failed to open temporary file for hashing")?;
			let mut hasher = xxhash_rust::xxh3::Xxh3::new();
			let mut buffer = [0; 8192];

			// Expected hash is the first 16 bytes of the file
			file.read_exact(&mut buffer[..16]).await.context("Failed to read expected hash from file")?;
			let expected_hash = u128::from_le_bytes(buffer[..16].try_into().context("Failed to read expected hash from file")?);

			loop {
				let bytes_read = file.read(&mut buffer).await.context("Failed to read from temporary file for hashing")?;
				if bytes_read == 0 {
					break; // EOF
				}
				hasher.update(&buffer[..bytes_read]);
			}

			let hash = hasher.digest128();
			if hash != expected_hash {
				tracing::warn!(
					tmp_path = %tmp_path.display(),
					"Hash mismatch for downloaded file. Expected: {:032x}, got: {:032x}. Will retry download.",
					expected_hash,
					hash
				);
				continue;
			}
		}

		// File downloaded successfully and hash verified, now we can move it to the destination path
		// Move the temporary file to the destination path atomically
		tmp_path.persist(dest_path).context("Failed to persist temporary file")?;

		return Ok(());
	}
}


/// Downloads an S3 object at the given URL to the specified path.
async fn s3_download<P: AsRef<Path>>(url: Url, tmp_path: P) -> anyhow::Result<()> {
	// Get the S3 endpoint URL and credentials
	let endpoint_url = env::var("S3_ENDPOINT_URL").unwrap_or_else(|_| "https://s3.amazonaws.com".to_string());
	let credentials = s3::creds::Credentials::default().context("Failed to get S3 credentials")?;

	tracing::info!(endpoint_url, bucket = url.host_str().unwrap_or(""), key = url.path(), "Downloading S3 object",);

	// Create the S3 bucket client
	let bucket_name = url.host_str().ok_or_else(|| anyhow::anyhow!("Invalid S3 URL: {}", url))?;
	let bucket = Bucket::new(
		bucket_name,
		s3::Region::Custom {
			region: "us-east-1".to_string(),
			endpoint: endpoint_url,
		},
		credentials,
	)
	.with_context(|| format!("Failed to establish S3 connection for bucket: {bucket_name}"))?
	.with_path_style();

	// Download the object from S3
	let file = fs::File::create(&tmp_path)
		.await
		.with_context(|| format!("Failed to create temporary file: {}", tmp_path.as_ref().display()))?;
	let mut file_writer = tokio::io::BufWriter::new(file);
	let status_code = bucket
		.get_object_to_writer(url.path(), &mut file_writer)
		.await
		.with_context(|| format!("Failed to download S3 object: {url}"))?;
	if status_code != 200 {
		bail!("Failed to download S3 object: {}. Status code: {}", url, status_code);
	}
	file_writer.flush().await.context("Failed to flush file writer")?;

	Ok(())
}


/// This is the main driver of the caching server.
/// It handles requests for shards from clients, downloading them and reaping shards to keep the cache size below the limit.
pub async fn server(addr: StdSocketAddr, cache_limit: u64, max_downloads: usize, cache_dir: String) -> std::io::Result<()> {
	let cache = ShardCache::new(cache_limit, cache_dir).await;
	let download_semaphore = Arc::new(Semaphore::new(max_downloads));

	// tokio doesn't directly support abstract namespace sockets yet, so we build a standard listener and then convert it to a tokio listener
	let std_listener = StdUnixListener::bind_addr(&addr)?;
	std_listener.set_nonblocking(true)?;
	let listener = UnixListener::from_std(std_listener)?;
	tracing::info!("Flowrider Server listening on {addr:?}");

	loop {
		let (mut stream, _) = listener.accept().await?;
		let cache = cache.clone();
		let download_semaphore = download_semaphore.clone();
		tokio::spawn(async move {
			// The first content from the client is its rank
			let client_ranks = match stream.read_u32_le().await {
				Ok(rank) => rank,
				Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
					tracing::warn!("Client disconnected before sending rank");
					return;
				},
				Err(e) => {
					tracing::error!(?e, "Failed to read client rank");
					return;
				},
			};
			let client_rank = client_ranks >> 16;
			let client_worker_id = client_ranks & 0xFFFF;
			tracing::info!(client_rank, client_worker_id, "Accepted connection from client");
			let span = tracing::info_span!("client", rank = client_rank, worker = client_worker_id);

			async move {
				if let Err(e) = handle_connection(stream, cache, download_semaphore).await {
					tracing::error!(?e, "connection error");
				}
			}
			.instrument(span)
			.await;
		});
	}
}


async fn handle_connection(mut stream: UnixStream, cache: ShardCache, download_semaphore: Arc<Semaphore>) -> anyhow::Result<()> {
	let mut buf = Vec::new();

	loop {
		// Receive a message
		// Messages always start with a 4-byte length prefix followed by their payload.
		let message_len = match stream.read_u32_le().await {
			Ok(v) => v,
			Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
			Err(e) => return Err(e).context("Failed to read message length"),
		};

		// sanity check: ensure the message length is within reasonable limits
		ensure!(
			message_len <= (131072 + 4 + 16),
			"Received message length {} exceeds maximum allowed size",
			message_len
		);

		buf.resize(message_len as usize, 0);
		stream.read_exact(&mut buf).await?;

		// Payload format: remote_uri (string), local_path (string)
		let mut cursor = std::io::Cursor::new(&buf);
		let remote = read_string(&mut cursor).context("Failed to read remote URI")?;
		let local = read_string(&mut cursor).context("Failed to read local path")?;

		// parse remote URI
		let remote_uri = Url::parse(&remote).context(format!("Failed to parse remote URI: {remote}"))?;

		// local path must be a relative path, since it will be joined with the cache directory
		ensure!(Path::new(&local).is_relative(), "Local path '{}' must be a relative path", local);

		// Get shard from cache
		// If it isn't in the cache, this will trigger a download
		// Once this returns, we can assume the shard is available at its local path (at least for awhile).
		cache.get_shard(remote_uri, &local, &download_semaphore).await?;

		stream.write_u8(1u8).await?;
	}
}


fn read_string<R: std::io::Read>(reader: &mut R) -> anyhow::Result<String> {
	let str_len = reader.read_u16::<byteorder::LittleEndian>().context("Failed to read string length")?;
	if str_len == 0 {
		return Ok(String::new());
	}

	let mut str_buf = vec![0; str_len as usize];
	reader.read_exact(&mut str_buf).context("Failed to read string data")?;

	let str_data = String::from_utf8(str_buf).context("Invalid UTF-8 data in string")?;
	Ok(str_data)
}


pub struct SocketConnection {
	addr: StdSocketAddr, // The address of the server we connect to.
	global_rank: u16,
	inner: std::sync::Mutex<Option<(StdUnixStream, u32)>>, // (stream, pid); Process ID of the process that created the connection. Used to detect forks.
}

fn connect_to_server<'py>(addr: &StdSocketAddr, global_rank: u16, worker_id: u16, py: Option<Python<'py>>) -> anyhow::Result<StdUnixStream> {
	let ranks = (global_rank as u32) << 16 | (worker_id as u32);
	let mut retries = 0;

	log::trace!("Connecting to Flowrider server at {addr:?} with global_rank: {global_rank}, worker_id: {worker_id}");

	loop {
		py.check_signals()?;
		retries += 1;

		match py.allow_threads(|| StdUnixStream::connect_addr(addr)) {
			Ok(mut stream) => {
				stream
					.set_read_timeout(Some(std::time::Duration::from_secs(1)))
					.map_err(|e| anyhow::anyhow!("Failed to set read timeout: {}", e))?;

				// introduce ourselves to the server
				if let Err(e) = py.allow_threads(|| stream.write_u32::<byteorder::LittleEndian>(ranks)) {
					log::warn!("Failed to send ranks to server (will retry): {e:?}");
					std_sleep_allow_threads(std::time::Duration::from_millis(1000), py);
					continue;
				}

				return Ok(stream);
			},
			Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
				// If the socket doesn't exist, wait and retry
				std_sleep_allow_threads(std::time::Duration::from_millis(100), py);
			},
			Err(e) => {
				// The server takes a little while to start up, so ignore the first few connection errors.
				if retries > 4 {
					log::warn!("Failed to connect to socket at {addr:?} (will retry): {e:?}");
				}
				std_sleep_allow_threads(std::time::Duration::from_millis(1000), py);
			},
		}
	}
}

impl SocketConnection {
	pub fn new(addr: StdSocketAddr, global_rank: u16) -> Self {
		SocketConnection {
			addr,
			global_rank,
			inner: std::sync::Mutex::new(None),
		}
	}

	/// Sends a message to the server and waits for a response.
	/// Cannot be used concurrently.
	/// `remote_uri` must fit in a u16, so it must be less than 65536 bytes.
	/// `local_path` must also fit in a u16, so it must be less than 65536 bytes.
	/// The server is always expected to respond with 1.
	pub fn send_message<'py>(&self, remote_uri: &str, local_path: &str, py: Option<Python<'py>>, worker_id: u16) -> anyhow::Result<u8> {
		// Prevent concurrent usage.
		let mut guard = if let Some(py) = py {
			self.inner.lock_py_attached(py)
		} else {
			self.inner.lock()
		}
		.map_err(|e| anyhow::anyhow!("Failed to lock socket connection: {:?}", e))?;

		// TODO: Timeout
		let mut buf = vec![0u8; 4]; // 4 bytes for the message length
		let remote_uri_len: u16 = remote_uri.len().try_into().expect("remote_uri length should fit in u16");
		let local_path_len: u16 = local_path.len().try_into().expect("local_path length should fit in u16");

		// Write the remote URI
		WriteBytesExt::write_u16::<byteorder::LittleEndian>(&mut buf, remote_uri_len)?;
		buf.extend_from_slice(remote_uri.as_bytes());

		// Write the local path
		WriteBytesExt::write_u16::<byteorder::LittleEndian>(&mut buf, local_path_len)?;
		buf.extend_from_slice(local_path.as_bytes());

		// Inject the message length at the start of the buffer.
		let message_len: u32 = (buf.len() - 4).try_into().expect("Message length should fit in u32");
		buf[..4].copy_from_slice(&message_len.to_le_bytes());

		// We pull the connection out of the Mutex<Option<>>.  If anything goes wrong, the connection may be left in an inconsistent state.
		// By taking the connection, we know it'll be dropped if anything goes wrong, and we can re-establish it on the next call.
		let (mut stream, pid) = match guard.take() {
			Some((stream, pid)) if pid == std::process::id() => (stream, pid), // Reuse existing connection if PID matches.
			_ => {
				let stream = connect_to_server(&self.addr, self.global_rank, worker_id, py).context("Failed to connect to server")?;
				(stream, std::process::id())
			},
		};

		// Send the message
		stream.write_all(&buf)?;

		// Wait for a response (1 byte)
		// We use a loop with a short read timeout so we can repeatedly call `check_signals` in the loop to handle ctrl+c, etc.
		let start = std::time::Instant::now();
		let mut time_warning_issued = false;
		let response = loop {
			py.check_signals()?;

			if start.elapsed() > std::time::Duration::from_secs(60) && !time_warning_issued {
				log::warn!("Waiting for response from server took longer than 60 seconds. This may indicate a problem with the server.");
				time_warning_issued = true;
			}

			match py.allow_threads(|| stream.read_u8()) {
				Ok(response) => {
					break response;
				},
				Err(e) if e.kind() == std::io::ErrorKind::TimedOut || e.kind() == std::io::ErrorKind::WouldBlock => {
					continue;
				},
				Err(e) => {
					return Err(e).context("Failed to read response from server");
				},
			}
		};

		// Put the connection back in the Mutex<Option<>>.
		*guard = Some((stream, pid));

		Ok(response)
	}
}


#[cfg(test)]
mod tests {
	use crate::server::start_server;

	use super::{SocketConnection, read_string};
	use byteorder::{LittleEndian, WriteBytesExt};
	use rand::Rng;
	use std::{
		fs,
		io::{Cursor, Read},
		os::{linux::net::SocketAddrExt, unix::net::SocketAddr as StdSocketAddr},
		path::PathBuf,
		thread,
	};
	use tempfile::TempDir;
	use url::Url;

	// ──────────────────────── read_string ──────────────────────────
	#[test]
	fn read_string_happy_and_error_paths() {
		// Empty ⇒ ""
		let mut buf = Vec::<u8>::new();
		buf.write_u16::<LittleEndian>(0).unwrap();
		assert_eq!(read_string(&mut Cursor::new(buf)).unwrap(), "");

		// Non-empty ASCII
		let mut buf = Vec::new();
		let payload = b"hello!";
		buf.write_u16::<LittleEndian>(payload.len() as u16).unwrap();
		buf.extend_from_slice(payload);
		assert_eq!(read_string(&mut Cursor::new(buf)).unwrap(), "hello!");

		// Invalid UTF-8 should error
		let mut bad = Vec::new();
		bad.write_u16::<LittleEndian>(2).unwrap();
		bad.extend_from_slice(&[0xff, 0xff]); // not UTF-8
		assert!(read_string(&mut Cursor::new(bad)).is_err());

		// Truncated stream -> error
		let mut short = Vec::new();
		short.write_u16::<LittleEndian>(10).unwrap(); // claims 10, gives 3
		short.extend_from_slice(b"abc");
		assert!(read_string(&mut Cursor::new(short)).is_err());
	}

	// Helper to build a minimal “shard” file with Flowrider hash header
	fn make_sample_file(dir: &TempDir, name: &str, payload: &[u8]) -> PathBuf {
		use xxhash_rust::xxh3::xxh3_128;
		let mut bytes = xxh3_128(payload).to_le_bytes().to_vec();
		bytes.extend_from_slice(payload);
		let path = dir.path().join(name);
		fs::write(&path, bytes).unwrap();
		path
	}

	#[test]
	fn full_server_round_trip() {
		let remote_dir = TempDir::new().unwrap();
		let cache_dir = TempDir::new().unwrap();

		let payload = b"flowrider-full-server!";
		let src_path = make_sample_file(&remote_dir, "sample.bin", payload);
		let remote_url = Url::from_file_path(&src_path).unwrap();

		// Relative path inside cache where we want the shard to end up.
		let local_rel = "data/shard.bin";

		// ────── Start Flowrider server in a background thread ─────
		let socket_name = format!("flowrider-test-{}", rand::rng().random::<u64>());
		let cache_dir_str = cache_dir.path().to_str().unwrap().to_owned();
		let addr = StdSocketAddr::from_abstract_name(socket_name.as_bytes()).expect("abstract socket addr");

		// Spawn and detach; the thread blocks forever, so we don’t join.
		thread::spawn(move || {
			// unlimited cache, at most 2 concurrent downloads, 2 worker threads
			start_server(&socket_name, 0, 2, cache_dir_str, 2, None);
		});

		// ────── Client: send request via SocketConnection ─────────
		let conn = SocketConnection::new(addr, 0);

		// This waits until the server is ready internally.
		let response = conn.send_message(remote_url.as_str(), local_rel, None, 0).expect("send_message should succeed");
		assert_eq!(response, 1, "server must reply with byte 1");

		// ────── Validate side-effects on disk ─────────────────────
		let cached_path = cache_dir.path().join(local_rel);
		assert!(cached_path.exists(), "shard should exist in cache after server response");

		// The download path for file:// URLs is a symlink to the source.
		let link_target = cached_path.read_link().expect("should be symlink");
		assert_eq!(
			link_target.canonicalize().unwrap(),
			src_path.canonicalize().unwrap(),
			"symlink should point at original source file"
		);

		// Sanity: read header back and confirm xxh3 hash matches payload.
		let mut f = fs::File::open(&src_path).unwrap();
		let mut hdr = [0u8; 16];
		f.read_exact(&mut hdr).unwrap();
		let expected = u128::from_le_bytes(hdr);
		let got = xxhash_rust::xxh3::xxh3_128(payload);
		assert_eq!(expected, got, "hash header should match payload");
	}
}
