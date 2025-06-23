use anyhow::{Context, bail, ensure};
use byteorder::{ReadBytesExt, WriteBytesExt};
use log::{error, info, warn};
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
	path::Path,
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
use url::Url;

use crate::{OptionPythonExt, cache::ShardCache, std_sleep_allow_threads};


pub fn start_server(socket_name: &str, cache_limit: u64, max_downloads: usize, cache_dir: String, worker_threads: usize) {
	let socket_addr =
		StdSocketAddr::from_abstract_name(socket_name.as_bytes()).unwrap_or_else(|_| panic!("Failed to create abstract socket address: {}", socket_name));

	let rt = runtime::Builder::new_multi_thread()
		.worker_threads(worker_threads)
		.enable_all()
		.build()
		.expect("Failed to build Tokio runtime");

	info!("Starting Flowrider cache server...");
	rt.block_on(async {
		if let Err(e) = server(socket_addr, cache_limit, max_downloads, cache_dir).await {
			error!("Error in server: {}", e);
		}
	});
}


// TODO: Timeout
pub async fn download_file<P: AsRef<Path>>(url: &Url, dest_path: P, expected_hash: Option<u128>, semaphore: &Semaphore) -> anyhow::Result<()> {
	let dest_path = dest_path.as_ref();
	let dest_parent = dest_path
		.parent()
		.ok_or_else(|| anyhow::anyhow!("Destination path must have a parent directory"))?;
	let dest_filename = dest_path.file_name().ok_or_else(|| anyhow::anyhow!("Destination path must have a file name"))?;

	// Acquire a permit from the semaphore to limit concurrent downloads
	let _download_permit = semaphore.acquire().await.expect("Failed to acquire semaphore for download");

	// If the file already exists, we can skip downloading it
	if tokio::fs::try_exists(dest_path).await.unwrap_or(false) {
		// If the file already exists, we can skip downloading it
		return Ok(());
	}

	info!("Downloading file from {} to {}", url, dest_path.display());

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
				// Get the S3 endpoint URL and credentials
				let endpoint_url = env::var("S3_ENDPOINT_URL").unwrap_or_else(|_| "https://s3.amazonaws.com".to_string());
				let credentials = s3::creds::Credentials::default().context("Failed to get S3 credentials")?;

				info!(
					"Downloading S3 object, endpoint: {}, bucket: {}, key: {}",
					endpoint_url,
					url.host_str().unwrap_or(""),
					url.path()
				);

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
				.context("Failed to establish S3 connection")?
				.with_path_style();

				// Download the object from S3
				let response_data = bucket.get_object(url.path()).await.context(format!("Failed to download S3 object: {}", url))?;
				if response_data.status_code() != 200 {
					bail!("Failed to download S3 object: {}. Status code: {}", url, response_data.status_code());
				}

				// Write the object data to the temporary file
				fs::write(&tmp_path, response_data.as_slice())
					.await
					.context(format!("Failed to write S3 object to temporary file: {}", tmp_path.display()))?;
			},
			_ => bail!("Unsupported URL scheme: {}", url.scheme()),
		}

		// Verify the hash of the downloaded file
		if let Some(expected_hash) = expected_hash {
			let mut file = fs::File::open(&tmp_path).await.context("Failed to open temporary file for hashing")?;
			let mut hasher = xxhash_rust::xxh3::Xxh3::new();
			let mut buffer = [0; 8192];

			loop {
				let bytes_read = file.read(&mut buffer).await.context("Failed to read from temporary file")?;
				if bytes_read == 0 {
					break; // EOF
				}
				hasher.update(&buffer[..bytes_read]);
			}

			let hash = hasher.digest128();
			if hash != expected_hash {
				warn!(
					"Hash mismatch for downloaded file {}. Expected: {:032x}, got: {:032x}. Will retry download.",
					tmp_path.display(),
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


/// This is the main driver of the caching server.
/// It handles requests for shards from clients, downloading them and reaping shards to keep the cache size below the limit.
pub async fn server(addr: StdSocketAddr, cache_limit: u64, max_downloads: usize, cache_dir: String) -> std::io::Result<()> {
	let cache = ShardCache::new(cache_limit, &cache_dir).await;
	let download_semaphore = Arc::new(Semaphore::new(max_downloads));

	// tokio doesn't directly support abstract namespace sockets yet, so we build a standard listener and then convert it to a tokio listener
	let std_listener = StdUnixListener::bind_addr(&addr)?;
	std_listener.set_nonblocking(true)?;
	let listener = UnixListener::from_std(std_listener)?;
	info!("Flowrider Server listening on {:?}", addr);

	loop {
		let (mut stream, _) = listener.accept().await?;
		let cache = cache.clone();
		let download_semaphore = download_semaphore.clone();
		tokio::spawn(async move {
			// The first content from the client is its rank
			let client_ranks = match stream.read_u32_le().await {
				Ok(rank) => rank,
				Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
					warn!("Client disconnected before sending rank");
					return;
				},
				Err(e) => {
					error!("Failed to read client rank: {:?}", e);
					return;
				},
			};
			let client_rank = client_ranks >> 16;
			let client_worker_id = client_rank & 0xFFFF;

			if let Err(e) = handle_connection(stream, cache, download_semaphore).await {
				error!(
					"an error occurred while handling connection from rank={},worker={}: {:?}",
					client_rank, client_worker_id, e
				);
			}
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

		// Payload format: remote_uri (string), local_path (string), expected_hash (u128)
		let mut cursor = std::io::Cursor::new(&buf);
		let remote = read_string(&mut cursor).context("Failed to read remote URI")?;
		let local = read_string(&mut cursor).context("Failed to read local path")?;
		let expected_hash = ReadBytesExt::read_u128::<byteorder::LittleEndian>(&mut cursor).context("Failed to read expected hash")?;
		info!("Received request for {} -> {} ({:032x})", remote, local, expected_hash);
		let expected_hash = if expected_hash == 0 { None } else { Some(expected_hash) };

		// parse remote URI
		let remote_uri = Url::parse(&remote).context(format!("Failed to parse remote URI: {}", remote))?;

		// local path must be a relative path, since it will be joined with the cache directory
		ensure!(Path::new(&local).is_relative(), "Local path '{}' must be a relative path", local);

		// Get shard from cache
		// If it isn't in the cache, this will trigger a download
		// Once this returns, we can assume the shard is available at its local path (at least for awhile).
		cache.get_shard(remote_uri, &local, expected_hash, &download_semaphore).await?;

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

	loop {
		py.check_signals()?;

		match py.allow_threads(|| StdUnixStream::connect_addr(addr)) {
			Ok(mut stream) => {
				stream
					.set_read_timeout(Some(std::time::Duration::from_secs(1)))
					.map_err(|e| anyhow::anyhow!("Failed to set read timeout: {}", e))?;

				// introduce ourselves to the server
				if let Err(e) = py.allow_threads(|| stream.write_u32::<byteorder::LittleEndian>(ranks)) {
					warn!("Failed to send ranks to server (will retry): {:?}", e);
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
				warn!("Failed to connect to socket at {:?} (will retry): {:?}", addr, e);
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
	pub fn send_message<'py>(
		&self,
		remote_uri: &str,
		local_path: &str,
		expected_hash: Option<u128>,
		py: Option<Python<'py>>,
		worker_id: u16,
	) -> anyhow::Result<u8> {
		// Prevent concurrent usage.
		let mut guard = if let Some(py) = py {
			self.inner.lock_py_attached(py)
		} else {
			self.inner.lock()
		}
		.map_err(|e| anyhow::anyhow!("Failed to lock socket connection: {:?}", e))?;

		// TODO: Timeout
		let mut buf = Vec::new();
		let remote_uri_len: u16 = remote_uri.len().try_into().expect("remote_uri length should fit in u16");
		let local_path_len: u16 = local_path.len().try_into().expect("local_path length should fit in u16");

		// Write the message length
		let message_len = 2 + remote_uri.len() + 2 + local_path.len() + 16; // 2 for remote_uri length, 2 for local_path length, 16 for expected_hash
		WriteBytesExt::write_u32::<byteorder::LittleEndian>(&mut buf, message_len as u32)?;

		// Write the remote URI
		WriteBytesExt::write_u16::<byteorder::LittleEndian>(&mut buf, remote_uri_len)?;
		buf.extend_from_slice(remote_uri.as_bytes());

		// Write the local path
		WriteBytesExt::write_u16::<byteorder::LittleEndian>(&mut buf, local_path_len)?;
		buf.extend_from_slice(local_path.as_bytes());

		// Write the expected hash
		WriteBytesExt::write_u128::<byteorder::LittleEndian>(&mut buf, expected_hash.unwrap_or(0))?;

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
		let response = loop {
			py.check_signals()?;

			match py.allow_threads(|| stream.read_u8()) {
				Ok(response) => {
					break response;
				},
				Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
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
