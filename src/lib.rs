mod server;
mod cache;


use std::{
	collections::HashSet, fmt::Display, fs, io::{Cursor, Read, Seek, SeekFrom}, os::linux::net::SocketAddrExt, path::{Path, PathBuf}, sync::Arc
};

use anyhow::{bail, Context};
use byteorder::ReadBytesExt;
use pyo3::{
	IntoPyObjectExt,
	exceptions::{PyIOError, PyValueError},
	prelude::*,
	types::{PyBytes, PyDict, PyString},
};
use serde::Deserialize;
use tokio::{runtime, sync::Semaphore};
use url::Url;
use std::os::unix::net::SocketAddr as StdSocketAddr;

use crate::server::{download_file, server_entrypoint, start_server, SocketConnection};


struct GlobalConfig {
	local_rank: u32,
	node_rank: u32,
	#[allow(dead_code)]
	world_size: u32,
	socket_name: String,
	cache_dir: PathBuf,
}

static GLOBAL_CONFIG: std::sync::OnceLock<GlobalConfig> = std::sync::OnceLock::new();

fn get_global_config() -> &'static GlobalConfig {
	GLOBAL_CONFIG.get().expect("Global configuration not initialized. Did you call `init`?")
}


#[pymodule]
fn flowrider(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<Stream>()?;
	m.add_class::<StreamingDataset>()?;
	m.add_function(wrap_pyfunction!(build_streams, m)?)?;
	m.add_function(wrap_pyfunction!(server_entrypoint, m)?)?;
	m.add_function(wrap_pyfunction!(init, m)?)?;
	Ok(())
}


/// This function must be called before using any other functionality in this module.
/// cache_limit: The maximum size of the cache in bytes.
/// max_downloads: The maximum number of concurrent downloads allowed.
/// local_rank: The rank of this process on the local node (0 for the first process).
/// node_rank: The rank of this process in the distributed group (between 0 and number of nodes - 1).  (Usually GROUP_RANK)
/// world_size: The total number of processes in the distributed group.
/// master_addr: The address of the master node (usually MASTER_ADDR).
/// master_port: The port of the master node (usually MASTER_PORT).
/// 
/// If running on a single rank (no distributed training), set `local_rank` and `node_rank` to 0 and `world_size` to 1; master_addr and master_port can be None.
/// 
/// A unique socket name will be generated based on the master address and port, or the process ID if they are not set. This is used to communicate with the cache server.
#[pyfunction]
fn init(cache_limit: u64, max_downloads: usize, local_rank: u32, node_rank: u32, world_size: u32, cache_dir: &str, master_addr: Option<&str>, master_port: Option<u16>) {
	// create a socket name unique to this run, based on the master address and port, or the process ID if they are not set (non-distributed case).
	let socket_name = match (master_addr, master_port) {
		(Some(addr), Some(port)) => {
			let run_hash = xxhash_rust::xxh3::xxh3_128(format!("{}:{}", addr, port).as_bytes());
			format!("flowrider-socket-{:032x}", run_hash)
		},
		(None, None) => {
			let run_hash = xxhash_rust::xxh3::xxh3_128(format!("pid={}", std::process::id()).as_bytes());
			format!("flowrider-socket-{:032x}", run_hash)
		},
		_ => panic!("master_addr and master_port must both be set or both be None"),
	};
	let config = GlobalConfig {
		local_rank,
		node_rank,
		world_size,
		socket_name: socket_name.clone(),
		cache_dir: PathBuf::from(cache_dir),
	};

	if GLOBAL_CONFIG.set(config).is_err() {
		panic!("Global configuration already initialized. Please ensure you call `init` only once per process.");
	}

		// We only want to spawn the flowrider server once (per node), so it's hidden after the panic check above and only spawnned on the local leader
	if local_rank == 0 {
		println!("Spawning flowrider server...");
		/*let exe_dir = std::env::current_exe()
			.expect("Failed to get current executable directory")
			.parent()
			.expect("Failed to get parent directory of executable")
			.join("flowrider-server");
		//let server_path = exe_dir.join("flowrider-server");

		unsafe {
			let _child = Command::new(exe_dir)
				.args(["--socket-name", &socket_name, "--cache-limit", &cache_limit.to_string(), "--max-downloads", &max_downloads.to_string(), "--cache-dir", cache_dir])
				.stdout(Stdio::inherit())
				.stderr(Stdio::inherit())
				.pre_exec(move || {
					// this code runs inside the child process just after fork and before execve
					// we want to as kthe kernel to kill us when the original process dies
					// to ensure we don't leave behind a zombie server
					if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
						return Err(std::io::Error::last_os_error());
					}
					if libc::getppid() == 1 {
						// parent vanished between fork and here - bail out!
						return Err(std::io::Error::new(std::io::ErrorKind::Other, "parent died before exec"));
					}
					Ok(())
				})
				.spawn()
				.expect("Failed to spawn flowrider server process");
		}*/

		let cache_dir = cache_dir.to_owned();
		let socket_name = socket_name.clone();
		std::thread::spawn(move || {
			// This will block until the server is stopped.
			start_server(&socket_name, cache_limit, max_downloads, cache_dir, 8); // todo: number of workers
		});

		println!("Flowrider server spawned successfully.");
	}
}


// TODO: Return numpy types


#[pyclass(frozen, str)]
struct StreamingDataset {
	shards: Vec<MDSShardReader>,
	shards_cum: Vec<u64>,
	stream_ranges: Vec<(u64, u64)>, // (start, end) for each stream
	conn: std::sync::Mutex<SocketConnection>,
}

impl StreamingDataset {
	/// Returns the total number of samples in this dataset.
	fn total_samples(&self) -> u64 {
		*self.shards_cum.last().unwrap_or(&0)
	}
}

#[pymethods]
impl StreamingDataset {
	#[new]
	fn new<'py>(streams: Vec<PyRef<'py, Stream>>) -> PyResult<StreamingDataset> {
		let mut shards: Vec<MDSShardReader> = Vec::new();
		let mut locals = HashSet::new();  // to ensure unique local paths
		let stream_ranges = Vec::new();    // todo

		for stream in streams.iter() {
			for shard in &stream.shards {
				if locals.contains(&shard.local) {
					return Err(PyValueError::new_err(format!(
						"Duplicate shard local path found: {}. Each shard must have a unique local path.",
						shard.local
					)));
				}

				// TODO: Is there some way for us to take ownership of the shard so we don't have to clone it here?
				shards.push(shard.clone());
				locals.insert(&shard.local);
			}
		}

		// cumsum of the shards to quickly find which shard contains a sample
		let mut shards_cum = Vec::with_capacity(shards.len() + 1);
		let mut cum = 0;
		shards_cum.push(cum);
		for shard in &shards {
			cum += shard.samples as u64;
			shards_cum.push(cum);
		}

		// connection to the cache server
		let conn = std::sync::Mutex::new(SocketConnection::new(get_socket_path())
				.map_err(|e| PyIOError::new_err(format!("Failed to create socket connection: {:?}", e)))?);
		

		Ok(StreamingDataset {
			shards,
			shards_cum,
			stream_ranges,
			conn,
		})
	}

	/// Read a sample based on its global sample index.
	fn get_sample<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyDict>> {
		if index >= self.total_samples() as usize {
			return Err(PyValueError::new_err(format!("Sample index {} out of bounds for dataset with {} samples", index, self.total_samples())));
		}

		// find the shard that contains the sample using binary search
		let shard_index = self.shards_cum.partition_point(|&b| b <= index as u64) - 1;
		let offset = index - self.shards_cum[shard_index] as usize;
		let shard = &self.shards[shard_index];
		let shard_hash = shard.hashes.xxh3_128
			.ok_or_else(|| PyValueError::new_err("Shard does not have xxh3_128 hash"))?;

		println!("[{}] Getting sample {} from shard {} ({} samples), offset {}, hash: {:032x}",
			get_local_rank(), index, shard_index, shard.samples, offset, shard_hash);

		// ask the cache server to make the shard available
		self.conn.lock()
			.map_err(|e| PyIOError::new_err(format!("Failed to lock cache connection: {:?}", e)))?
			.send_message(shard.remote.as_str(), &shard.local, shard_hash, py)
			.map_err(|e| PyIOError::new_err(format!("Failed to send message to cache server: {:?}", e)))?;

		// once the above request returns, the shard should be available on the filesystem
		let data = shard.read_sample(offset)
			.map_err(|e| PyIOError::new_err(format!("Failed to read sample from shard: {:?}", e)))?;
		shard.decode_sample(py, &data)
			.map_err(|e| PyValueError::new_err(format!("Failed to decode sample data: {:?}", e)))
	}

	fn __len__(&self) -> usize {
		self.total_samples() as usize
	}
}

impl Display for StreamingDataset {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "StreamingDataset with {} shards and {} samples", self.shards.len(), self.total_samples())
	}
}


/// We use abstract namespace to avoid leaving behind sockets in case of a crash or unexpected exit.
/// Abstract namespace sockets get automatically cleaned up by the kernel when the process exits.
fn get_socket_path() -> StdSocketAddr {
	let sock_name = &get_global_config().socket_name;
	StdSocketAddr::from_abstract_name(sock_name.as_bytes()).expect(format!("Failed to create abstract socket address: {}", sock_name).as_str())
}


fn get_local_rank() -> u32 {
	get_global_config().local_rank
}


/// Which node is this (between 0 and number of nodes - 1, inclusive).
fn get_node_rank() -> u32 {
	get_global_config().node_rank
}


#[derive(Deserialize, Debug)]
struct IndexJson {
	version: u32,
	shards: Vec<ShardJson>,
}

#[derive(Deserialize, Debug)]
struct ShardJson {
	column_encodings: Vec<ColumnEncoding>,
	column_names: Vec<String>,
	column_sizes: Vec<Option<u32>>,
	/// Not supported yet
	compression: Option<()>,
	format: String,
	raw_data: RawDataJson,
	samples: u32,
	version: u32,
}

#[derive(Deserialize, Debug)]
struct RawDataJson {
	basename: String,
	bytes: u64,
	hashes: ShardHashes,
}

//#[pyclass(frozen)]
#[derive(Clone, Debug)]
struct MDSShardReader {
	remote: Url,
	local: String,
	#[allow(dead_code)]
	bytes: u64,
	samples: u32,
	column_encodings: Vec<ColumnEncoding>,
	column_names: Vec<String>,
	column_sizes: Vec<Option<u32>>,
	hashes: ShardHashes,
}

#[derive(Deserialize, Debug, Clone)]
struct ShardHashes {
	#[serde(deserialize_with = "hex_string_to_u128")]
	xxh3_128: Option<u128>,
}

impl MDSShardReader {
	/// Reads the raw data for a specific sample from this shard.
	fn read_sample(&self, index: usize) -> anyhow::Result<Vec<u8>> {
		let cache_dir = &GLOBAL_CONFIG.get()
			.ok_or_else(|| anyhow::anyhow!("Global configuration not initialized. Did you call `init`?"))?
			.cache_dir;

		if index >= self.samples as usize {
			bail!("Sample index {} out of bounds for shard with {} samples", index, self.samples);
		}

		// First we need to seek into the offset table to find the start and end of the desired sample.
		let mut file = fs::File::open(cache_dir.join(&self.local)).with_context(|| format!("Failed to open shard file: {}", self.local))?;
		file.seek(SeekFrom::Start((index as u64 + 1) * 4)).with_context(|| {
			format!(
				"Failed to seek to offset table for sample index {} in shard file: {}",
				index,
				self.local
			)
		})?;
		let mut offsets: [u8; 8] = [0; 8];
		file.read_exact(&mut offsets)
			.with_context(|| format!("Failed to read offsets for sample index {} in shard file: {}", index, self.local))?;
		let offset = u32::from_le_bytes(offsets[0..4].try_into().unwrap()) as usize;
		let end = u32::from_le_bytes(offsets[4..8].try_into().unwrap()) as usize;

		// Now read the sample data.
		file.seek(SeekFrom::Start(offset as u64))
			.with_context(|| format!("Failed to seek to {} offset in shard file: {}", offset, self.local))?;
		let mut data = vec![0; end - offset];
		file.read_exact(&mut data).with_context(|| {
			format!(
				"Failed to read sample data from shard file ({}) at offset {}, len {}",
				self.local,
				offset,
				end - offset
			)
		})?;
		Ok(data)
	}

	fn decode_sample<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
		let sample = PyDict::new(py);
		let mut reader = Cursor::new(data);
		let mut sizes = Vec::with_capacity(self.column_sizes.len());

		// If any columns have a size of None, we need to read the size from the data first.
		for column_size in &self.column_sizes {
			if let Some(column_size) = column_size {
				sizes.push(*column_size);
			} else {
				let column_size = reader
					.read_u32::<byteorder::LittleEndian>()
					.map_err(|e| PyIOError::new_err(format!("Failed to read column size: {:?}", e)))?;
				sizes.push(column_size);
			}
		}

		// Now we can read each column's data based on the sizes we have.
		for ((column_name, encoding), size) in self.column_names.iter().zip(self.column_encodings.iter()).zip(sizes.iter()) {
			let value = encoding
				.decode_to_python(py, &mut reader, *size)
				.map_err(|e| PyValueError::new_err(format!("Failed to decode column '{}': {:?}", column_name, e)))?;
			sample.set_item(column_name, value)?;
		}

		Ok(sample)
	}
}


#[pyclass(frozen)]
struct Stream {
	shards: Vec<MDSShardReader>,
}

impl Stream {
	/// Reads the index.json file for this stream and builds a list of information for each shard.
	fn new(remote: &Url, local: &str) -> PyResult<Stream> {
		let cache_dir = &get_global_config().cache_dir;
		let local_path = Path::new(local);
		let index_path = cache_dir.join(local).join("index.json");

		// Parse the index.json file to get the shard information.
		let json: IndexJson = {
			let file =
				std::fs::File::open(&index_path).map_err(|e| PyIOError::new_err(format!("Failed to open index file ({}): {:?}", index_path.display(), e)))?;
			let reader = std::io::BufReader::new(file);
			serde_json::from_reader(reader).map_err(|e| PyValueError::new_err(format!("Failed to parse index JSON ({}): {:?}", index_path.display(), e)))?
		};

		if json.version != 2 {
			return Err(PyValueError::new_err(format!("Unsupported index version: {}", json.version)));
		}

		let mut shards = Vec::new();

		for shard in json.shards {
			let shard_local = local_path.join(&shard.raw_data.basename).to_str()
				.ok_or_else(|| PyValueError::new_err(format!("Local path is not valid UTF-8: {}", local_path.display())))?
				.to_string();
			let shard_remote = remote.join(&shard.raw_data.basename)
				.map_err(|e| PyValueError::new_err(format!("Failed to join remote URL: {:?}", e)))?;

			if shard.version != 2 {
				return Err(PyValueError::new_err(format!("Unsupported shard version: {}", shard.version)));
			}
			if shard.format != "mds" {
				return Err(PyValueError::new_err(format!("Unsupported shard format: {}", shard.format)));
			}
			if shard.compression.is_some() {
				return Err(PyValueError::new_err("Compression is not supported yet"));
			}

			shards.push(MDSShardReader {
					remote: shard_remote,
					local: shard_local,
					bytes: shard.raw_data.bytes,
					samples: shard.samples,
					column_encodings: shard.column_encodings,
					column_names: shard.column_names,
					column_sizes: shard.column_sizes,
					hashes: shard.raw_data.hashes,
				});
		}

		Ok(Stream {
			shards,
		})
	}
}


fn local_leader_download_indexes(remotes_and_locals: &[(Url, String)]) -> anyhow::Result<()> {
	// create a temporary Tokio runtime to download the index files concurrently
	// since tokio is not fork safe, and pytorch will fork the process for its workers,
	// we need to ensure this runtime is destroyed when we're done and no global state is left behind.
	let rt = runtime::Builder::new_current_thread()
		.enable_all()
		.build()
		.expect("Failed to build Tokio runtime");
	
	rt.block_on(async {
		let cache_dir = &get_global_config().cache_dir;

		// limit the number of concurrent downloads to avoid overwhelming the server
		let download_semaphore = Arc::new(Semaphore::new(8));

		// spawn async tasks to download all the index files concurrently
		let mut download_tasks = tokio::task::JoinSet::new();

		for (remote, local) in remotes_and_locals {
			let remote_index = remote.join("index.json")
				.context("Failed to construct index.json URL")?;
			let local_index = cache_dir.join(local).join("index.json");
			let download_semaphore = download_semaphore.clone();

			download_tasks.spawn(async move {
				// if the index file already exists, we don't need to download it again.
				if local_index.exists() {
					println!("Index file already exists at {}, skipping download", local_index.display());
					return Ok(());
				}

				download_file(&remote_index, &local_index, None, &download_semaphore).await
			});
		}

		while let Some(res) = download_tasks.join_next().await {
			res.context("Failed to join download task")?
				.context("Failed to download index file")?;
		}

		Ok::<(), anyhow::Error>(())
	})?;

	// explicitly destroy the runtime
	// according to the tokio docs, dropping the runtime should block until the runtime is fully shut down.
	drop(rt);

	Ok(())
}


/// Given a list of stream remotes and their corresponding local paths, build a list of `Stream` objects.
/// This will cause the index.json files to be downloaded (if not already) and parsed.
/// local paths must be relative paths, since they will be joined with the configured cache directory.
/// Remote paths must be valid URLs.
#[pyfunction]
fn build_streams(remotes_and_locals: Vec<(String, String)>) -> PyResult<Vec<Stream>> {
	let cache_dir = &get_global_config()
		.cache_dir;

	// parse remotes to ensure they are valid URLs
	// and ensure locals are relative paths
	let remotes_and_locals = remotes_and_locals.into_iter()
		.map(|(remote, local)| {
			if !Path::new(&local).is_relative() {
				return Err(PyValueError::new_err(format!("Local path '{}' must be a relative path", local)));
			}

			// remotes must be directories, so ensure they end with a slash
			// otherwise Url will treat the last part as a file name
			let remote = if remote.ends_with('/') {
				remote
			} else {
				format!("{}/", remote)
			};

			let remote = Url::parse(&remote).map_err(|e| PyValueError::new_err(format!("Invalid remote URL: {}", e)))?;
			println!("Remote mapped to URL: {}", remote);
			Ok((remote, local))
		})
		.collect::<Result<Vec<_>, PyErr>>()?;

	// local leader will download the index files (if not already present)
	if get_local_rank() == 0 {
		local_leader_download_indexes(&remotes_and_locals)
			.map_err(|e| PyValueError::new_err(format!("Failed to download index files: {:?}", e)))?;
	}

	// wait for the index files to be ready
	// TODO: Timeout
	for (_, local) in &remotes_and_locals {
		let local_path = cache_dir.join(local).join("index.json");
		loop {
			if local_path.exists() {
				break;
			}
			std::thread::sleep(std::time::Duration::from_millis(100));
		}
	}

	// build all the streams
	remotes_and_locals.into_iter()
		.map(|(remote, local)| {
			Stream::new(&remote, &local)
				.map_err(|e| PyValueError::new_err(format!("Failed to create Stream for {}: {:?}", local, e)))
		})
		.collect::<Result<Vec<_>, PyErr>>()
}


#[derive(Deserialize, Debug, Clone)]
enum ColumnEncoding {
	#[serde(rename = "str")]
	Str,
	#[serde(rename = "int8")]
	Int8,
	#[serde(rename = "int16")]
	Int16,
	#[serde(rename = "int32")]
	Int32,
	#[serde(rename = "int64")]
	Int64,
	#[serde(rename = "int")]
	Int,
	#[serde(rename = "uint8")]
	Uint8,
	#[serde(rename = "uint16")]
	Uint16,
	#[serde(rename = "uint32")]
	Uint32,
	#[serde(rename = "uint64")]
	Uint64,
	#[serde(rename = "bytes")]
	Bytes,
	#[serde(rename = "float16")]
	Float16,
	#[serde(rename = "float32")]
	Float32,
	#[serde(rename = "float64")]
	Float64,
	// TODO: ndarray, str_int, str_float, str_decimal, pil, jpeg, jpeg_array, jpegarray, png, list[pil], list[jpeg], list[png], pkl, json
}

impl ColumnEncoding {
	fn decode_to_python<'py, R: Read>(&self, py: Python<'py>, mut reader: R, size: u32) -> PyResult<Bound<'py, PyAny>> {
		match self {
			ColumnEncoding::Str => {
				let mut buf = vec![0; size as usize];
				reader.read_exact(&mut buf)?;
				let value = String::from_utf8(buf).map_err(|e| PyValueError::new_err(format!("Failed to decode UTF-8 string: {:?}", e)))?;
				Ok(PyString::new(py, &value).into_any())
			},
			ColumnEncoding::Int8 => {
				let value = reader.read_i8()?;
				value.into_bound_py_any(py)
			},
			ColumnEncoding::Int16 => {
				let value = reader.read_i16::<byteorder::LittleEndian>()?;
				value.into_bound_py_any(py)
			},
			ColumnEncoding::Int32 => {
				let value = reader.read_i32::<byteorder::LittleEndian>()?;
				value.into_bound_py_any(py)
			},
			ColumnEncoding::Int64 | ColumnEncoding::Int => {
				let value = reader.read_i64::<byteorder::LittleEndian>()?;
				value.into_bound_py_any(py)
			},
			ColumnEncoding::Uint8 => {
				let value = reader.read_u8()?;
				value.into_bound_py_any(py)
			},
			ColumnEncoding::Uint16 => {
				let value = reader.read_u16::<byteorder::LittleEndian>()?;
				value.into_bound_py_any(py)
			},
			ColumnEncoding::Uint32 => {
				let value = reader.read_u32::<byteorder::LittleEndian>()?;
				value.into_bound_py_any(py)
			},
			ColumnEncoding::Uint64 => {
				let value = reader.read_u64::<byteorder::LittleEndian>()?;
				value.into_bound_py_any(py)
			},
			ColumnEncoding::Bytes => {
				let mut buf = vec![0; size as usize];
				reader.read_exact(&mut buf)?;
				Ok(PyBytes::new(py, &buf).into_any())
			},
			ColumnEncoding::Float16 => {
				unimplemented!("Float16 decoding is not implemented yet, due to lack of support for float16 in Rust standard library");
			},
			ColumnEncoding::Float32 => {
				let value = reader.read_f32::<byteorder::LittleEndian>()?;
				value.into_bound_py_any(py)
			},
			ColumnEncoding::Float64 => {
				let value = reader.read_f64::<byteorder::LittleEndian>()?;
				value.into_bound_py_any(py)
			},
		}
	}
}


fn hex_string_to_u128<'de, D>(deserializer: D) -> Result<Option<u128>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	use serde::de::Error;

	let opt_string: Option<String> = Option::deserialize(deserializer)?;
	match opt_string {
		Some(hex_str) => {
			let bytes = hex::decode(&hex_str).map_err(|e| D::Error::custom(format!("Invalid hex string: {}", e)))?;
			if bytes.len() != 16 {
				return Err(D::Error::custom("Hex string must be exactly 16 bytes (128 bits)"));
			}
			Ok(Some(u128::from_be_bytes(bytes.try_into().map_err(|_| D::Error::custom("Failed to convert bytes to u128"))?)))
		},
		None => Ok(None),
	}
}


/*async fn wait_for_file(path: &Path, timeout: std::time::Duration) -> anyhow::Result<()> {
	let start = std::time::Instant::now();
	while !path.exists() {
		if start.elapsed() > timeout {
			bail!("Timeout waiting for file: {}", path.display());
		}
		tokio::time::sleep(std::time::Duration::from_millis(100)).await;
	}
	Ok(())
}*/