mod cache;
mod server;


use std::{
	collections::HashSet,
	env,
	fmt::Display,
	fs::{self},
	io::{Cursor, Read, Seek, SeekFrom, Write},
	os::linux::net::SocketAddrExt,
	path::{Path, PathBuf},
	sync::{Arc, Mutex, atomic::AtomicUsize},
	thread::{self, JoinHandle},
};

use anyhow::{Context, bail};
use byteorder::ReadBytesExt;
use log::{error, info};
use pyo3::{
	exceptions::{PyIOError, PyValueError},
	marker::Ungil,
	prelude::*,
	types::{PyBytes, PyDict, PyString},
};
use pythonize::{depythonize, pythonize};
use rand::{prelude::*, seq::index::sample};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::os::unix::net::SocketAddr as StdSocketAddr;
use tempfile::{NamedTempFile, TempPath};
use url::Url;
use xxhash_rust::xxh3::xxh3_128;

use crate::server::{SocketConnection, start_server};


const DEFAULT_NUM_CACHE_WORKERS: usize = 8;


/// The PID of the process we were initialized in.  This should mitigate some weird forking issues.  If a forked process tries to use this module, it will still reference the original process's PID, enabling it
/// to find the cache server socket.
static INITIAL_PID: std::sync::OnceLock<u32> = std::sync::OnceLock::new();


#[pymodule]
fn flowrider(m: &Bound<'_, PyModule>) -> PyResult<()> {
	env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
		.format_timestamp_millis()
		.init();

	INITIAL_PID
		.set(std::process::id())
		.map_err(|_| PyValueError::new_err("INITIAL_PID was already set. This module should only be initialized once per process."))?;

	m.add_class::<StreamingDataset>()?;
	m.add_class::<Config>()?;
	//m.add_function(wrap_pyfunction!(server_entrypoint, m)?)?;
	Ok(())
}


#[pyclass(frozen, str)]
#[derive(Clone, Serialize, Deserialize)]
struct Config {
	/// Rank of this process on the local node (0 for the first process).
	local_rank: u32,
	/// Rank of this process in the distributed group (between 0 and world size - 1).
	global_rank: u32,
	/// Total number of processes in the distributed group.
	#[allow(dead_code)]
	world_size: u32,
	/// Name that uniquely identifies the socket for the cache server.
	socket_name: String,
	/// Directory where the cache is stored.
	cache_dir: PathBuf,
	/// Maximum number of concurrent downloads allowed.
	max_downloads: usize,
	/// How far ahead to to prefetch shards.
	readahead: usize,
}

#[pymethods]
impl Config {
	/// cache_dir: The directory where the cache is stored.
	/// cache_limit: The maximum size of the cache in bytes.  If set to 0, the cache will not be limited.
	/// max_downloads: The maximum number of concurrent downloads allowed.
	/// local_rank: The rank of this process on the local node (0 for the first process).
	/// global_rank: The rank of this process in the distributed group (between 0 and world size - 1).
	/// world_size: The total number of processes in the distributed group.
	/// master_addr: The address of the master node (usually MASTER_ADDR).
	/// master_port: The port of the master node (usually MASTER_PORT).
	///
	/// If running on a single rank (no distributed training), set `local_rank` and `global_rank` to 0 and `world_size` to 1; master_addr and master_port can be None.
	/// If local_rank, global_rank, etc are left as None they will be set from the typical PyTorch environment variables if available.  If those environment variables are not set, this will default to a single process run (local_rank = 0, global_rank = 0, world_size = 1).
	/// Note: This function will start the background cache server if it is not already running.  The cache server will only be started once per node.  It runs as a background thread of one of the processes on this local node.
	/// If the process running the cache server gets forked (which is the default for PyTorch dataloader workers), the fact that the cache server is in a different thread ensures it does not survive to the child processes (which is good, because it has non-fork safe state).
	#[new]
	#[pyo3(signature = (cache_dir, cache_limit=0, max_downloads=8, readahead=6, num_cache_workers=None, local_rank=None, global_rank=None, world_size=None, master_addr=None, master_port=None))]
	#[allow(clippy::too_many_arguments)]
	fn new(
		cache_dir: &str,
		cache_limit: u64,
		max_downloads: usize,
		readahead: usize,
		num_cache_workers: Option<usize>,
		local_rank: Option<u32>,
		global_rank: Option<u32>,
		world_size: Option<u32>,
		master_addr: Option<&str>,
		master_port: Option<u16>,
	) -> PyResult<Self> {
		let num_cache_workers = num_cache_workers.unwrap_or(DEFAULT_NUM_CACHE_WORKERS);
		let local_rank = local_rank
			.or_else(|| env::var("LOCAL_RANK").ok().and_then(|s| s.parse::<u32>().ok()))
			.unwrap_or(0);
		let global_rank = global_rank.or_else(|| env::var("RANK").ok().and_then(|s| s.parse::<u32>().ok())).unwrap_or(0);
		let world_size = world_size
			.or_else(|| env::var("WORLD_SIZE").ok().and_then(|s| s.parse::<u32>().ok()))
			.unwrap_or(1);
		let master_addr = master_addr.map(String::from).or_else(|| env::var("MASTER_ADDR").ok());
		let master_port = master_port.or_else(|| env::var("MASTER_PORT").ok().and_then(|s| s.parse::<u16>().ok()));

		if world_size == 0 {
			return Err(PyValueError::new_err("world_size cannot be 0"));
		}

		if local_rank >= world_size {
			return Err(PyValueError::new_err(format!(
				"local_rank ({}) must be less than world_size ({})",
				local_rank, world_size
			)));
		}

		if global_rank >= world_size {
			return Err(PyValueError::new_err(format!(
				"global_rank ({}) must be less than world_size ({})",
				global_rank, world_size
			)));
		}

		// create a socket name unique to this run, based on the master address and port, or the process ID in the non-distributed case.
		let socket_name = match (master_addr, master_port) {
			(Some(addr), Some(port)) => format!("flowrider-socket-{}-{}", addr, port),
			(None, None) => {
				if world_size != 1 {
					return Err(PyValueError::new_err(
						"master_addr and master_port must both be set if running in distributed mode (local_rank != 0, global_rank != 0, world_size != 1)",
					));
				}
				format!("flowrider-socket-pid-{}", INITIAL_PID.get().expect("INITIAL_PID not set"))
			},
			_ => return Err(PyValueError::new_err("master_addr and master_port must both be set or both be None")),
		};

		// Create a temporary file and write our server configuration to it.
		let mut tempfile = NamedTempFile::new().map_err(|e| PyIOError::new_err(format!("Failed to create temporary file: {:?}", e)))?;
		tempfile
			.write_all(cache_dir.as_bytes())
			.map_err(|e| PyIOError::new_err(format!("Failed to write to temporary file: {:?}", e)))?;
		tempfile
			.as_file_mut()
			.sync_all()
			.map_err(|e| PyIOError::new_err(format!("Failed to sync temporary file: {:?}", e)))?;

		// Now try to atomically move it to a location all processes agree on.  In this case, the temp folder with the name `<socket_name>-server-config`.
		// Only one process will succeed in this.  This process will start the cache server, and `<socket_name>-server-config` will reflect the configuration of that running server.
		// All other processes will read this file to confirm the server is running with the same configuration they expect.
		// Note that `<socket_name>-server-config` will be cleaned up automatically when the server stops, so we don't need to worry about leaving behind stale files.
		let server_config_path = tempfile.path().with_file_name(format!("{}-server-config", socket_name));
		let server_config = match tempfile.persist_noclobber(&server_config_path) {
			Ok(_) => {
				// We won the race - spawn the server
				info!("Spawning flowrider server...");

				let server_config_path = TempPath::from_path(&server_config_path);
				let cache_dir_clone = cache_dir.to_owned();
				let socket_name = socket_name.clone();
				thread::spawn(move || {
					// This will block until the server is stopped.
					start_server(&socket_name, cache_limit, max_downloads, cache_dir_clone, num_cache_workers);

					// Ensure we hang onto the server config file until the server stops, at which point it will be dropped and thus deleted.
					drop(server_config_path);
				});

				info!("Flowrider server spawned successfully.");

				cache_dir.to_owned()
			},
			Err(e) if e.error.kind() == std::io::ErrorKind::AlreadyExists => {
				// Another process beat us.  Read the running server's configuration from the file.
				fs::read_to_string(server_config_path).map_err(|e| PyIOError::new_err(format!("Failed to read server config file: {:?}", e)))?
			},
			Err(e) => {
				// Something weird happened
				return Err(PyIOError::new_err(format!("Failed to persist temporary file: {:?}", e)));
			},
		};

		// Make sure the running server has the same configuration as us.
		if server_config.trim() != cache_dir {
			return Err(PyValueError::new_err(format!(
				"Cache server already running with a different configuration. Expected cache_dir: {}, but got: {}",
				cache_dir, server_config
			)));
		}

		Ok(Config {
			local_rank,
			global_rank,
			world_size,
			socket_name,
			cache_dir: PathBuf::from(cache_dir),
			max_downloads,
			readahead,
		})
	}
}

impl Display for Config {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"Config(local_rank={}, global_rank={}, world_size={}, socket_name={}, cache_dir={})",
			self.local_rank,
			self.global_rank,
			self.world_size,
			self.socket_name,
			self.cache_dir.display()
		)
	}
}

impl Config {
	fn get_socket_addr(&self) -> StdSocketAddr {
		// Use abstract namespace to avoid leaving behind sockets in case of a crash or unexpected exit.
		// Abstract namespace sockets get automatically cleaned up by the kernel when the process exits.
		StdSocketAddr::from_abstract_name(self.socket_name.as_bytes()).expect("Failed to create abstract socket address")
	}
}


#[derive(Deserialize, Serialize)]
struct ShardRanges {
	shards: Vec<MDSShardReader>,
	shards_cum: Vec<u64>,
}

impl ShardRanges {
	/// Returns the total number of samples in this dataset.
	fn total_samples(&self) -> u64 {
		*self.shards_cum.last().unwrap_or(&0)
	}

	/// Given a global sample index, returns the (shard_index, offset) tuple.
	fn index_to_shard(&self, index: usize) -> (usize, usize) {
		assert!(index < self.total_samples() as usize);

		// Find the shard that contains the sample using binary search
		let shard_index = self.shards_cum.partition_point(|&b| b <= index as u64) - 1;
		let offset = index - self.shards_cum[shard_index] as usize;

		(shard_index, offset)
	}
}


#[pyclass(frozen, str)]
struct StreamingDataset {
	shards: Arc<ShardRanges>,
	stream_ranges: Vec<(u64, u64)>, // [start, end) for each stream (in global sample indices)
	conn: SocketConnection,

	seed: Vec<u8>,
	shuffle: bool,
	drop_last: bool,
	micro_batch_size: usize,
	config: Config,
}

#[pymethods]
impl StreamingDataset {
	#[new]
	fn new<'py>(
		remotes_and_locals: Vec<(String, String)>,
		config: Config,
		seed: &[u8],
		shuffle: bool,
		drop_last: bool,
		micro_batch_size: usize,
		py: Python<'py>,
	) -> PyResult<StreamingDataset> {
		// Build the streams
		let mut streams = build_streams(remotes_and_locals, &config, py).map_err(|e| PyValueError::new_err(format!("Failed to build streams: {:?}", e)))?;

		// Sort streams by their local path to ensure consistent ordering.
		// This is important for distributed training to ensure all processes see the same order even if the streams were handed to us differently.
		streams.sort_by(|a, b| a.local.cmp(&b.local));

		let mut shards: Vec<MDSShardReader> = Vec::new();
		let mut locals = HashSet::new(); // to ensure unique local paths
		let mut stream_ranges = Vec::new();
		let mut shards_cum = vec![0]; // cumulative sum of samples in shards
		let mut cum = 0;

		for stream in streams.iter() {
			let stream_start = cum;
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
				cum += shard.samples as u64;
				shards_cum.push(cum);
			}
			stream_ranges.push((stream_start, cum)); // store the range of samples for this stream
		}

		// connection to the cache server
		let conn = SocketConnection::new(config.get_socket_addr(), config.global_rank as u16);

		Ok(StreamingDataset {
			shards: Arc::new(ShardRanges { shards, shards_cum }),
			stream_ranges,
			conn,

			seed: seed.to_vec(),
			shuffle,
			drop_last,
			micro_batch_size,
			config,
		})
	}

	/// Read a sample based on its global sample index.
	fn get_sample<'py>(&self, py: Python<'py>, index: usize, worker_id: u16) -> PyResult<Bound<'py, PyDict>> {
		if index >= self.shards.total_samples() as usize {
			return Err(PyValueError::new_err(format!(
				"Sample index {} out of bounds for dataset with {} samples",
				index,
				self.shards.total_samples()
			)));
		}

		let (shard_index, offset) = self.shards.index_to_shard(index);
		let shard = &self.shards.shards[shard_index];
		let shard_hash = shard
			.hashes
			.xxh3_128
			.ok_or_else(|| PyValueError::new_err("Shard does not have xxh3_128 hash"))?;

		info!(
			"[{}] Getting sample {} from shard {} ({} samples), offset {}, hash: {:032x}",
			self.config.local_rank, index, shard_index, shard.samples, offset, shard_hash
		);

		// ask the cache server to make the shard available
		self.conn
			.send_message(shard.remote.as_str(), &shard.local, Some(shard_hash), Some(py), worker_id)
			.map_err(|e| PyIOError::new_err(format!("Failed to send message to cache server: {:?}", e)))?;

		// once the above request returns, the shard should be available on the filesystem
		let data = shard
			.read_sample(offset, &self.config.cache_dir)
			.map_err(|e| PyIOError::new_err(format!("Failed to read sample from shard: {:?}", e)))?;
		shard
			.decode_sample(py, &data)
			.map_err(|e| PyValueError::new_err(format!("Failed to decode sample data: {:?}", e)))
	}

	fn __len__(&self) -> usize {
		self.shards.total_samples() as usize
	}

	fn get_iter(&self, epoch: u64, worker_id: u16, num_workers: u16) -> DatasetIterator {
		let indices = get_work(
			&self.stream_ranges,
			self.config.global_rank,
			self.config.world_size,
			worker_id,
			num_workers,
			&self.seed,
			epoch,
			self.shuffle,
			self.drop_last,
			self.micro_batch_size,
		);

		DatasetIterator::new(indices, self.shards.clone(), self.config.clone(), worker_id)
	}

	/// Called to pickle the object.
	/// Most of our state is bland and can be straightforwardly serialized, except for the connection to the cache server,
	/// which we can just rebuild when unpickling.
	fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
		#[derive(Serialize)]
		struct StreamingDatasetState<'a> {
			shards: &'a ShardRanges,
			stream_ranges: &'a [(u64, u64)],
			seed: &'a [u8],
			shuffle: bool,
			drop_last: bool,
			micro_batch_size: usize,
			config: &'a Config,
		}

		let snapshot = StreamingDatasetState {
			shards: &self.shards,
			stream_ranges: &self.stream_ranges,
			seed: &self.seed,
			shuffle: self.shuffle,
			drop_last: self.drop_last,
			micro_batch_size: self.micro_batch_size,
			config: &self.config,
		};
		let obj = pythonize(py, &snapshot).map_err(|e| PyValueError::new_err(format!("Failed to pythonize StreamingDataset state: {:?}", e)))?;

		obj.downcast_into::<PyDict>()
			.map_err(|_| PyValueError::new_err("Failed to downcast StreamingDataset state to PyDict"))
	}

	/// Called to unpickle the object.
	#[staticmethod]
	fn __setstate__(state: Bound<'_, PyDict>) -> PyResult<StreamingDataset> {
		#[derive(Deserialize)]
		struct StreamingDatasetState {
			shards: ShardRanges,
			stream_ranges: Vec<(u64, u64)>,
			seed: Vec<u8>,
			shuffle: bool,
			drop_last: bool,
			micro_batch_size: usize,
			config: Config,
		}

		let snapshot: StreamingDatasetState =
			depythonize(&state).map_err(|e| PyValueError::new_err(format!("Failed to depythonize StreamingDataset state: {:?}", e)))?;

		let conn = SocketConnection::new(snapshot.config.get_socket_addr(), snapshot.config.global_rank as u16);

		Ok(StreamingDataset {
			shards: Arc::new(snapshot.shards),
			stream_ranges: snapshot.stream_ranges,
			conn,

			seed: snapshot.seed,
			shuffle: snapshot.shuffle,
			drop_last: snapshot.drop_last,
			micro_batch_size: snapshot.micro_batch_size,
			config: snapshot.config,
		})
	}
}

impl Display for StreamingDataset {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"StreamingDataset with {} shards and {} samples",
			self.shards.shards.len(),
			self.shards.total_samples()
		)
	}
}


#[pyclass(frozen)]
struct DatasetIterator {
	inner: Arc<DatasetIteratorInner>,
}

struct DatasetIteratorInner {
	indices: Vec<i64>,
	/// Where indices will be read from next
	read_index: AtomicUsize,
}

impl DatasetIterator {
	fn new(indices: Vec<i64>, dataset: Arc<ShardRanges>, config: Config, worker_id: u16) -> Self {
		let iter = DatasetIterator {
			inner: Arc::new(DatasetIteratorInner {
				indices,
				read_index: AtomicUsize::new(0),
			}),
		};

		// Spawn the readahead thread
		let iter_clone = iter.inner.clone();
		thread::spawn(move || {
			dataset_readahead(iter_clone, dataset, config, worker_id);
		});

		iter
	}
}

#[pymethods]
impl DatasetIterator {
	fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
		slf
	}

	fn __next__(&self) -> Option<i64> {
		let index = self.inner.read_index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
		if index >= self.inner.indices.len() {
			return None;
		}
		Some(self.inner.indices[index])
	}
}

impl Drop for DatasetIterator {
	fn drop(&mut self) {
		// This will signal the readahead thread to quit.
		self.inner.read_index.store(self.inner.indices.len(), std::sync::atomic::Ordering::SeqCst);
	}
}

fn dataset_readahead(iter: Arc<DatasetIteratorInner>, dataset: Arc<ShardRanges>, config: Config, worker_id: u16) {
	let conn = SocketConnection::new(config.get_socket_addr(), config.global_rank as u16);
	let mut index = 0;
	let total_indices = iter.indices.len();

	loop {
		let read_index = iter.read_index.load(std::sync::atomic::Ordering::SeqCst);
		// In case the worker has gotten ahead of us
		index = index.max(read_index);

		if index >= total_indices {
			break;
		}

		if index - read_index >= config.readahead {
			thread::sleep(std::time::Duration::from_millis(7));
			continue;
		}

		let sample_index = iter.indices[index];
		index += 1;

		if sample_index < 0 {
			continue;
		}

		let (shard_index, _) = dataset.index_to_shard(sample_index as usize);
		let shard = &dataset.shards[shard_index];
		let Some(shard_hash) = shard.hashes.xxh3_128 else {
			error!("Shard '{}' does not have xxh3_128 hash", shard.local);
			continue;
		};

		info!("[worker={}] readahead {}", worker_id, shard.local);

		if let Err(err) = conn.send_message(shard.remote.as_str(), &shard.local, Some(shard_hash), None, worker_id) {
			error!("[worker={}] Failed to send message to cache server for readahead: {:?}", worker_id, err);
			continue;
		}
	}
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Deserialize, Debug, Clone, Serialize)]
struct ShardHashes {
	#[serde(deserialize_with = "hex_string_to_u128", serialize_with = "u128_to_hex_string")]
	xxh3_128: Option<u128>,
}

impl MDSShardReader {
	/// Reads the raw data for a specific sample from this shard.
	fn read_sample<P: AsRef<Path>>(&self, index: usize, cache_dir: P) -> anyhow::Result<Vec<u8>> {
		let cache_dir = cache_dir.as_ref();
		if index >= self.samples as usize {
			bail!("Sample index {} out of bounds for shard with {} samples", index, self.samples);
		}

		// First we need to seek into the offset table to find the start and end of the desired sample.
		let mut file = fs::File::open(cache_dir.join(&self.local)).with_context(|| format!("Failed to open shard file: {}", self.local))?;
		file.seek(SeekFrom::Start((index as u64 + 1) * 4))
			.with_context(|| format!("Failed to seek to offset table for sample index {} in shard file: {}", index, self.local))?;
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

		// Prep numpy
		let np = py.import("numpy")?;
		let np_frombuffer = np.getattr("frombuffer")?;

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
				.decode_to_python(py, &mut reader, *size, &np_frombuffer)
				.map_err(|e| PyValueError::new_err(format!("Failed to decode column '{}': {:?}", column_name, e)))?;
			sample.set_item(column_name, value)?;
		}

		Ok(sample)
	}
}


#[pyclass(frozen)]
struct Stream {
	#[allow(dead_code)]
	remote: Url,
	local: String,
	shards: Vec<MDSShardReader>,
}

impl Stream {
	/// Reads the index.json file for this stream and builds a list of information for each shard.
	fn new<P: AsRef<Path>>(remote: Url, local: String, cache_dir: P) -> PyResult<Stream> {
		let local_path = Path::new(&local);
		let index_path = cache_dir.as_ref().join(&local).join("index.json");

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
			let shard_local = local_path
				.join(&shard.raw_data.basename)
				.to_str()
				.ok_or_else(|| PyValueError::new_err(format!("Local path is not valid UTF-8: {}", local_path.display())))?
				.to_string();
			let shard_remote = remote
				.join(&shard.raw_data.basename)
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

		Ok(Stream { remote, local, shards })
	}
}


fn download_indexes<'py>(remotes_and_locals: &[(Url, String)], config: &Config, py: Python<'py>) -> anyhow::Result<()> {
	let socket_addr = config.get_socket_addr();
	let global_rank = config.global_rank as u16;

	let mut remotes_and_locals = remotes_and_locals
		.iter()
		.map(|(remote, local)| {
			let remote_index = remote.join("index.json").context("Failed to construct index.json URL")?;
			let local_index = Path::new(local).join("index.json");
			let local_index = local_index
				.to_str()
				.ok_or_else(|| anyhow::anyhow!("Local index path is not valid UTF-8: {}", local_index.display()))?
				.to_string();

			Ok((remote_index, local_index))
		})
		.collect::<Result<Vec<_>, anyhow::Error>>()?;

	// Shuffle so that if we're in a distributed setting, everyone makes different requests to maximize parallelism.
	remotes_and_locals.shuffle(&mut rand::rng());

	// Work queue for the threads
	let remotes_and_locals = Arc::new(Mutex::new(remotes_and_locals));

	// Spawn threads to make requests to the cache server
	// We spawn max_downloads threads, to saturate the server (which does the actual downloading and limiting).
	// In a distributed setting, this will result in a lot of temporary threads, but that shouldn't be a big deal since this is only for downloading index files.
	let mut threads: Vec<JoinHandle<anyhow::Result<()>>> = Vec::new();

	for _ in 0..config.max_downloads {
		let remotes_and_locals = remotes_and_locals.clone();
		let socket_addr = socket_addr.clone();

		threads.push(thread::spawn(move || {
			let conn = SocketConnection::new(socket_addr, global_rank);

			while let Some((remote_index, local_index)) = remotes_and_locals.lock().unwrap().pop() {
				if let Err(err) = conn.send_message(remote_index.as_str(), local_index.as_str(), None, None, 0) {
					error!("Failed to send message to cache server: {:?}", err);
				}
			}

			Ok(())
		}));
	}

	// Wait for all threads to finish
	// Calls check_signals to ensure we handle any signals that might have been sent to the process.
	while !threads.is_empty() {
		py.check_signals()?;
		py.allow_threads(|| {
			thread::sleep(std::time::Duration::from_millis(100));

			for i in (0..threads.len()).rev() {
				if threads[i].is_finished() {
					let thread = threads.remove(i);
					thread.join().map_err(|e| anyhow::anyhow!("Thread panicked: {:?}", e))??;
				}
			}

			Ok::<(), anyhow::Error>(())
		})?;
	}

	Ok(())
}


/// Given a list of stream remotes and their corresponding local paths, build a list of `Stream` objects.
/// This will cause the index.json files to be downloaded (if not already) and parsed.
/// local paths must be relative paths, since they will be joined with the configured cache directory.
/// Remote paths must be valid URLs.
fn build_streams<'py>(remotes_and_locals: Vec<(String, String)>, config: &Config, py: Python<'py>) -> anyhow::Result<Vec<Stream>> {
	// parse remotes to ensure they are valid URLs
	// and ensure locals are relative paths
	let remotes_and_locals = remotes_and_locals
		.into_iter()
		.map(|(remote, local)| {
			if !Path::new(&local).is_relative() {
				return Err(anyhow::anyhow!("Local path '{}' must be a relative path", local));
			}

			// remotes must be directories, so ensure they end with a slash
			// otherwise Url will treat the last part as a file name
			let remote = if remote.ends_with('/') { remote } else { format!("{}/", remote) };

			let remote = Url::parse(&remote).map_err(|e| PyValueError::new_err(format!("Invalid remote URL: {}", e)))?;
			Ok((remote, local))
		})
		.collect::<anyhow::Result<Vec<_>>>()?;

	// Download the index files (if not already present)
	// This should only return once all index files are available.
	download_indexes(&remotes_and_locals, config, py).context("Failed to download index files")?;

	// build all the streams
	remotes_and_locals
		.into_iter()
		.map(|(remote, local)| Stream::new(remote, local.clone(), &config.cache_dir).with_context(|| format!("Failed to create Stream for {}", local)))
		.collect::<anyhow::Result<Vec<_>>>()
}


/// Based on the parameters, returns a list of global sample indices that this worker should process.
/// Micro-batches are always homogeneous with respect to streams, meaning that a micro-batch will only contain samples from a single stream.
///
/// Work is split first across ranks, round-robin style.  That means, for example:
/// Rank 0 micro-batches: 0, 4, 8
/// Rank 1 micro-batches: 1, 5, 9
/// Rank 2 micro-batches: 2, 6, 10
/// Rank 3 micro-batches: 3, 7, 11
/// That way, the full, distributed batches are correctly sequential: (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), ...
/// Then, within each rank, the work is split across workers, also round-robin style, for similar reasons and guarantees.
///
/// This function is stable even when changing the total number of ranks.  The sequence of samples/micro-batches
/// is the same regardless of the number of ranks, meaning training runs will progress deterministically.
/// This also means runs can be resumed on setups with a different number of ranks than the original run.
///
/// When shuffling is enabled, the seed and epoch are used to create a deterministic shuffle of the samples.
///
/// When drop_last is disabled, two things will happen.  First, some samples from each stream might be dropped if any stream
/// isn't divisble by the micro batch size.  Second, some micro batches will be dropped if the number of micro batches is
/// not divisible by the number of global ranks.
/// If drop_last is enabled, instead of dropping samples, the last micro batch in each stream will be padded with -1 to
/// fill it up.  And micro batches of all -1s will be added to pad out across ranks.
/// (No padding is performed to divide by the number of workers, so some workers might end up idle in some cases)
#[allow(clippy::too_many_arguments)]
fn get_work(
	stream_ranges: &[(u64, u64)],
	global_rank: u32,
	world_size: u32,
	worker_id: u16,
	num_workers: u16,
	seed: &[u8],
	epoch: u64,
	shuffle: bool,
	drop_last: bool,
	micro_batch_size: usize,
) -> Vec<i64> {
	info!(
		"[{}-{}] Getting work for epoch {} with seed {:?}, shuffle: {}, drop_last: {}, micro_batch_size: {}",
		global_rank, worker_id, epoch, seed, shuffle, drop_last, micro_batch_size,
	);
	let global_rank = global_rank as usize;
	let world_size = world_size as usize;
	let total_samples = stream_ranges.last().expect("Stream ranges should not be empty").1 as usize;

	// Deterministic shuffling based on the seed and epoch (only used if shuffle is true).
	let seed1 = xxh3_128(seed);
	let seed2 = xxh3_128(&epoch.to_le_bytes());
	let mut seed = [0u8; 32];
	seed[..16].copy_from_slice(&seed1.to_le_bytes());
	seed[16..].copy_from_slice(&seed2.to_le_bytes());
	let mut rng = ChaCha8Rng::from_seed(seed);

	// We start by going through each stream, because batches cannot contain samples from different streams.
	let mut ids = Vec::with_capacity(total_samples);

	for (sample_begin, sample_end) in stream_ranges {
		// get a list of all sample IDs in this stream, shuffled if requested.
		let stream_samples = sample_end - sample_begin;
		let mut stream_ids = if shuffle {
			sample(&mut rng, stream_samples as usize, stream_samples as usize)
				.into_iter()
				.map(|i| i as i64 + *sample_begin as i64)
				.collect::<Vec<_>>()
		} else {
			((*sample_begin as i64)..(*sample_end as i64)).collect::<Vec<_>>()
		};

		// Pad or truncate to a multiple of the micro batch size
		let remainder = stream_ids.len() % micro_batch_size;
		if remainder > 0 && drop_last {
			stream_ids.truncate(stream_ids.len() - remainder);
		} else if remainder > 0 {
			let padding = micro_batch_size - remainder;
			stream_ids.extend((0..padding).map(|_| -1)); // Use -1 to indicate padding
		}

		// Add the stream IDs to the main list.
		ids.extend(stream_ids.into_iter());
	}

	// At this point ids is already a multiple of micro_batch_size, but we'll also want it to divide evenly by the number of devices.
	let n = micro_batch_size * world_size;
	let remaining = ids.len() % n;
	if remaining > 0 && drop_last {
		ids.truncate(ids.len() - remaining);
	} else if remaining > 0 {
		ids.extend((0..(n - remaining)).map(|_| -1)); // Use -1 to indicate padding
	}

	// Now shape into a 2D array of shape (_, micro_batch_size).
	let mut chunks: Vec<_> = ids.chunks_exact(micro_batch_size).collect();
	//let num_micro_batches = ids.len() / micro_batch_size;
	//let mut ids = Array::from_shape_vec((num_micro_batches, micro_batch_size), ids).expect("Failed to create micro-batch array from IDs");

	// Shuffle the micro-batches if requested.
	if shuffle {
		chunks.shuffle(&mut rng);
		//let indices = sample(&mut rng, ids.shape()[0], ids.shape()[0]);
		//ids = ids.select(Axis(0), &indices.into_vec());
	}

	// Select the micro-batches for this device (global_rank), round robin so that, for example:
	// - Device 0 gets micro-batches 0, 2, 4, ...
	// - Device 1 gets micro-batches 1, 3, 5, ...
	let ids = chunks
		.into_iter()
		.enumerate()
		.filter_map(|(i, batch)| if i % world_size == global_rank { Some(batch) } else { None });

	// Select the micro-batches for this worker, also round-robin.
	let ids = ids
		.enumerate()
		.filter_map(|(i, batch)| if i % num_workers as usize == worker_id as usize { Some(batch) } else { None })
		.flatten()
		.cloned()
		.collect::<Vec<_>>();

	info!("[{}-{}] Work for epoch {}: {:?}", global_rank, worker_id, epoch, ids);

	ids
}


#[derive(Deserialize, Debug, Clone, Serialize)]
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
	fn decode_to_python<'py, R: Read>(&self, py: Python<'py>, mut reader: R, size: u32, np_frombuffer: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
		match self {
			ColumnEncoding::Str => {
				let mut buf = vec![0; size as usize];
				reader.read_exact(&mut buf)?;
				let value = String::from_utf8(buf).map_err(|e| PyValueError::new_err(format!("Failed to decode UTF-8 string: {:?}", e)))?;
				Ok(PyString::new(py, &value).into_any())
			},
			ColumnEncoding::Int8 => numpy_frombuffer_scalar(np_frombuffer, "<i1", &mut reader, 1),
			ColumnEncoding::Int16 => numpy_frombuffer_scalar(np_frombuffer, "<i2", &mut reader, 2),
			ColumnEncoding::Int32 => numpy_frombuffer_scalar(np_frombuffer, "<i4", &mut reader, 4),
			ColumnEncoding::Int64 | ColumnEncoding::Int => numpy_frombuffer_scalar(np_frombuffer, "<i8", &mut reader, 8),
			ColumnEncoding::Uint8 => numpy_frombuffer_scalar(np_frombuffer, "<u1", &mut reader, 1),
			ColumnEncoding::Uint16 => numpy_frombuffer_scalar(np_frombuffer, "<u2", &mut reader, 2),
			ColumnEncoding::Uint32 => numpy_frombuffer_scalar(np_frombuffer, "<u4", &mut reader, 4),
			ColumnEncoding::Uint64 => numpy_frombuffer_scalar(np_frombuffer, "<u8", &mut reader, 8),
			ColumnEncoding::Bytes => {
				let mut buf = vec![0; size as usize];
				reader.read_exact(&mut buf)?;
				Ok(PyBytes::new(py, &buf).into_any())
			},
			ColumnEncoding::Float16 => numpy_frombuffer_scalar(np_frombuffer, "<f2", &mut reader, 2),
			ColumnEncoding::Float32 => numpy_frombuffer_scalar(np_frombuffer, "<f4", &mut reader, 4),
			ColumnEncoding::Float64 => numpy_frombuffer_scalar(np_frombuffer, "<f8", &mut reader, 8),
		}
	}
}


fn numpy_frombuffer_scalar<'py, R: Read>(np_frombuffer: &Bound<'py, PyAny>, dtype: &str, reader: &mut R, size: usize) -> PyResult<Bound<'py, PyAny>> {
	assert!(size > 0 && size <= 8, "Size must be between 1 and 8 bytes for scalar types");
	let mut buf = [0u8; 8]; // max size is 8 bytes for scalar types
	reader
		.read_exact(&mut buf[..size])
		.map_err(|e| PyIOError::new_err(format!("Failed to read {} bytes: {:?}", size, e)))?;
	let array = np_frombuffer.call1((buf, dtype))?;
	array.get_item(0)
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
			Ok(Some(u128::from_be_bytes(
				bytes.try_into().map_err(|_| D::Error::custom("Failed to convert bytes to u128"))?,
			)))
		},
		None => Ok(None),
	}
}


fn u128_to_hex_string<S>(value: &Option<u128>, serializer: S) -> Result<S::Ok, S::Error>
where
	S: serde::Serializer,
{
	match value {
		Some(v) => {
			let bytes = v.to_be_bytes();
			let hex_str = hex::encode(bytes);
			serializer.serialize_some(&hex_str)
		},
		None => serializer.serialize_none(),
	}
}


trait OptionPythonExt {
	fn check_signals(&self) -> PyResult<()>;
	fn allow_threads<T, F>(&self, f: F) -> T
	where
		F: Ungil + FnOnce() -> T,
		T: Ungil;
}

impl OptionPythonExt for Option<Python<'_>> {
	fn check_signals(&self) -> PyResult<()> {
		if let Some(py) = self { py.check_signals() } else { Ok(()) }
	}

	fn allow_threads<T, F>(&self, f: F) -> T
	where
		F: Ungil + FnOnce() -> T,
		T: Ungil,
	{
		if let Some(py) = self { py.allow_threads(f) } else { f() }
	}
}


/// Does a std::thread::sleep, but allows the Python GIL to be released during the sleep (if `py` is Some).
fn std_sleep_allow_threads<'py>(duration: std::time::Duration, py: Option<Python<'py>>) {
	if let Some(py) = py {
		py.allow_threads(|| thread::sleep(duration));
	} else {
		thread::sleep(duration);
	}
}


#[cfg(test)]
mod tests {
	use super::{ColumnEncoding, get_work};
	use pyo3::prelude::*;
	use std::io::Cursor;

	/// Helper: call `decode_to_python` and return the raw `PyAny`
	fn decode<'py>(py: Python<'py>, enc: ColumnEncoding, bytes: Vec<u8>, declared_size: u32) -> Bound<'py, PyAny> {
		let np = py.import("numpy").expect("`numpy` not found – install it before running tests");
		let frombuffer = np.getattr("frombuffer").unwrap();
		enc.decode_to_python(py, Cursor::new(bytes), declared_size, &frombuffer).expect("decode failed")
	}

	/// Helper: unwraps the NumPy scalar back to a Rust value
	fn extract_scalar<'py, T: FromPyObject<'py>>(obj: Bound<'py, PyAny>) -> T {
		// `.item()` gives a pure-Python scalar that PyO3 can directly extract
		obj.call_method0("item").unwrap().extract().unwrap()
	}

	#[test]
	fn str_roundtrip() {
		Python::with_gil(|py| {
			let text = "FlowRider🚀";
			let obj = decode(py, ColumnEncoding::Str, text.as_bytes().to_vec(), text.len() as u32);
			assert_eq!(obj.extract::<&str>().unwrap(), text);
		});
	}

	#[test]
	fn bytes_roundtrip() {
		Python::with_gil(|py| {
			let data = b"\x00\xff\x10\x20".to_vec();
			let obj = decode(py, ColumnEncoding::Bytes, data.clone(), data.len() as u32);
			assert_eq!(obj.extract::<&[u8]>().unwrap(), data);
		});
	}

	macro_rules! numeric_test {
		($name:ident, $enc:path, $ty:ty, $val:expr) => {
			#[test]
			fn $name() {
				Python::with_gil(|py| {
					let value: $ty = $val;
					let obj = decode(
						py,
						$enc,
						value.to_le_bytes().to_vec(), // little-endian, just like the loader expects
						std::mem::size_of::<$ty>() as u32,
					);
					let extracted: $ty = extract_scalar(obj);
					assert_eq!(extracted, value);
				});
			}
		};
	}

	numeric_test!(int8_scalar, ColumnEncoding::Int8, i8, -123);
	numeric_test!(uint8_scalar, ColumnEncoding::Uint8, u8, 250);
	numeric_test!(int16_scalar, ColumnEncoding::Int16, i16, -30000);
	numeric_test!(uint16_scalar, ColumnEncoding::Uint16, u16, 60000);
	numeric_test!(int32_scalar, ColumnEncoding::Int32, i32, -2_000_000_000);
	numeric_test!(uint32_scalar, ColumnEncoding::Uint32, u32, 3_900_000_000);
	numeric_test!(int64_scalar, ColumnEncoding::Int64, i64, -9_000_000_000);
	numeric_test!(int_alias, ColumnEncoding::Int, i64, 42);
	numeric_test!(uint64_scalar, ColumnEncoding::Uint64, u64, 18_446_744_073_709_551_615u64);
	numeric_test!(float32_scalar, ColumnEncoding::Float32, f32, -1_234.5678_f32);
	numeric_test!(float64_scalar, ColumnEncoding::Float64, f64, 8.901234567890123e55_f64);

	#[test]
	fn insufficient_buffer_returns_error() {
		Python::with_gil(|py| {
			let np = py.import("numpy").unwrap();
			let frombuffer = np.getattr("frombuffer").unwrap();

			let buf = vec![0_u8; 2]; // Int32 needs 4 bytes
			let res = ColumnEncoding::Int32.decode_to_python(py, Cursor::new(buf), 4, &frombuffer);
			assert!(res.is_err());
		});
	}

	// Tests for get_work function
	#[test]
	fn get_work_basic_functionality() {
		// Simple test with one stream, no shuffling, no padding needed
		let stream_ranges = vec![(0, 8)]; // 8 samples in one stream
		let work = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			b"test_seed",
			0,     // epoch
			false, // shuffle
			false, // drop_last
			2,     // micro_batch_size
		);

		// Should get all samples: [0, 1, 2, 3, 4, 5, 6, 7]
		assert_eq!(work, vec![0, 1, 2, 3, 4, 5, 6, 7]);
	}

	#[test]
	fn get_work_multiple_streams_no_shuffle() {
		// Two streams with different sizes
		let stream_ranges = vec![(0, 4), (4, 8)]; // Stream 1: [0,1,2,3], Stream 2: [4,5,6,7]
		let work = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			b"test_seed",
			0,     // epoch
			false, // shuffle
			false, // drop_last
			2,     // micro_batch_size
		);

		// Should get all samples in order: [0, 1, 2, 3, 4, 5, 6, 7]
		assert_eq!(work, vec![0, 1, 2, 3, 4, 5, 6, 7]);
	}

	#[test]
	fn get_work_padding_with_drop_last_false() {
		// Test padding when stream size is not divisible by micro_batch_size
		let stream_ranges = vec![(0, 5)]; // 5 samples, micro_batch_size = 2
		let work = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			b"test_seed",
			0,     // epoch
			false, // shuffle
			false, // drop_last (padding enabled)
			2,     // micro_batch_size
		);

		// Should pad: [0, 1, 2, 3, 4, -1]
		assert_eq!(work, vec![0, 1, 2, 3, 4, -1]);
	}

	#[test]
	fn get_work_truncation_with_drop_last_true() {
		// Test truncation when drop_last is true
		let stream_ranges = vec![(0, 5)]; // 5 samples, micro_batch_size = 2
		let work = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			b"test_seed",
			0,     // epoch
			false, // shuffle
			true,  // drop_last (truncation enabled)
			2,     // micro_batch_size
		);

		// Should truncate: [0, 1, 2, 3] (last sample dropped)
		assert_eq!(work, vec![0, 1, 2, 3]);
	}

	#[test]
	fn get_work_round_robin_across_ranks() {
		// Test round-robin distribution across multiple ranks
		let stream_ranges = vec![(0, 8)]; // 8 samples
		let micro_batch_size = 2;
		let world_size = 2;

		// Get work for rank 0
		let work_rank0 = get_work(
			&stream_ranges,
			0, // global_rank
			world_size,
			0, // worker_id
			1, // num_workers
			b"test_seed",
			0,     // epoch
			false, // shuffle
			false, // drop_last
			micro_batch_size,
		);

		// Get work for rank 1
		let work_rank1 = get_work(
			&stream_ranges,
			1, // global_rank
			world_size,
			0, // worker_id
			1, // num_workers
			b"test_seed",
			0,     // epoch
			false, // shuffle
			false, // drop_last
			micro_batch_size,
		);

		// Rank 0 should get micro-batches 0, 2 -> [0, 1, 4, 5]
		// Rank 1 should get micro-batches 1, 3 -> [2, 3, 6, 7]
		assert_eq!(work_rank0, vec![0, 1, 4, 5]);
		assert_eq!(work_rank1, vec![2, 3, 6, 7]);
	}

	#[test]
	fn get_work_round_robin_across_workers() {
		// Test round-robin distribution across multiple workers within a rank
		let stream_ranges = vec![(0, 8)]; // 8 samples
		let micro_batch_size = 2;
		let num_workers = 2;

		// Get work for worker 0
		let work_worker0 = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			num_workers,
			b"test_seed",
			0,     // epoch
			false, // shuffle
			false, // drop_last
			micro_batch_size,
		);

		// Get work for worker 1
		let work_worker1 = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			1, // worker_id
			num_workers,
			b"test_seed",
			0,     // epoch
			false, // shuffle
			false, // drop_last
			micro_batch_size,
		);

		// Worker 0 should get micro-batches 0, 2 -> [0, 1, 4, 5]
		// Worker 1 should get micro-batches 1, 3 -> [2, 3, 6, 7]
		assert_eq!(work_worker0, vec![0, 1, 4, 5]);
		assert_eq!(work_worker1, vec![2, 3, 6, 7]);
	}

	#[test]
	fn get_work_deterministic_shuffling() {
		// Test that shuffling is deterministic with same seed and epoch
		let stream_ranges = vec![(0, 8)]; // 8 samples
		let seed = b"test_seed";
		let epoch = 5;

		let work1 = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			seed,
			epoch,
			true,  // shuffle
			false, // drop_last
			2,     // micro_batch_size
		);

		let work2 = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			seed,
			epoch,
			true,  // shuffle
			false, // drop_last
			2,     // micro_batch_size
		);

		// Should be identical
		assert_eq!(work1, work2);

		// But should be different from non-shuffled
		let work_no_shuffle = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			seed,
			epoch,
			false, // shuffle
			false, // drop_last
			2,     // micro_batch_size
		);

		assert_ne!(work1, work_no_shuffle);
	}

	#[test]
	fn get_work_different_epochs_different_shuffle() {
		// Test that different epochs produce different shuffles
		let stream_ranges = vec![(0, 8)]; // 8 samples
		let seed = b"test_seed";

		let work_epoch0 = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			seed,
			0,     // epoch
			true,  // shuffle
			false, // drop_last
			2,     // micro_batch_size
		);

		let work_epoch1 = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			seed,
			1,     // epoch
			true,  // shuffle
			false, // drop_last
			2,     // micro_batch_size
		);

		// Should be different (very high probability)
		assert_ne!(work_epoch0, work_epoch1);
	}

	#[test]
	fn get_work_stream_homogeneity() {
		// Test that micro-batches are homogeneous with respect to streams
		let stream_ranges = vec![(0, 4), (4, 8)]; // Two streams: [0,1,2,3] and [4,5,6,7]
		let micro_batch_size = 2;

		let work = get_work(
			&stream_ranges,
			0, // global_rank
			1, // world_size
			0, // worker_id
			1, // num_workers
			b"test_seed",
			0,     // epoch
			false, // shuffle
			false, // drop_last
			micro_batch_size,
		);

		// Check that each micro-batch contains samples from only one stream
		let micro_batches: Vec<_> = work.chunks(micro_batch_size).collect();

		for batch in micro_batches {
			// All samples in a micro-batch should be from the same stream
			let first_sample = batch[0];
			if first_sample == -1 {
				// Skip padding samples
				continue;
			}

			let stream_id = if first_sample < 4 { 0 } else { 1 };

			for &sample in batch {
				if sample == -1 {
					continue; // Skip padding
				}
				let sample_stream = if sample < 4 { 0 } else { 1 };
				assert_eq!(stream_id, sample_stream, "Micro-batch contains samples from different streams: {:?}", batch);
			}
		}
	}

	#[test]
	fn get_work_rank_stability_guarantee() {
		// The rank stability guarantee means that the sequence of samples/micro-batches
		// should be the same regardless of world size or number of workers.
		let stream_ranges = vec![(0, 128)];
		let micro_batch_size = 2;
		let batch_size = 8;
		let seed = b"stability_test";
		let epoch = 0;
		let total_samples = stream_ranges[0].1 as usize;

		// Try both shuffling enabled and disabled
		for shuffle in [false, true] {
			let mut results = Vec::new();

			// Try different world sizes and worker counts
			for (num_workers, world_size) in (1..=4).flat_map(|w| (1..=8).map(move |r| (w, r))) {
				// Only consider cases where the samples are cleanly divisible
				if total_samples % (micro_batch_size * world_size) != 0 {
					continue;
				}

				println!("Testing world_size={} num_workers={} shuffle={}", world_size, num_workers, shuffle);

				// Gather work
				let mut ranks = Vec::new();
				for rank in 0..world_size {
					let mut workers = Vec::new();
					for worker_id in 0..num_workers {
						workers.push(get_work(
							&stream_ranges,
							rank as u32,
							world_size as u32,
							worker_id as u16,
							num_workers as u16,
							seed,
							epoch,
							shuffle,
							false,
							micro_batch_size,
						));
					}
					ranks.push(workers);
				}

				// Simulate a training run
				// worker_ptrs is the current worker a given rank will pull from
				// rank_ptr is the current rank that will be pulled from
				// We iterate over all micro-batches, round-robinning like a real training loop.
				let mut worker_ptrs = vec![0usize; world_size];
				let mut rank_ptr = 0usize;
				let mut micro_batches = Vec::new();

				for _ in 0..(stream_ranges[0].1 as usize / micro_batch_size) {
					let worker_ptr = worker_ptrs[rank_ptr];
					let rank = &mut ranks[rank_ptr];
					let worker = &mut rank[worker_ptr];
					let mut micro_batch = Vec::new();
					for _ in 0..micro_batch_size {
						micro_batch.push(worker.remove(0));
					}

					micro_batches.push(micro_batch);
					worker_ptrs[rank_ptr] = (worker_ptr + 1) % num_workers;
					rank_ptr = (rank_ptr + 1) % world_size;
				}

				// Build full batches from micro-batches
				let batches = micro_batches
					.chunks_exact(batch_size / micro_batch_size)
					.map(|chunk| chunk.to_vec())
					.collect::<Vec<_>>();
				results.push(batches);
			}

			// Verify that all results are equal
			for i in 1..results.len() {
				assert_eq!(
					results[0], results[i],
					"Results should be equal across different rank and worker configurations: {:?} vs {:?}",
					results[0], results[i]
				);
			}
		}
	}

	#[test]
	fn get_work_cross_rank_worker_distribution() {
		// Test complex scenario with multiple ranks and workers
		let stream_ranges = vec![(0, 16)]; // 16 samples
		let micro_batch_size = 2;
		let world_size = 2;
		let num_workers = 2;

		// Collect work from all rank-worker combinations
		let mut all_samples = std::collections::HashSet::new();
		let mut all_work = Vec::new();

		for rank in 0..world_size {
			for worker in 0..num_workers {
				let work = get_work(
					&stream_ranges,
					rank,
					world_size,
					worker,
					num_workers,
					b"test_seed",
					0,     // epoch
					false, // shuffle
					false, // drop_last
					micro_batch_size,
				);

				// Collect all non-padding samples
				for &sample in &work {
					if sample != -1 {
						all_samples.insert(sample);
					}
				}

				all_work.push((rank, worker, work));
			}
		}

		// Should cover all samples exactly once (no duplicates, no missing)
		let expected_samples: std::collections::HashSet<_> = (0..16).collect();
		assert_eq!(
			all_samples, expected_samples,
			"All samples should be processed exactly once across all rank-worker combinations"
		);

		// Verify no work overlap between different rank-worker combinations
		for i in 0..all_work.len() {
			for j in (i + 1)..all_work.len() {
				let (rank1, worker1, work1) = &all_work[i];
				let (rank2, worker2, work2) = &all_work[j];

				let samples1: std::collections::HashSet<_> = work1.iter().filter(|&&s| s != -1).collect();
				let samples2: std::collections::HashSet<_> = work2.iter().filter(|&&s| s != -1).collect();

				let intersection: Vec<_> = samples1.intersection(&samples2).collect();
				assert!(
					intersection.is_empty(),
					"Rank {}-Worker {} and Rank {}-Worker {} should not have overlapping samples: {:?}",
					rank1,
					worker1,
					rank2,
					worker2,
					intersection
				);
			}
		}
	}

	#[test]
	fn get_work_padding_across_ranks() {
		// Test padding behavior when total micro-batches don't divide evenly across ranks
		let stream_ranges = vec![(0, 7)]; // 7 samples
		let micro_batch_size = 2;
		let world_size = 4; // This will require significant padding

		let mut all_work = Vec::new();
		for rank in 0..world_size {
			let work = get_work(
				&stream_ranges,
				rank,
				world_size,
				0, // worker_id
				1, // num_workers
				b"test_seed",
				0,     // epoch
				false, // shuffle
				false, // drop_last (padding enabled)
				micro_batch_size,
			);
			all_work.push(work);
		}

		// Each rank should get the same number of samples (due to padding)
		let work_lengths: Vec<_> = all_work.iter().map(|w| w.len()).collect();
		assert!(
			work_lengths.iter().all(|&len| len == work_lengths[0]),
			"All ranks should get the same amount of work when padding is enabled: {:?}",
			work_lengths
		);

		// Count total non-padding samples
		let total_real_samples: usize = all_work.iter().flat_map(|work| work.iter()).filter(|&&s| s != -1).count();

		// Should be equal to the original 7 samples (possibly padded within streams)
		// The stream will be padded to 8 samples (7 + 1 padding), then all samples distributed
		assert_eq!(total_real_samples, 7, "Should preserve all original samples");
	}

	#[test]
	fn get_work_empty_result_with_insufficient_samples() {
		// Test edge case where there are fewer samples than ranks/workers need
		let stream_ranges = vec![(0, 2)]; // Only 2 samples
		let micro_batch_size = 2;
		let world_size = 4; // 4 ranks, but only 1 micro-batch total

		// Check what actually happens with drop_last=true
		let mut ranks_with_work = 0;
		let mut total_work = Vec::new();

		for rank in 0..world_size {
			let work = get_work(
				&stream_ranges,
				rank,
				world_size,
				0, // worker_id
				1, // num_workers
				b"test_seed",
				0,     // epoch
				false, // shuffle
				true,  // drop_last
				micro_batch_size,
			);

			if !work.is_empty() {
				ranks_with_work += 1;
				total_work.extend(work);
			}
		}

		// With drop_last=true and insufficient samples to fill world_size micro-batches,
		// the algorithm should drop everything, so no ranks get work
		assert_eq!(ranks_with_work, 0, "No ranks should get work when insufficient samples and drop_last=true");
		assert!(total_work.is_empty(), "No work should be distributed");

		// Now test with drop_last=false to see padding behavior
		let mut ranks_with_work_padded = 0;
		let mut total_work_padded = Vec::new();

		for rank in 0..world_size {
			let work = get_work(
				&stream_ranges,
				rank,
				world_size,
				0, // worker_id
				1, // num_workers
				b"test_seed",
				0,     // epoch
				false, // shuffle
				false, // drop_last (padding enabled)
				micro_batch_size,
			);

			if !work.is_empty() {
				ranks_with_work_padded += 1;
				total_work_padded.extend(work);
			}
		}

		// With padding, all ranks should get equal work
		assert_eq!(ranks_with_work_padded, world_size, "All ranks should get work when padding is enabled");
	}
}
