/// This module implements HeavyData, a simple key-value store for large binary blobs,
/// such as for a training dataset's images, latents, etc.
///
/// The values for each key are stored in a custom binary format in "shard" files, and are compressed with zstd.
/// Using shards allows us to handle situations where there are millions of samples.  e.g. 20M samples might end up
/// being only a thousand 4GB files.  That's much more manageable than 20M individual files.
///
/// The index is stored in a SQLite database, which maps keys to their location in the shard files.
/// This allows fast lookups without storing the full index in memory.
///
/// The system is crash resilient - in case of a crash the worst case is that some samples that were in the process of being written
/// get lost, but the index will remain consistent and no data will be corrupted.
///
/// Currently does not support deleting or updating existing samples.  Writing the same key twice will have no effect.
///
/// When `write` is called the sample is handed off to a background thread, so the main thread can continue working.
use anyhow::{Context as _, ensure};
use pyo3::{
	Bound, PyResult, Python,
	exceptions::{PyIOError, PyValueError},
	pyclass, pymethods,
	types::{PyAnyMethods, PyBytes, PyDict, PySet, PySetMethods},
};
use rusqlite::{
	Connection, OptionalExtension as _, ToSql, params,
	types::{FromSql, ValueRef},
};
use std::{
	collections::HashMap,
	fs::{self, OpenOptions},
	io::{Read as _, Seek as _, SeekFrom, Write},
	path::{Path, PathBuf},
	sync::{Mutex, mpsc},
	thread::{self, JoinHandle},
};
use xxhash_rust::xxh3::xxh3_128;

use crate::encoding::{ColumnEncoding, ColumnValue};


const MAX_SHARD_SIZE: u64 = u32::MAX as u64; // 4GB
const MAX_WRITE_QUEUE: usize = 1024; // Maximum number of pending write jobs in the queue


const INDEX_SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS index_table (
    key       BLOB PRIMARY KEY,
    shard_num INTEGER NOT NULL,
    position  INTEGER NOT NULL,
    length    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS columns (
    name     TEXT PRIMARY KEY,
    encoding TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shard_positions (
    shard_num INTEGER PRIMARY KEY,
    position  INTEGER NOT NULL
);
"#;


#[pyclass]
pub struct HeavyData {
	path: PathBuf,
	conn: Mutex<Connection>,
	columns: Vec<(String, ColumnEncoding)>, // Sorted and canonical
	compression_level: i32,
	worker: Option<HeavyDataWorker>,
}

struct HeavyDataWorker {
	handle: JoinHandle<()>,
	work_tx: mpsc::SyncSender<SampleWriterJob>,
	result_rx: Mutex<mpsc::Receiver<anyhow::Error>>,
	owner_pid: u32,
}

impl HeavyDataWorker {
	fn new(path: PathBuf, compression_level: i32) -> Self {
		let (result_tx, result_rx) = mpsc::channel();
		let (work_tx, work_rx) = mpsc::sync_channel(MAX_WRITE_QUEUE);

		let path_clone = path.clone();
		let handle = thread::spawn(move || {
			if let Err(e) = sample_writer_worker(work_rx, path_clone, compression_level) {
				let _ = result_tx.send(e);
			}
		});

		Self {
			handle,
			work_tx,
			result_rx: Mutex::new(result_rx),
			owner_pid: std::process::id(),
		}
	}
}

#[pymethods]
impl HeavyData {
	#[new]
	#[pyo3(signature = (path, columns=None, *, compression_level=9))]
	fn new(path: &str, columns: Option<HashMap<String, ColumnEncoding>>, compression_level: i32) -> PyResult<HeavyData> {
		let path = PathBuf::from(path);

		let (conn, columns) =
			create_or_resume_index(&path, columns.as_ref()).map_err(|e| PyValueError::new_err(format!("Failed to create or resume index: {e:?}")))?;

		Ok(HeavyData {
			path,
			conn: Mutex::new(conn),
			columns,
			compression_level,
			worker: None,
		})
	}

	fn read<'py>(&self, key: &[u8], py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
		let Some(sample_data) = py
			.detach(|| self.read_sample_data(key))
			.map_err(|e| PyIOError::new_err(format!("Failed to read sample data: {e:?}")))?
		else {
			return Ok(None);
		};

		// Decode the sample data into a Python dict
		crate::encoding::decode_sample(py, &sample_data[key.len()..], &self.columns)
			.map_err(|e| PyIOError::new_err(format!("Failed to decode sample data: {e:?}")))
			.map(Some)
	}

	fn write<'py>(&mut self, key: &[u8], sample: Bound<'py, PyDict>, py: Python<'py>) -> PyResult<()> {
		let worker = self
			.worker
			.get_or_insert_with(|| HeavyDataWorker::new(self.path.clone(), self.compression_level));

		if worker.owner_pid != std::process::id() {
			return Err(PyValueError::new_err(
				"HeavyData writer was inherited across fork; this is not supported since the worker thread does not exist in the child process",
			));
		}

		let sample_values = self
			.columns
			.iter()
			.map(|(name, enc)| {
				let v = sample.get_item(name)?;
				ColumnValue::from_python(v, enc)
			})
			.collect::<PyResult<Vec<_>>>()?;

		let job = SampleWriterJob {
			key: key.to_vec(),
			sample: sample_values,
		};

		py.detach(|| worker.work_tx.send(job))
			.map_err(|e| PyIOError::new_err(format!("Failed to send job to worker: {e:?}")))?;

		// Check for errors from workers
		let result_rx = worker
			.result_rx
			.lock()
			.map_err(|e| PyIOError::new_err(format!("Failed to lock result receiver: {e:?}")))?;
		if let Ok(err) = result_rx.try_recv() {
			return Err(PyValueError::new_err(format!("Worker error: {err:?}")));
		}

		Ok(())
	}

	fn finish<'py>(&mut self, py: Python<'py>) -> PyResult<()> {
		let Some(worker) = self.worker.take() else {
			return Ok(()); // Already finished
		};

		if worker.owner_pid != std::process::id() {
			// Process was forked after creating the worker, which means the thread doesn't exist here
			return Err(PyValueError::new_err(
				"HeavyData writer was inherited across fork; this is not supported since the worker thread does not exist in the child process",
			));
		}

		// Signal worker to finish by dropping the sender
		drop(worker.work_tx);

		// Wait for the worker to finish
		py.detach(|| worker.handle.join().map_err(|_| PyValueError::new_err("Worker thread panicked")))?;

		// Check for errors from workers
		let result_rx = worker
			.result_rx
			.lock()
			.map_err(|e| PyIOError::new_err(format!("Failed to lock result receiver: {e:?}")))?;
		if let Ok(err) = result_rx.try_recv() {
			return Err(PyValueError::new_err(format!("Worker error: {err:?}")));
		}

		Ok(())
	}

	#[getter]
	fn existing_keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
		let map_sql = |e| PyIOError::new_err(format!("Failed to query existing keys from index database: {e}"));

		let conn = self
			.conn
			.lock()
			.map_err(|e| PyIOError::new_err(format!("Failed to lock database connection: {e:?}")))?;
		let mut stmt = conn.prepare_cached("SELECT key FROM index_table").map_err(map_sql)?;
		let mut rows = stmt.query(()).map_err(map_sql)?;
		let existing_keys: Bound<'_, PySet> = PySet::empty(py)?;

		while let Some(row) = rows.next().map_err(map_sql)? {
			let v = row.get_ref(0).map_err(map_sql)?;
			let key: &[u8] = match v {
				ValueRef::Blob(b) => b,
				_ => return Err(PyIOError::new_err("key column was not a BLOB")),
			};

			PySetMethods::add(&existing_keys, PyBytes::new(py, key))?;
		}

		Ok(existing_keys)
	}
}

impl HeavyData {
	fn read_sample_data(&self, key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
		let (shard_num, position, length) = {
			let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("Failed to lock database connection: {e:?}"))?;

			// Look up the key in the index to find the shard and position of the sample data
			let mut stmt = conn
				.prepare_cached("SELECT shard_num, position, length FROM index_table WHERE key = ?1")
				.context("Failed to prepare SQL statement")?;
			let Some(result) = stmt
				.query_row(params![key], |row| {
					let shard_num: u16 = row.get(0)?;
					let position = row.get::<_, i64>(1)? as u64;
					let length = row.get::<_, i64>(2)? as u64;
					Ok((shard_num, position, length))
				})
				.optional()
				.context("Failed to query index database")?
			else {
				return Ok(None);
			};
			result
		};
		ensure!(length <= MAX_SHARD_SIZE, "Sample data length exceeds maximum shard size, data may be corrupted");

		// Read the sample data from the shard file
		let shard_path = shard_path(&self.path, shard_num);
		let mut file = OpenOptions::new().read(true).open(shard_path).context("Failed to open shard file")?;
		file.seek(SeekFrom::Start(position)).context("Failed to seek in shard file")?;
		let mut compressed_data = vec![0u8; length as usize];
		file.read_exact(&mut compressed_data).context("Failed to read sample data from shard file")?;

		// Decompress the sample data
		let mut sample_data = zstd::decode_all(&compressed_data[..]).context("Failed to decompress sample data")?;

		// Verify checksum and key
		ensure!(sample_data.len() >= (key.len() + 4), "Sample data is too short to contain key and checksum");

		let checksum = u32::from_le_bytes(sample_data[sample_data.len() - 4..].try_into().unwrap());
		let calculated_checksum = (xxh3_128(&sample_data[..sample_data.len() - 4]) & 0xFFFF_FFFF) as u32;
		ensure!(checksum == calculated_checksum, "Checksum mismatch for sample data, data may be corrupted");
		sample_data.truncate(sample_data.len() - 4); // Remove checksum from data

		ensure!(&sample_data[..key.len()] == key, "Key mismatch for sample data, data may be corrupted");

		Ok(Some(sample_data))
	}
}

impl Drop for HeavyData {
	fn drop(&mut self) {
		// We ignore errors in drop since there's not much we can do about them
		Python::try_attach(|py| {
			let _ = self.finish(py);
		});
	}
}


struct SampleWriterJob {
	key: Vec<u8>,
	sample: Vec<ColumnValue>,
}


fn sample_writer_worker(rx: mpsc::Receiver<SampleWriterJob>, path: PathBuf, compression_level: i32) -> anyhow::Result<()> {
	// The index file should already exist at this point, so we don't need to configure or set it up here.
	let mut flags = rusqlite::OpenFlags::default();
	flags.set(rusqlite::OpenFlags::SQLITE_OPEN_CREATE, false);
	let mut conn = Connection::open_with_flags(path.join("index.sqlite"), flags).context("Failed to open index database")?;

	let mut current_shard = 0u16;
	let mut shard_file = None;

	let mut buf = Vec::new();
	let mut buf2 = Vec::new();

	while let Ok(sample) = rx.recv() {
		// Check if the key already exists in the index.  If it does, we skip writing this sample since we don't support updates.
		if conn.prepare_cached("SELECT 1 FROM index_table WHERE key = ?1")?.exists(params![&sample.key])? {
			continue;
		}

		// Encode the sample
		buf.clear();
		buf.extend_from_slice(&sample.key);
		for column in sample.sample {
			column.encode(&mut buf).context("Failed to encode column value")?;
		}

		// Checksum
		let checksum = (xxh3_128(&buf) & 0xFFFF_FFFF) as u32; // 32-bit hash for integrity checking
		buf.extend_from_slice(&checksum.to_le_bytes());

		// Compress the sample
		buf2.clear();
		zstd::stream::copy_encode(&*buf, &mut buf2, compression_level).context("Failed to compress sample")?;

		// If the current shard is too large, go to the next one until we find one with enough space
		let sample_size: u32 = buf2.len().try_into().context("Sample size exceeds 4GB")?;

		let shard_position = loop {
			let mut shard_pos_stmt = conn.prepare_cached("SELECT COALESCE((SELECT position FROM shard_positions WHERE shard_num = ?1), 0)")?;
			let shard_position: u64 = shard_pos_stmt
				.query_row(params![current_shard], |row| row.get::<_, i64>(0))?
				.try_into()
				.context("Shard position corrupted")?;

			if shard_position.checked_add(sample_size as u64).unwrap() > MAX_SHARD_SIZE {
				current_shard = current_shard.checked_add(1).context("Exceeded maximum number of shards (65535)")?;
				shard_file = None;
			} else {
				break shard_position;
			}
		};

		// Write the sample to the shard
		let shard_file = match &mut shard_file {
			Some(f) => f,
			None => {
				let file = OpenOptions::new()
					.read(true)
					.write(true)
					.create(true)
					.truncate(false)
					.open(shard_path(&path, current_shard))
					.context("Failed to open shard file")?;
				shard_file.get_or_insert(file)
			},
		};
		shard_file.seek(SeekFrom::Start(shard_position)).context("Failed to seek in shard file")?;
		shard_file.write_all(&buf2).context("Failed to write sample data")?;

		// Sync shard write to disk so we can guarantee it exists before writing the index entry
		shard_file.sync_all().context("Failed to sync shard file")?;

		// Update the index
		let tx = conn.transaction()?;
		tx.execute(
			"INSERT INTO index_table (key, shard_num, position, length) VALUES (?1, ?2, ?3, ?4)",
			params![&sample.key, current_shard, shard_position as i64, sample_size],
		)?;
		tx.execute(
			"INSERT OR REPLACE INTO shard_positions (shard_num, position) VALUES (?1, ?2)",
			params![current_shard, (shard_position + sample_size as u64) as i64],
		)?;
		tx.commit()?;
	}

	Ok(())
}


/// Open the index at the given path, creating it if it doesn't exist, and verify that the columns match the provided schema.  Returns a connection to the index database.
/// Assumes columns is already sorted by name.
/// If columns is None, this will read the existing schema from the database.  If the database does not exist in this case an error will be returned.
/// Returns the index database connection and the canonical column schema (matching the on-disk schema).
fn create_or_resume_index(path: &Path, columns: Option<&HashMap<String, ColumnEncoding>>) -> anyhow::Result<(Connection, Vec<(String, ColumnEncoding)>)> {
	fs::create_dir_all(path).context("Failed to create index directory")?;
	ensure!(
		columns.map(|cols| !cols.is_empty()).unwrap_or(true),
		"At least one column must be provided when creating a new index"
	);

	// Open the index database
	let index_path = path.join("index.sqlite");
	let mut flags = rusqlite::OpenFlags::default();
	if columns.is_none() {
		// If columns is None, we expect the database to already exist, so we don't allow creating a new one.
		flags.set(rusqlite::OpenFlags::SQLITE_OPEN_CREATE, false);
	}
	let mut conn = Connection::open_with_flags(&index_path, flags).context("Failed to open index SQLite database")?;
	conn.pragma_update(None, "journal_mode", "WAL")?;

	// Create tables if they don't exist
	conn.execute_batch(INDEX_SCHEMA_SQL).context("Failed to create/verify index schema")?;

	let db_columns: Vec<(String, ColumnEncoding)> = {
		// Check if the database already has a schema or not
		let column_count = conn.query_one("SELECT COUNT(*) FROM columns", [], |row| row.get::<_, i64>(0))?;
		match (column_count, columns) {
			// Database does not have a schema, but columns were provided - we create the schema from the provided columns
			(0, Some(cols)) => {
				let tx = conn.transaction()?;
				for (name, encoding) in cols {
					tx.execute("INSERT INTO columns (name, encoding) VALUES (?1, ?2)", (name, encoding))?;
				}
				tx.commit()?;
			},

			// Database does not have a schema and no columns were provided - we cannot proceed since we don't know the schema
			(0, None) => anyhow::bail!("No columns provided and index database does not contain an existing schema"),

			// Schema already exists in the database.
			(_, Some(_)) => {},
			(_, None) => {},
		}

		// Read the existing schema from the database, sorted for consistency
		let mut stmt = conn.prepare("SELECT name, encoding FROM columns ORDER BY name ASC")?;
		stmt.query_map([], |row| {
			let name: String = row.get(0)?;
			let encoding: ColumnEncoding = row.get(1)?;
			Ok((name, encoding))
		})?
		.collect::<Result<_, _>>()
		.context("Failed to query existing columns from index database")?
	};

	// If columns were provided, verify that they match the existing schema in the database.  This ensures that if the index already exists on disk, we don't accidentally use the wrong schema to interpret it.
	if let Some(provided_columns) = columns {
		let mut provided_columns = provided_columns.iter().collect::<Vec<_>>();
		provided_columns.sort_by(|a, b| a.0.cmp(b.0));
		ensure!(
			db_columns.iter().map(|(a, b)| (a, b)).eq(provided_columns.iter().copied()),
			"Existing index schema does not match the provided schema.  Existing: {db_columns:?}, Provided: {provided_columns:?}"
		);
	}

	Ok((conn, db_columns))
}


fn shard_path<P: AsRef<Path>>(path: P, shard_num: u16) -> PathBuf {
	path.as_ref().join(format!("shard_{shard_num:04}.bin"))
}


impl ToSql for ColumnEncoding {
	fn to_sql(&self) -> Result<rusqlite::types::ToSqlOutput<'_>, rusqlite::Error> {
		let s = self.to_str();
		Ok(rusqlite::types::ToSqlOutput::from(s))
	}
}

impl FromSql for ColumnEncoding {
	fn column_result(value: ValueRef<'_>) -> Result<Self, rusqlite::types::FromSqlError> {
		let s = value.as_str()?;
		ColumnEncoding::from_str(s).ok_or_else(|| rusqlite::types::FromSqlError::Other(format!("Invalid column encoding: {s}").into()))
	}
}


#[cfg(test)]
mod tests {
	use super::HeavyData;
	use crate::{
		encoding::ColumnEncoding,
		heavy_data::{create_or_resume_index, shard_path},
	};
	use pyo3::{
		Python,
		types::{PyAnyMethods, PyBytes, PyDict, PySetMethods},
	};
	use rand::RngExt as _;
	use rusqlite::params;
	use std::collections::HashMap;
	use tempfile::tempdir;
	use xxhash_rust::xxh3::xxh3_128;

	#[derive(Debug, Clone)]
	struct Expected {
		txt: String,
		n: u32,
		small: i16,
		blob: Vec<u8>,
		flt: f32,
	}

	#[test]
	fn heavy_data_roundtrip_integration() {
		Python::attach(|py| {
			let tmp = tempdir().expect("create tempdir");
			let store_dir = tmp.path().join("heavydata");
			let store_dir_str = store_dir.to_str().expect("tempdir path must be utf-8");

			// Schema
			let mut columns = HashMap::new();
			columns.insert("txt".to_string(), ColumnEncoding::Str);
			columns.insert("n".to_string(), ColumnEncoding::Uint32);
			columns.insert("small".to_string(), ColumnEncoding::Int16);
			columns.insert("blob".to_string(), ColumnEncoding::Bytes);
			columns.insert("flt".to_string(), ColumnEncoding::Float32);

			// Create store + write random data
			let mut hd = HeavyData::new(store_dir_str, Some(columns.clone()), 3).expect("create HeavyData");
			let n_samples = 200usize;
			let mut cases = Vec::with_capacity(n_samples);

			for i in 0..n_samples {
				let key: [u8; 32] = rand::random();
				let txt = format!("row{i}-{:016x}", rand::random::<u64>());
				let n: u32 = rand::random();
				let small: i16 = rand::random();
				let flt: f32 = rand::random();

				let blob_len = rand::random_range(0..32768);
				let mut blob = vec![0u8; blob_len];
				rand::rng().fill(&mut blob[..]);

				let d = PyDict::new(py);
				d.set_item("txt", &txt).unwrap();
				d.set_item("n", n).unwrap();
				d.set_item("small", small).unwrap();
				d.set_item("blob", PyBytes::new(py, &blob)).unwrap();
				d.set_item("flt", flt).unwrap();

				hd.write(&key, d, py).expect("write sample");
				cases.push((key, Expected { txt, n, small, blob, flt }));
			}

			// Ensure the background worker flushes everything.
			hd.finish(py).expect("finish HeavyData");
			drop(hd);

			// Re-open the DB/store from disk and read everything back.
			let mut hd = HeavyData::new(store_dir_str, Some(columns), 3).expect("reopen HeavyData");

			// Validate existing_keys contains everything (and count matches)
			let existing = hd.existing_keys(py).expect("existing_keys");
			assert_eq!(existing.len(), n_samples, "unexpected number of keys in index");

			for (key, _) in &cases {
				assert!(existing.contains(PyBytes::new(py, key)).unwrap(), "missing key in existing_keys");
			}

			// Validate round-trip on read()
			for (key, exp) in &cases {
				let got = hd.read(key, py).unwrap().expect("key must exist");

				let got_txt: String = got.get_item("txt").unwrap().extract().unwrap();
				assert_eq!(got_txt, exp.txt);

				let got_n: u32 = got.get_item("n").unwrap().call_method0("item").unwrap().extract().unwrap();
				assert_eq!(got_n, exp.n);

				let got_small: i16 = got.get_item("small").unwrap().call_method0("item").unwrap().extract().unwrap();
				assert_eq!(got_small, exp.small);

				let got_blob: Vec<u8> = got.get_item("blob").unwrap().extract::<&[u8]>().unwrap().to_vec();
				assert_eq!(got_blob, exp.blob);

				let got_flt: f32 = got.get_item("flt").unwrap().call_method0("item").unwrap().extract().unwrap();
				assert!((got_flt - exp.flt).abs() <= 1e-6, "flt mismatch: got={got_flt} expected={}", exp.flt);
			}

			// Missing key should return None
			let missing = vec![0u8; 32];
			assert!(hd.read(&missing, py).unwrap().is_none());

			hd.finish(py).expect("finish reopened HeavyData");
		});
	}

	/// `create_or_resume_index` should enforce that the on-disk schema matches the
	/// requested schema on subsequent opens.
	#[test]
	fn create_or_resume_index_schema_mismatch_fails() {
		let tmp = tempdir().unwrap();
		let path = tmp.path();

		let cols1 = HashMap::from([("a".to_string(), ColumnEncoding::Str)]);
		let _conn = create_or_resume_index(path, Some(&cols1)).expect("first create_or_resume_index should succeed");

		// Same column name but different encoding
		let cols2 = HashMap::from([("a".to_string(), ColumnEncoding::Uint32)]);
		let err = create_or_resume_index(path, Some(&cols2)).expect_err("schema mismatch must error");
		let msg = format!("{err:#}");
		assert!(
			msg.contains("Existing index schema does not match the provided schema"),
			"unexpected error: {msg}"
		);
	}

	/// read_sample_data: detects checksum mismatches when the payload and checksum
	/// don't agree.
	#[test]
	fn read_sample_data_detects_checksum_mismatch() {
		let tmp = tempdir().unwrap();
		let path = tmp.path();

		let columns = HashMap::from([("txt".to_string(), ColumnEncoding::Str)]);
		let key = b"abcd-efgh-ijkl-mnop".to_vec();

		let (conn, _columns) = create_or_resume_index(path, Some(&columns)).unwrap();

		// Build a well-formed payload (key + bytes + *wrong* checksum).
		let mut sample_bytes = Vec::new();
		sample_bytes.extend_from_slice(&key);
		sample_bytes.extend_from_slice(b"payload");
		let good = (xxh3_128(&sample_bytes) & 0xFFFF_FFFF) as u32;
		let bad = good.wrapping_add(1); // ensure mismatch
		sample_bytes.extend_from_slice(&bad.to_le_bytes());

		let compressed = zstd::encode_all(&sample_bytes[..], 1).unwrap();
		std::fs::write(shard_path(path, 0), &compressed).unwrap();

		conn.execute(
			"INSERT INTO index_table (key, shard_num, position, length) VALUES (?1, ?2, ?3, ?4)",
			params![&key[..], 0i64, 0i64, compressed.len() as i64],
		)
		.unwrap();

		let hd = HeavyData::new(path.to_str().unwrap(), None, 1).unwrap();

		let err = hd.read_sample_data(&key).unwrap_err();
		let msg = format!("{err:#}");
		assert!(
			msg.contains("Checksum mismatch for sample data"),
			"expected checksum mismatch error, got: {msg}"
		);
	}

	/// read_sample_data: detects when the stored key prefix doesn't match the key
	/// that was looked up, even if checksum is correct.
	#[test]
	fn read_sample_data_detects_key_mismatch() {
		let tmp = tempdir().unwrap();
		let path = tmp.path();

		let columns = HashMap::from([("txt".to_string(), ColumnEncoding::Str)]);
		let key = b"original-key-1234".to_vec();
		let on_disk_key = b"otherxxx-key-5678".to_vec();
		assert_eq!(key.len(), on_disk_key.len(), "keys must be same length for this test");

		let (conn, _columns) = create_or_resume_index(path, Some(&columns)).unwrap();

		let mut sample_bytes = Vec::new();
		sample_bytes.extend_from_slice(&on_disk_key);
		sample_bytes.extend_from_slice(b"payload");
		let checksum = (xxh3_128(&sample_bytes) & 0xFFFF_FFFF) as u32;
		sample_bytes.extend_from_slice(&checksum.to_le_bytes());

		let compressed = zstd::encode_all(&sample_bytes[..], 1).unwrap();
		std::fs::write(shard_path(path, 0), &compressed).unwrap();

		// Index points to this shard using the *original* key.
		conn.execute(
			"INSERT INTO index_table (key, shard_num, position, length) VALUES (?1, ?2, ?3, ?4)",
			params![&key[..], 0i64, 0i64, compressed.len() as i64],
		)
		.unwrap();

		let hd = HeavyData::new(path.to_str().unwrap(), None, 1).unwrap();

		let err = hd.read_sample_data(&key).unwrap_err();
		let msg = format!("{err:#}");
		assert!(msg.contains("Key mismatch for sample data"), "expected key mismatch error, got: {msg}");
	}
}
