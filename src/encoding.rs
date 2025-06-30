use std::{
	collections::{HashMap, VecDeque},
	fs,
	io::{BufWriter, Read, Write},
	path::PathBuf,
	sync::{Arc, Mutex, mpsc},
	thread::{self, JoinHandle},
};

use anyhow::Context;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use flate2::{Compression, write::GzEncoder};
use pyo3::{
	Bound, PyAny, PyResult, Python,
	exceptions::{PyIOError, PyValueError},
	pyclass, pymethods,
	types::{PyAnyMethods, PyBytes, PyBytesMethods, PyDict, PyString},
};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_128;


#[pyclass]
pub struct SampleWriter {
	out_dir: PathBuf,
	compress: bool,
	n_written: u64,
	columns: Vec<(String, ColumnEncoding)>,
	workers: Vec<JoinHandle<()>>,
	work_queue: Arc<Mutex<VecDeque<Option<SampleWriterJob>>>>,
	result_rx: Mutex<mpsc::Receiver<anyhow::Error>>,
}

#[pymethods]
impl SampleWriter {
	#[new]
	fn new(out_dir: PathBuf, compress: bool, columns: HashMap<String, ColumnEncoding>, n_workers: usize) -> Self {
		let (result_tx, result_rx) = mpsc::channel();
		let work_queue = Arc::new(Mutex::new(VecDeque::new()));
		let mut workers = Vec::with_capacity(n_workers);

		for _ in 0..n_workers {
			let work_queue_clone = work_queue.clone();
			let result_tx_clone = result_tx.clone();
			let out_dir_clone = out_dir.clone();

			let worker = thread::spawn(move || {
				if let Err(e) = sample_writer_worker(work_queue_clone, out_dir_clone, compress) {
					result_tx_clone.send(e).expect("Failed to send error from worker");
				}
			});
			workers.push(worker);
		}

		// Sort the columns by name to ensure consistent order
		let mut columns: Vec<_> = columns.into_iter().collect();
		columns.sort_by(|a, b| a.0.cmp(&b.0));

		SampleWriter {
			out_dir,
			compress,
			n_written: 0,
			columns,
			workers,
			work_queue,
			result_rx: Mutex::new(result_rx),
		}
	}

	fn write<'py>(&mut self, sample: Bound<'py, PyDict>) -> PyResult<()> {
		let mut sample_values = Vec::with_capacity(self.columns.len());

		for (column_name, encoding) in &self.columns {
			let value = sample.get_item(column_name)?;
			let column_value = ColumnValue::from_python(value, encoding)?;
			sample_values.push(column_value);
		}

		let job = SampleWriterJob {
			sample: sample_values,
			n: self.n_written,
		};

		self.n_written += 1;
		self.work_queue
			.lock()
			.map_err(|e| PyIOError::new_err(format!("Failed to lock work queue: {e:?}")))?
			.push_back(Some(job));

		// Check for errors from workers
		let result_rx = self
			.result_rx
			.lock()
			.map_err(|e| PyIOError::new_err(format!("Failed to lock result receiver: {e:?}")))?;
		if let Ok(err) = result_rx.try_recv() {
			return Err(PyValueError::new_err(format!("Worker error: {err:?}")));
		}

		Ok(())
	}

	fn finish(&mut self) -> PyResult<()> {
		// Signal workers to finish
		for _ in 0..self.workers.len() {
			self.work_queue
				.lock()
				.map_err(|e| PyIOError::new_err(format!("Failed to lock work queue: {e:?}")))?
				.push_back(None);
		}

		// Wait for all workers to finish
		for worker in self.workers.drain(..) {
			worker.join().map_err(|_| PyValueError::new_err("Worker thread panicked"))?;
		}

		// Check for errors from workers
		let result_rx = self
			.result_rx
			.lock()
			.map_err(|e| PyIOError::new_err(format!("Failed to lock result receiver: {e:?}")))?;
		if let Ok(err) = result_rx.try_recv() {
			return Err(PyValueError::new_err(format!("Worker error: {err:?}")));
		}

		// Write index file
		self.write_index().map_err(|e| PyValueError::new_err(format!("Failed to write index: {e:?}")))?;

		Ok(())
	}
}

impl SampleWriter {
	fn write_index(&self) -> anyhow::Result<()> {
		let index_path = self.out_dir.join("index.json");
		let index_json = IndexJson {
			version: 1,
			column_encodings: self.columns.iter().map(|(_, encoding)| encoding.clone()).collect(),
			column_names: self.columns.iter().map(|(name, _)| name.clone()).collect(),
			compression: if self.compress { Some("gzip".to_string()) } else { None },
			samples: self.n_written,
		};

		let index_data = serde_json::to_string_pretty(&index_json).context("Failed to serialize index JSON")?;
		fs::write(&index_path, index_data).with_context(|| format!("Failed to write index JSON to {}", index_path.display()))?;
		Ok(())
	}
}


struct SampleWriterJob {
	sample: Vec<ColumnValue>,
	n: u64,
}

fn sample_writer_worker(work_queue: Arc<Mutex<VecDeque<Option<SampleWriterJob>>>>, out_dir: PathBuf, compress: bool) -> anyhow::Result<()> {
	loop {
		let sample = {
			let mut queue = work_queue.lock().map_err(|e| anyhow::anyhow!("Failed to lock work queue: {}", e))?;
			match queue.pop_front() {
				Some(Some(job)) => job,
				Some(None) => break, // Signal to stop processing
				None => {
					drop(queue); // Release the lock before sleeping
					thread::sleep(std::time::Duration::from_millis(1)); // Wait for new work
					continue;
				},
			}
		};

		let path = out_dir.join(sample_index_to_path(sample.n, compress));
		let mut buf = vec![0u8; 16]; // Reserve space for the hash

		if compress {
			let mut writer = GzEncoder::new(&mut buf, Compression::default());

			for column in &sample.sample {
				column.encode(&mut writer).context("Failed to encode column value")?;
			}

			writer.finish().context("Failed to finish compression")?;
		} else {
			let mut writer = BufWriter::new(&mut buf);

			for column in &sample.sample {
				column.encode(&mut writer).context("Failed to encode column value")?;
			}

			writer.flush().context("Failed to flush writer")?;
		};

		let hash = xxh3_128(&buf[16..]); // Skip the reserved space for hash
		buf[..16].copy_from_slice(&hash.to_le_bytes());

		fs::create_dir_all(path.parent().unwrap()).with_context(|| format!("Failed to create directory for {}", path.display()))?;
		fs::write(&path, buf).with_context(|| format!("Failed to write sample to {}", path.display()))?;
	}

	Ok(())
}


pub fn sample_index_to_path(sample_index: u64, compress: bool) -> PathBuf {
	let dir_a = sample_index & 0xFF;
	let dir_b = (sample_index >> 8) & 0xFF;
	let mut path = PathBuf::from(dir_a.to_string()).join(dir_b.to_string()).join(format!("{sample_index}.bin"));
	if compress {
		path.set_extension("bin.gz");
	}
	path
}


#[derive(Serialize, Deserialize, Debug)]
pub struct IndexJson {
	pub version: u32,
	pub column_encodings: Vec<ColumnEncoding>,
	pub column_names: Vec<String>,
	pub compression: Option<String>,
	pub samples: u64,
}


#[pyclass]
#[derive(Deserialize, Debug, Clone, Serialize)]
pub enum ColumnEncoding {
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
}

impl ColumnEncoding {
	pub fn decode_to_python<'py, R: Read>(&self, py: Python<'py>, mut reader: R, np_frombuffer: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
		match self {
			ColumnEncoding::Str => {
				let len = reader
					.read_u32::<LittleEndian>()
					.map_err(|e| PyIOError::new_err(format!("Failed to read string length: {e:?}")))?;
				let mut buf = vec![0; len as usize];
				reader.read_exact(&mut buf)?;
				let value = String::from_utf8(buf).map_err(|e| PyValueError::new_err(format!("Failed to decode UTF-8 string: {e:?}")))?;
				Ok(PyString::new(py, &value).into_any())
			},
			ColumnEncoding::Int8 => numpy_frombuffer_scalar(np_frombuffer, "<i1", &mut reader, 1),
			ColumnEncoding::Int16 => numpy_frombuffer_scalar(np_frombuffer, "<i2", &mut reader, 2),
			ColumnEncoding::Int32 => numpy_frombuffer_scalar(np_frombuffer, "<i4", &mut reader, 4),
			ColumnEncoding::Int64 => numpy_frombuffer_scalar(np_frombuffer, "<i8", &mut reader, 8),
			ColumnEncoding::Uint8 => numpy_frombuffer_scalar(np_frombuffer, "<u1", &mut reader, 1),
			ColumnEncoding::Uint16 => numpy_frombuffer_scalar(np_frombuffer, "<u2", &mut reader, 2),
			ColumnEncoding::Uint32 => numpy_frombuffer_scalar(np_frombuffer, "<u4", &mut reader, 4),
			ColumnEncoding::Uint64 => numpy_frombuffer_scalar(np_frombuffer, "<u8", &mut reader, 8),
			ColumnEncoding::Bytes => {
				let len = reader
					.read_u32::<LittleEndian>()
					.map_err(|e| PyIOError::new_err(format!("Failed to read bytes length: {e:?}")))?;
				let pybytes = PyBytes::new_with(py, len as usize, |slice| {
					reader.read_exact(slice).map_err(|e| PyIOError::new_err(format!("Failed to read bytes: {e:?}")))
				})?;
				//let mut buf = vec![0; len as usize];
				//reader.read_exact(&mut buf)?;
				Ok(pybytes.into_any())
			},
			ColumnEncoding::Float16 => numpy_frombuffer_scalar(np_frombuffer, "<f2", &mut reader, 2),
			ColumnEncoding::Float32 => numpy_frombuffer_scalar(np_frombuffer, "<f4", &mut reader, 4),
			ColumnEncoding::Float64 => numpy_frombuffer_scalar(np_frombuffer, "<f8", &mut reader, 8),
		}
	}
}


pub enum ColumnValue {
	Str(String),
	Int8(i8),
	Int16(i16),
	Int32(i32),
	Int64(i64),
	Uint8(u8),
	Uint16(u16),
	Uint32(u32),
	Uint64(u64),
	Bytes(Vec<u8>),
	Float32(f32),
	Float64(f64),
}

impl ColumnValue {
	pub fn from_python<'py>(value: Bound<'py, PyAny>, encoding: &ColumnEncoding) -> PyResult<Self> {
		match encoding {
			ColumnEncoding::Str => Ok(ColumnValue::Str(value.extract()?)),
			ColumnEncoding::Int8 => Ok(ColumnValue::Int8(value.extract()?)),
			ColumnEncoding::Int16 => Ok(ColumnValue::Int16(value.extract()?)),
			ColumnEncoding::Int32 => Ok(ColumnValue::Int32(value.extract()?)),
			ColumnEncoding::Int64 => Ok(ColumnValue::Int64(value.extract()?)),
			ColumnEncoding::Uint8 => Ok(ColumnValue::Uint8(value.extract()?)),
			ColumnEncoding::Uint16 => Ok(ColumnValue::Uint16(value.extract()?)),
			ColumnEncoding::Uint32 => Ok(ColumnValue::Uint32(value.extract()?)),
			ColumnEncoding::Uint64 => Ok(ColumnValue::Uint64(value.extract()?)),
			ColumnEncoding::Bytes => Ok(ColumnValue::Bytes(value.downcast::<PyBytes>()?.as_bytes().to_vec())),
			ColumnEncoding::Float32 => Ok(ColumnValue::Float32(value.extract()?)),
			ColumnEncoding::Float64 => Ok(ColumnValue::Float64(value.extract()?)),
			_ => Err(PyValueError::new_err("Unsupported column encoding for value extraction")),
		}
	}

	pub fn encode<W: Write>(&self, writer: &mut W) -> anyhow::Result<()> {
		match self {
			ColumnValue::Str(s) => {
				let bytes = s.as_bytes();
				writer.write_u32::<LittleEndian>(bytes.len().try_into().context("String length exceeds u32 limit")?)?;
				writer.write_all(bytes)?;
			},
			ColumnValue::Int8(v) => writer.write_i8(*v)?,
			ColumnValue::Int16(v) => writer.write_i16::<LittleEndian>(*v)?,
			ColumnValue::Int32(v) => writer.write_i32::<LittleEndian>(*v)?,
			ColumnValue::Int64(v) => writer.write_i64::<LittleEndian>(*v)?,
			ColumnValue::Uint8(v) => writer.write_u8(*v)?,
			ColumnValue::Uint16(v) => writer.write_u16::<LittleEndian>(*v)?,
			ColumnValue::Uint32(v) => writer.write_u32::<LittleEndian>(*v)?,
			ColumnValue::Uint64(v) => writer.write_u64::<LittleEndian>(*v)?,
			ColumnValue::Bytes(bytes) => {
				writer.write_u32::<LittleEndian>(bytes.len().try_into().map_err(|_| PyValueError::new_err("Bytes length exceeds u32 limit"))?)?;
				writer.write_all(bytes)?;
			},
			ColumnValue::Float32(v) => writer.write_f32::<LittleEndian>(*v)?,
			ColumnValue::Float64(v) => writer.write_f64::<LittleEndian>(*v)?,
		}
		Ok(())
	}
}


fn numpy_frombuffer_scalar<'py, R: Read>(np_frombuffer: &Bound<'py, PyAny>, dtype: &str, reader: &mut R, size: usize) -> PyResult<Bound<'py, PyAny>> {
	assert!(size > 0 && size <= 8, "Size must be between 1 and 8 bytes for scalar types");
	let mut buf = [0u8; 8]; // max size is 8 bytes for scalar types
	reader
		.read_exact(&mut buf[..size])
		.map_err(|e| PyIOError::new_err(format!("Failed to read {size} bytes: {e:?}")))?;
	let array = np_frombuffer.call1((buf, dtype))?;
	array.get_item(0)
}


pub fn decode_sample<'py, R: Read>(py: Python<'py>, mut reader: R, columns: &[(String, ColumnEncoding)]) -> PyResult<Bound<'py, PyDict>> {
	let sample = PyDict::new(py);

	// Prep numpy
	let np = py.import("numpy")?;
	let np_frombuffer = np.getattr("frombuffer")?;

	for (column_name, encoding) in columns {
		let value = encoding
			.decode_to_python(py, &mut reader, &np_frombuffer)
			.map_err(|e| PyValueError::new_err(format!("Failed to decode column '{column_name}': {e:?}")))?;
		sample.set_item(column_name, value)?;
	}

	Ok(sample)
}


#[cfg(test)]
mod tests {
	use std::io::Cursor;

	use super::*;
	use pyo3::prelude::*;
	use tempfile::tempdir;

	/// Helper: call `decode_to_python` and return the raw `PyAny`
	fn decode<'py>(py: Python<'py>, enc: ColumnEncoding, bytes: Vec<u8>) -> Bound<'py, PyAny> {
		let np = py.import("numpy").expect("`numpy` not found – install it before running tests");
		let frombuffer = np.getattr("frombuffer").unwrap();
		enc.decode_to_python(py, Cursor::new(bytes), &frombuffer).expect("decode failed")
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
			let mut buf = Vec::new();
			ColumnValue::Str(text.to_string()).encode(&mut buf).unwrap();
			let obj = decode(py, ColumnEncoding::Str, buf);
			assert_eq!(obj.extract::<&str>().unwrap(), text);
		});
	}

	#[test]
	fn bytes_roundtrip() {
		Python::with_gil(|py| {
			let data = b"\x00\xff\x10\x20".to_vec();
			let mut buf = Vec::new();
			ColumnValue::Bytes(data.clone()).encode(&mut buf).unwrap();
			let obj = decode(py, ColumnEncoding::Bytes, buf);
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
	numeric_test!(uint64_scalar, ColumnEncoding::Uint64, u64, 18_446_744_073_709_551_615u64);
	numeric_test!(float32_scalar, ColumnEncoding::Float32, f32, -1_234.567_8_f32);
	numeric_test!(float64_scalar, ColumnEncoding::Float64, f64, 8.901234567890123e55_f64);

	#[test]
	fn insufficient_buffer_returns_error() {
		Python::with_gil(|py| {
			let np = py.import("numpy").unwrap();
			let frombuffer = np.getattr("frombuffer").unwrap();

			let buf = vec![0_u8; 2]; // Int32 needs 4 bytes
			let res = ColumnEncoding::Int32.decode_to_python(py, Cursor::new(buf), &frombuffer);
			assert!(res.is_err());
		});
	}

	#[test]
	fn sample_index_to_path_layout() {
		// index          dir_A dir_B  filename
		// 0x0102_03EA -> 0xEA   0x03   0x010203EA
		let p = sample_index_to_path(0x0102_03EA, false);
		assert_eq!(p, PathBuf::from("234").join("3").join("16909290.bin"));
		let gz = sample_index_to_path(0x0102_03EA, true);
		assert!(gz.ends_with("16909290.bin.gz"));
	}

	#[test]
	fn float16_roundtrip() {
		Python::with_gil(|py| {
			// IEEE‑754 half‑precision 1.0  → 0x3c00 (little endian)
			let half_1_0 = vec![0x00, 0x3c];
			let np = py.import("numpy").unwrap();
			let obj = ColumnEncoding::Float16
				.decode_to_python(py, Cursor::new(half_1_0), &np.getattr("frombuffer").unwrap())
				.unwrap();
			let v: f32 = obj.call_method0("item").unwrap().extract().unwrap();
			assert!((v - 1.0).abs() < 1e-3, "expected 1.0 from float16 decode");
		});
	}

	macro_rules! encode_roundtrip {
		($name:ident, $val:expr, $enc:ident, $ty:ty) => {
			#[test]
			fn $name() {
				Python::with_gil(|py| {
					let value: $ty = $val;

					// encode ---------------------------------------------------
					let mut buf = Vec::new();
					ColumnValue::$enc(value).encode(&mut buf).unwrap();

					// decode ---------------------------------------------------
					let np = py.import("numpy").unwrap();
					let frombuffer = np.getattr("frombuffer").unwrap();
					let decoded = ColumnEncoding::$enc.decode_to_python(py, Cursor::new(buf), &frombuffer).unwrap();

					// compare --------------------------------------------------
					let round: $ty = decoded.call_method0("item").unwrap().extract().unwrap();
					assert_eq!(round, value);
				});
			}
		};
	}

	// Only numeric types require extra coverage (string/bytes handled elsewhere)
	encode_roundtrip!(int8_cv, -8i8, Int8, i8);
	encode_roundtrip!(uint8_cv, 250u8, Uint8, u8);
	encode_roundtrip!(int16_cv, -12345i16, Int16, i16);
	encode_roundtrip!(uint16_cv, 54321u16, Uint16, u16);
	encode_roundtrip!(int32_cv, -2_000_000i32, Int32, i32);
	encode_roundtrip!(uint32_cv, 3_900_000u32, Uint32, u32);
	encode_roundtrip!(int64_cv, -9_876_543_210i64, Int64, i64);
	encode_roundtrip!(uint64_cv, 12_345_678_901_234_567u64, Uint64, u64);
	encode_roundtrip!(float32_cv, -std::f32::consts::PI, Float32, f32);
	encode_roundtrip!(float64_cv, std::f64::consts::E, Float64, f64);

	#[test]
	fn decode_sample_mixed_columns() {
		Python::with_gil(|py| {
			let columns = vec![
				("txt".to_owned(), ColumnEncoding::Str),
				("num".to_owned(), ColumnEncoding::Int16),
				("bin".to_owned(), ColumnEncoding::Bytes),
				("flt".to_owned(), ColumnEncoding::Float32),
			];

			// build raw buffer via ColumnValue::encode
			let mut buf = Vec::new();
			ColumnValue::Str("hi".into()).encode(&mut buf).unwrap();
			ColumnValue::Int16(-123).encode(&mut buf).unwrap();
			ColumnValue::Bytes(vec![1, 2, 3]).encode(&mut buf).unwrap();
			ColumnValue::Float32(0.5).encode(&mut buf).unwrap();

			let sample = decode_sample(py, Cursor::new(buf), &columns).unwrap();

			assert_eq!(sample.get_item("txt").unwrap().unwrap().extract::<&str>().unwrap(), "hi");
			assert_eq!(
				sample.get_item("num").unwrap().unwrap().call_method0("item").unwrap().extract::<i16>().unwrap(),
				-123
			);
			assert_eq!(sample.get_item("bin").unwrap().unwrap().extract::<&[u8]>().unwrap(), &[1, 2, 3]);
			assert!((sample.get_item("flt").unwrap().unwrap().call_method0("item").unwrap().extract::<f32>().unwrap() - 0.5).abs() < 1e-6);
		});
	}

	fn exercise_sample_writer(compress: bool) {
		Python::with_gil(|py| {
			let tmp = tempdir().unwrap();
			let mut cols = HashMap::new();
			cols.insert("txt".into(), ColumnEncoding::Str);
			cols.insert("num".into(), ColumnEncoding::Int32);

			let mut writer = SampleWriter::new(tmp.path().into(), compress, cols.clone(), 2);

			for n in 0..3 {
				let d = PyDict::new(py);
				d.set_item("txt", format!("row{n}")).unwrap();
				d.set_item("num", n).unwrap();
				writer.write(d).unwrap();
			}
			writer.finish().unwrap();

			// index.json exists and matches
			let idx: IndexJson = serde_json::from_str(&std::fs::read_to_string(tmp.path().join("index.json")).unwrap()).unwrap();
			assert_eq!(idx.samples, 3);
			assert_eq!(idx.compression.is_some(), compress);

			// load first sample
			let p = tmp.path().join(sample_index_to_path(0, compress));
			let raw = std::fs::read(p).unwrap();
			let payload = if compress {
				use flate2::read::GzDecoder;
				let mut dec = GzDecoder::new(&raw[16..]);
				let mut out = Vec::new();
				dec.read_to_end(&mut out).unwrap();
				out
			} else {
				raw[16..].to_vec()
			};

			let cols_vec: Vec<_> = idx
				.column_names
				.iter()
				.zip(idx.column_encodings.iter())
				.map(|(name, enc)| (name.clone(), enc.clone()))
				.collect();
			let decoded = decode_sample(py, Cursor::new(payload), &cols_vec).unwrap();
			assert_eq!(decoded.get_item("txt").unwrap().unwrap().extract::<&str>().unwrap(), "row0");
			assert_eq!(
				decoded
					.get_item("num")
					.unwrap()
					.unwrap()
					.call_method0("item")
					.unwrap()
					.extract::<i32>()
					.unwrap(),
				0
			);
		});
	}

	#[test]
	fn sample_writer_plain() {
		exercise_sample_writer(false);
	}
	#[test]
	fn sample_writer_gzip() {
		exercise_sample_writer(true);
	}
}
