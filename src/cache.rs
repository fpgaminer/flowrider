use anyhow::{Context, ensure};
use async_recursion::async_recursion;
use log::{info, warn};
use moka::future::{Cache, FutureExt};
use std::{
	path::{Path, PathBuf},
	sync::Arc,
	time::Instant,
};
use tokio::sync::Semaphore;
use url::Url;


pub struct ShardMeta {
	bytes: u32,
	remote: Option<Url>,
}


#[derive(Clone)]
pub struct ShardCache {
	cache: Cache<String, Arc<ShardMeta>>,
	cache_dir: PathBuf,
}

impl ShardCache {
	pub async fn new(cache_limit: u64, cache_dir: &str) -> ShardCache {
		let cache_dir = PathBuf::from(cache_dir);

		let mut cache = Cache::builder()
			.weigher(|_: &String, meta: &Arc<ShardMeta>| meta.bytes)
			.async_eviction_listener(|key, _meta, _cause| {
				async move {
					if Path::new(&*key).file_name().is_some_and(|name| name == "index.json") {
						// do not remove index files from the cache
						return;
					}

					if let Err(err) = tokio::fs::remove_file(Path::new(&*key)).await {
						warn!("Cache failed to remove file {}: {}", key, err);
					}
					info!("Cache removed file {}", key);
				}
				.boxed()
			});

		if cache_limit > 0 {
			cache = cache.max_capacity(cache_limit);
		}

		let cache = cache.build();

		let this = ShardCache { cache, cache_dir };

		// find existing shards in the cache directory and pre-populate the cache
		info!("Populating shard cache from {}", this.cache_dir.display());
		this.populate_cache(&this.cache_dir).await;
		info!("Shard cache populated");

		this
	}

	#[async_recursion]
	async fn populate_cache(&self, path: &Path) {
		if !path.exists() {
			return;
		}

		let Ok(mut entries) = tokio::fs::read_dir(path).await.inspect_err(|e| {
			warn!("Failed to read directory {}: {:?}", path.display(), e);
		}) else {
			return;
		};

		while let Some(entry) = match entries.next_entry().await {
			Ok(e) => e,
			Err(e) => {
				warn!("Failed to read entry in directory {}: {:?}", path.display(), e);
				return;
			},
		} {
			let path = entry.path();
			if path.is_file() && path.extension().is_some_and(|ext| ext == "mds") {
				let Ok(local_path) = path.canonicalize() else {
					warn!("Failed to canonicalize path {:?}. Skipping.", path.display());
					continue;
				};
				let Some(local) = local_path.to_str() else {
					warn!("Path {} is not valid UTF-8. Skipping.", local_path.display());
					continue;
				};

				let Ok(metadata) = tokio::fs::metadata(&path).await else {
					warn!("Failed to get metadata for path {}. Skipping.", path.display());
					continue;
				};

				let meta = Arc::new(ShardMeta {
					bytes: metadata.len().try_into().unwrap_or(u32::MAX),
					remote: None,
				});
				self.cache.insert(local.to_string(), meta).await;
			} else if path.is_dir() {
				self.populate_cache(&path).await;
			}
		}
	}

	pub async fn get_shard(&self, remote: Url, local: &str, expected_hash: Option<u128>, download_semaphore: &Semaphore) -> anyhow::Result<Arc<ShardMeta>> {
		// local path must be valid
		// since local paths cannot have traversal components, we guarantee they can be used as unique keys in the cache
		// (this assumes no symlinked directories or other tricks in the cache directory)
		ensure!(
			is_local_path_valid(local),
			"Local path '{}' is not valid. It must be a relative path without traversal components, must have a file name, and must not be empty.",
			local
		);

		// Check for footgun
		if self.cache_dir.components().zip(Path::new(local).components()).all(|(a, b)| a == b) {
			return Err(anyhow::anyhow!(
				"A shard was requested with local path '{}', but that starts with the cache directory '{}'. This is likely a mistake, and could mean something is broken with this code. Please report this issue.",
				local,
				self.cache_dir.display()
			));
		}

		let local_cache_path = self.cache_dir.join(local);

		// check for and avoid a footgun:
		// if the user uses a remote file:// URL, but sets the cache directory to the same, then cache reaping would delete the original dataset
		if remote.scheme() == "file" {
			let remote_path = remote
				.to_file_path()
				.map_err(|_| anyhow::anyhow!("Remote URL '{}' is not a valid file path", remote))?
				.parent()
				.ok_or_else(|| anyhow::anyhow!("Remote URL '{}' does not have a parent directory", remote))?
				.canonicalize()
				.with_context(|| format!("Failed to canonicalize remote path: {}", remote))?;
			let local_path = local_cache_path
				.parent()
				.ok_or_else(|| anyhow::anyhow!("Local cache path '{}' does not have a parent directory", local_cache_path.display()))?;

			if let Ok(local_path) = local_path.canonicalize() {
				ensure!(
					remote_path != local_path,
					"Remote path '{}' must not be the same as local cache path '{}'. This would cause the original dataset to be deleted when the cache evicts.",
					remote_path.display(),
					local_cache_path.display()
				);
			}
		}

		let local_cache_path = local_cache_path
			.to_str()
			.ok_or_else(|| anyhow::anyhow!("Local cache path '{}' is not valid UTF-8", local_cache_path.display()))?;

		// If the shard is in the cache, we can return immediately.
		// Otherwise, moka's Cache ensures that only a single instance of download_shard will run concurrently for the same key.
		// Once the download is complete, it will be cached and we (and all other waiting tasks) can return.
		match self
			.cache
			.try_get_with_by_ref(local, download_shard(&remote, local_cache_path, expected_hash, download_semaphore))
			.await
		{
			Ok(meta) => {
				if let Some(meta_remote) = &meta.remote {
					ensure!(
						meta_remote == &remote,
						"Cached shard at {} has different remote URL than requested: {} != {}",
						local,
						meta_remote,
						remote
					);
				}
				info!("Using cached shard at {}", local);
				Ok(meta)
			},
			Err(e) => Err(anyhow::anyhow!("Failed to get shard {}: {}", local, e)),
		}
	}
}


fn is_local_path_valid(path: &str) -> bool {
	if path.ends_with('/') {
		// trailing slashes are not allowed (since a filename is required)
		return false;
	}

	let path = Path::new(path);

	// must be a relative path
	if !path.is_relative() {
		return false;
	}

	// must not contain any path traversal components
	if path
		.components()
		.any(|c| c == std::path::Component::ParentDir || c == std::path::Component::CurDir)
	{
		return false;
	}

	// must have a file name
	if path.file_name().is_none() {
		return false;
	}

	// must not be an empty path
	if path.as_os_str().is_empty() {
		return false;
	}

	// must not contain any invalid characters
	if path.to_str().is_none() {
		return false;
	}

	true
}


async fn download_shard(remote: &Url, local: &str, expected_hash: Option<u128>, download_semaphore: &Semaphore) -> anyhow::Result<Arc<ShardMeta>> {
	let start = Instant::now();
	crate::server::download_file(remote, local, expected_hash, download_semaphore).await?;

	let bytes = tokio::fs::metadata(local)
		.await
		.context(format!("Failed to get metadata for shard at {}", local))?
		.len()
		.try_into()
		.context(format!("Shard at {} is too large", local))?;
	let meta = Arc::new(ShardMeta {
		bytes,
		remote: Some(remote.clone()),
	});

	let elapsed = start.elapsed();
	info!("Downloaded shard {} in {:?}", local, elapsed);

	Ok(meta)
}


#[cfg(test)]
mod tests {
	use super::*;
	use std::time::Duration;
	use tokio::time::sleep;

	#[test]
	fn test_is_local_path_valid() {
		assert!(is_local_path_valid("shard.mds"));
		assert!(is_local_path_valid("index.html"));
		assert!(is_local_path_valid("dir/file.txt"));
		assert!(is_local_path_valid("subdir/nested/file.json"));
		assert!(is_local_path_valid("cache/shard001.mds"));
		assert!(is_local_path_valid("data/train/batch01.parquet"));
		assert!(is_local_path_valid(".hidden"));
		assert!(is_local_path_valid("dir/.hidden"));
		assert!(is_local_path_valid(".config/settings.json"));
		assert!(is_local_path_valid("file.tar.gz"));
		assert!(is_local_path_valid("backup.db.bak"));
		assert!(is_local_path_valid("dir/./file.txt"), "Should allow current directory traversal"); // I don't technically want to allow this, but Path::components breaks this down without the curdir component, and it isn't harmful so *shrug*
		let long_path = format!("{}/{}/{}.json", "a".repeat(50), "b".repeat(50), "c".repeat(50));
		assert!(is_local_path_valid(&long_path));
	}

	#[test]
	fn test_is_local_path_valid_invalid() {
		assert!(!is_local_path_valid("/absolute/path/file.txt"), "Absolute paths are not valid");
		assert!(!is_local_path_valid("/tmp/cache/file.mds"), "Absolute paths are not valid");
		assert!(!is_local_path_valid("../file.txt"), "Path traversal is not allowed");
		assert!(!is_local_path_valid("dir/../file.txt"), "Path traversal is not allowed");
		assert!(!is_local_path_valid("../../etc/passwd"), "Path traversal is not allowed");
		assert!(!is_local_path_valid("subdir/../../../file.txt"), "Path traversal is not allowed");
		assert!(!is_local_path_valid("./file.txt"), "Current directory traversal is not allowed");
		assert!(!is_local_path_valid("./subdir/file.txt"), "Current directory traversal is not allowed");
		assert!(!is_local_path_valid(""), "Empty path is not valid");
		assert!(!is_local_path_valid("dir/"), "Filename is required for a valid local path");
		assert!(!is_local_path_valid("subdir/nested/"), "Filename is required for a valid local path");
	}

	/// Insert two shards whose combined weight is larger than the configured
	/// `cache_limit` and ensure that the cache evicts enough data to respect
	/// the limit.  Also confirm that the eviction listener removed at least
	/// one of the corresponding files from disk.
	#[tokio::test(flavor = "current_thread")]
	async fn shard_cache_respects_size_limit() {
		// ────── Arrange ──────
		// A tiny cache (1 KiB) and a temporary directory to act as the cache dir.
		let cache_limit: u64 = 1024;
		let tmpdir = tempfile::tempdir().expect("failed to create temp dir");
		let cache_dir = tmpdir.path().to_str().unwrap();

		// Build the cache (this also scans the directory, which is empty now).
		let shard_cache = ShardCache::new(cache_limit, cache_dir).await;

		// Create two dummy shard files, each 800 bytes – together they exceed the limit.
		let shard_a_path = tmpdir.path().join("shard_a.mds");
		let shard_b_path = tmpdir.path().join("shard_b.mds");
		tokio::fs::write(&shard_a_path, vec![0u8; 800]).await.expect("write shard_a");
		tokio::fs::write(&shard_b_path, vec![0u8; 800]).await.expect("write shard_b");

		// Helper to wrap a file into an Arc<ShardMeta>.
		let make_meta = |bytes| Arc::new(ShardMeta { bytes, remote: None });

		// ────── Act ──────
		// Insert both shards.  After the second insert the cache weight
		// (800 + 800) exceeds the 1 KiB limit, so Moka must evict.
		shard_cache.cache.insert(shard_a_path.to_str().unwrap().to_owned(), make_meta(800)).await;
		shard_cache.cache.insert(shard_b_path.to_str().unwrap().to_owned(), make_meta(800)).await;

		// Let the cache run its eviction tasks.
		shard_cache.cache.run_pending_tasks().await;

		// Sleep just in case.
		sleep(Duration::from_millis(100)).await;

		// ────── Assert ──────
		// 1. The in‑memory weighted size never exceeds the limit.
		assert!(
			shard_cache.cache.weighted_size() <= cache_limit && shard_cache.cache.weighted_size() > 0,
			"cache weighted_size={} exceeds limit={} or is zero",
			shard_cache.cache.weighted_size(),
			cache_limit
		);

		// 2. At least one of the shard files was removed from disk by
		//    the eviction listener (proving that the listener executed).
		let exists_a = tokio::fs::try_exists(&shard_a_path).await.unwrap();
		let exists_b = tokio::fs::try_exists(&shard_b_path).await.unwrap();
		assert!(!(exists_a && exists_b), "both shard files are still present; eviction listener did not run");
	}
}
