use std::{path::{Path, PathBuf}, sync::Arc, time::Instant};
use async_recursion::async_recursion;
use moka::future::{Cache, FutureExt};
use anyhow::{ensure, Context};
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
			.async_eviction_listener(|key, _meta, _cause| async move {
				if let Err(err) = tokio::fs::remove_file(Path::new(&*key)).await {
					eprintln!("Warning: Failed to remove file {}: {}", key, err);
				}
			}.boxed());

		if cache_limit > 0 {
			cache = cache.max_capacity(cache_limit);
		}

		let cache = cache.build();

		let this = ShardCache {
			cache,
			cache_dir,
		};
		
		// find existing shards in the cache directory and pre-populate the cache
		println!("Populating shard cache from {}", this.cache_dir.display());
		this.populate_cache(&this.cache_dir).await;
		println!("Shard cache populated");

		this
	}

	#[async_recursion]
	async fn populate_cache(&self, path: &Path) {
		if !path.exists() {
			return;
		}

		let Ok(mut entries) = tokio::fs::read_dir(path)
			.await
			.inspect_err(|e| {
    			eprintln!("Warning: Failed to read directory {}: {}", path.display(), e);
			}) else {
    			return;
		};

		while let Some(entry) = match entries.next_entry().await {
			Ok(e) => e,
			Err(e) => {
				eprintln!("Warning: Failed to read entry in directory {}: {}", path.display(), e);
				return;
			}
		} {
			let path = entry.path();
			if path.is_file() && path.extension().map_or(false, |ext| ext == "mds") {
				let Ok(local_path) = path.canonicalize() else {
					eprintln!("Warning: Failed to canonicalize path {}. Skipping.", path.display());
					continue;
				};
				let Some(local) = local_path.to_str() else {
					eprintln!("Warning: Path {} is not valid UTF-8. Skipping.", local_path.display());
					continue;
				};

				let Ok(metadata) = tokio::fs::metadata(&path).await else {
					eprintln!("Warning: Failed to get metadata for path {}. Skipping.", path.display());
					continue;
				};

				let meta = Arc::new(ShardMeta { bytes: metadata.len() as u32, remote: None });
				self.cache.insert(local.to_string(), meta).await;
			} else if path.is_dir() {
				self.populate_cache(&path).await;
			}
		}
	}
	
	pub async fn get_shard(&self, remote: Url, local: &str, expected_hash: u128, download_semaphore: &Semaphore) -> anyhow::Result<Arc<ShardMeta>> {
		// local path must be valid
		// since local paths cannot have traversal components, we guarantee they can be used as unique keys in the cache
		// (this assumes no symlinked directories or other tricks in the cache directory)
		ensure!(is_local_path_valid(local), "Local path '{}' is not valid. It must be a relative path without traversal components, must have a file name, and must not be empty.", local);

		let local_cache_path = self.cache_dir.join(local);

		// check for and avoid a footgun:
		// if the user uses a remote file:// URL, but sets the cache directory to the same, then cache reaping would delete the original dataset
		if remote.scheme() == "file" {
			let remote_path = remote.to_file_path()
				.map_err(|_| anyhow::anyhow!("Remote URL '{}' is not a valid file path", remote))?
				.parent()
				.ok_or_else(|| anyhow::anyhow!("Remote URL '{}' does not have a parent directory", remote))?
				.canonicalize()
				.with_context(|| format!("Failed to canonicalize remote path: {}", remote))?;
			let local_path = local_cache_path.parent()
				.ok_or_else(|| anyhow::anyhow!("Local cache path '{}' does not have a parent directory", local_cache_path.display()))?;

			if let Ok(local_path) = local_path.canonicalize() {
				ensure!(remote_path != local_path,
					"Remote path '{}' must not be the same as local cache path '{}'. This would cause the original dataset to be deleted when the cache evicts.",
					remote_path.display(), local_cache_path.display());
			}
		}

		let local_cache_path = local_cache_path.to_str()
			.ok_or_else(|| anyhow::anyhow!("Local cache path '{}' is not valid UTF-8", local_cache_path.display()))?;

		// If the shard is in the cache, we can return immediately.
		// Otherwise, moka's Cache ensures that only a single instance of download_shard will run concurrently for the same key.
		// Once the download is complete, it will be cached and we (and all other waiting tasks) can return.
		match self.cache.try_get_with_by_ref(local, download_shard(&remote, local_cache_path, expected_hash, download_semaphore)).await {
			Ok(meta) => {
				if let Some(meta_remote) = &meta.remote {
					ensure!(meta_remote == &remote, "Cached shard at {} has different remote URL than requested: {} != {}", local, meta_remote, remote);
				}
				println!("Using cached shard at {}", local);
				Ok(meta)
			},
			Err(e) => {
				Err(anyhow::anyhow!("Failed to get shard {}: {}", local, e))
			},
		}
	}

}


fn is_local_path_valid(path: &str) -> bool {
	let path = Path::new(path);

	// must be a relative path
	if !path.is_relative() {
		return false;
	}

	// must not contain any path traversal components
	if path.components().any(|c| c == std::path::Component::ParentDir || c == std::path::Component::CurDir) {
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


async fn download_shard(remote: &Url, local: &str, expected_hash: u128, download_semaphore: &Semaphore) -> anyhow::Result<Arc<ShardMeta>> {
	let start = Instant::now();
	crate::server::download_file(remote, local, Some(expected_hash), download_semaphore).await?;

	let bytes = tokio::fs::metadata(local)
		.await
		.context(format!("Failed to get metadata for shard at {}", local))?
		.len() as u32;
	let meta = Arc::new(ShardMeta { bytes, remote: Some(remote.clone()) });

	let elapsed = start.elapsed();
	println!("Downloaded shard {} in {:?}", local, elapsed);

	Ok(meta)
}
