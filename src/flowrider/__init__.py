from .flowrider import StreamingDataset as StreamingDatasetRust
from .flowrider import Config
from torch.utils.data import IterableDataset
import torch


__all__ = [
	'StreamingDataset',
	'Stream',
	'init',
	'build_streams',
]


# PyO3 doesn't allow Rust pyclasses to inherit from Python classes directly,
# so we create a wrapper class that inherits from IterableDataset.
class StreamingDataset(IterableDataset):
	def __init__(self, remotes_and_locals: list[tuple[str, str]], config: Config, seed: bytes | int, shuffle: bool, drop_last: bool, micro_batch_size: int):
		super().__init__()
		if isinstance(seed, int):
			seed = seed.to_bytes(8, byteorder='little')
		self._inner = StreamingDatasetRust(remotes_and_locals, config, seed, shuffle, drop_last, micro_batch_size)
		self.epoch = 0  # Initialize epoch to track the current epoch state
	
	def __iter__(self):
		info = torch.utils.data.get_worker_info()
		worker_id = info.id if info is not None else 0
		num_workers = info.num_workers if info is not None else 1
		indices = self._inner.get_iter(self.epoch, worker_id, num_workers)
		for idx in indices:
			yield self[idx]
		self.epoch += 1
		
	def __getstate__(self):
		return self._inner.__getstate__()

	def __setstate__(self, state):
		self._inner = StreamingDatasetRust.__setstate__(state)
	
	def __len__(self):
		return self._inner.__len__()
	
	def get_sample(self, idx: int):
		info = torch.utils.data.get_worker_info()
		return self._inner.get_sample(idx, info.id if info is not None else 0)
	
	def __str__(self):
		return str(self._inner)
