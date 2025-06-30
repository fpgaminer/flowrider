from .flowrider import StreamingDataset as StreamingDatasetRust
from .flowrider import Config, ColumnEncoding, SampleWriter
from torch.utils.data import IterableDataset, DataLoader
import torch
from collections.abc import Mapping, Sequence


__all__ = [
	'StreamingDataset',
	'Config',
	'StreamingDataLoader',
	'ColumnEncoding',
	'SampleWriter',
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
		self._resume = None
		self.config = config
	
	def __iter__(self):
		info = torch.utils.data.get_worker_info()
		worker_id = info.id if info is not None else 0
		num_workers = info.num_workers if info is not None else 1
		indices = self._inner.get_iter(self.epoch, worker_id, num_workers, self._resume)
		self._resume = None  # Reset resume after using it

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
	
	@property
	def micro_batch_size(self) -> int:
		return self._inner.micro_batch_size


class StreamingDataLoader(DataLoader):
	def __init__(self, *args, **kwargs):
		dataset = kwargs.get('dataset', None)
		assert isinstance(dataset, StreamingDataset), "Dataset must be an instance of StreamingDataset"
		super().__init__(batch_size=dataset.micro_batch_size, *args, **kwargs)
		self._samples_seen: int = 0
		self._epoch: int = 0
	
	@property
	def samples_seen(self) -> int:
		"""Total number of individual samples that have been yielded this epoch for this rank (not globally)."""
		return self._samples_seen
	
	@property
	def current_epoch(self) -> int:
		"""Epoch index that will be/was just iterated (0-based)."""
		return self._epoch

	def __iter__(self):
		assert isinstance(self.dataset, StreamingDataset), "Dataset must be an instance of StreamingDataset"
		self.dataset.epoch = self._epoch
		self.dataset._resume = self._samples_seen

		for batch in super().__iter__():
			batch_size = self._infer_batch_size(batch)
			self._samples_seen += batch_size
			yield batch
		
		self._epoch += 1
		self._samples_seen = 0
	
	def state_dict(self):
		state = {
			"samples_seen": self._samples_seen * self.dataset.config.world_size,
			"epoch": self._epoch,
		}
		if hasattr(super(), 'state_dict'):
			state.update(super().state_dict())
		return state
	
	def load_state_dict(self, state):
		self._samples_seen = state["samples_seen"]
		self._epoch = state["epoch"]
		if hasattr(super(), 'load_state_dict'):
			super().load_state_dict(state)
	
	@staticmethod
	def _infer_batch_size(batch) -> int:
		"""
		Figure out how many *individual* samples are in `batch`.
		Handles common collate outputs (tensor, mapping, sequence).
		"""
		if torch.is_tensor(batch):
			return batch.size(0)
		
		if isinstance(batch, Mapping):
			return StreamingDataLoader._infer_batch_size(next(iter(batch.values())))
		
		if isinstance(batch, Sequence):
			return len(batch)
		
		raise  TypeError(f"Cannot infer batch size of type {type(batch)}")
