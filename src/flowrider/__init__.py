from .flowrider import StreamingDataset as StreamingDatasetRust
from .flowrider import Config
from torch.utils.data import IterableDataset, DataLoader
import torch
from collections.abc import Mapping, Sequence
import itertools


__all__ = [
	'StreamingDataset',
	'Config',
	'StreamingDataLoader',
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
	def __init__(self, global_batch_size: int, *args, **kwargs):
		dataset = kwargs.get('dataset', None)
		assert isinstance(dataset, StreamingDataset), "Dataset must be an instance of StreamingDataset"
		super().__init__(batch_size=dataset.micro_batch_size, *args, **kwargs)
		self._samples_seen: int = 0
		self._epoch: int = 0
		self.global_batch_size = global_batch_size
		assert global_batch_size % self.dataset.config.world_size == 0, "Global batch size must be divisible by world size"
		self.rank_batch_size = global_batch_size // self.dataset.config.world_size
	
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
		batch_accum = []

		for batch in super().__iter__():
			batch_size = self._infer_batch_size(batch)

			# Batch accumulation logic
			# Some libraries have broken logic when it comes to handling gradient accumulation and expect the dataloader to yield full batches (instead of micro-batches).
			# To work around this the user can set batch_size to the micro batch size and global_batch_size to the desired global batch size.
			# This code will accumulate micro-batches before yielding a full device batch.
			if self.batch_size < self.rank_batch_size:
				batch_accum.append((batch, batch_size))
				batch_accum_size = sum(sz for _, sz in batch_accum)
				if batch_accum_size >= self.rank_batch_size:
					self._samples_seen += batch_accum_size
					yield self.concat_batches([b for b,_ in batch_accum])
					batch_accum = []
			else:
				self._samples_seen += self._infer_batch_size(batch)
				yield batch
		
		if len(batch_accum) > 0 and self.drop_last is False:
			# Pad the last batch if requested and needed
			batch_accum = self.concat_batches([b for b,_ in batch_accum])
			batch_accum = self.pad_batch(batch_accum, self.rank_batch_size)
			self._samples_seen += self._infer_batch_size(batch_accum)
			yield batch_accum
		
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
	
	def concat_batches(self, batches: list):
		"""Concatenate a list of batches into a single batch. This default implementation handles common batch types (tensor, mapping, sequence)."""
		if len(batches) == 0:
			raise ValueError("No batches to concatenate")
		
		first_batch = batches[0]
		if torch.is_tensor(first_batch):
			return torch.cat(batches, dim=0)
		
		if isinstance(first_batch, Mapping):
			return {k: self.concat_batches([b[k] for b in batches]) for k in first_batch}
		
		if isinstance(first_batch, Sequence):
			return list(itertools.chain.from_iterable(batches))
		
		raise TypeError(f"Cannot concatenate batches of type {type(first_batch)}")
	
	def pad_batch(self, batch, target_size: int):
		"""Pad a batch to the target size. This default implementation handles common batch types (tensor, mapping, sequence), and pads by calling the dataset with a -1 index."""
		current_size = self._infer_batch_size(batch)
		if current_size >= target_size:
			return batch  # No padding needed
		
		padding = [self.dataset[-1] for _ in range(target_size - current_size)]
		padding_batch = self._collate_fn(padding)

		return self.concat_batches([batch, padding_batch])


