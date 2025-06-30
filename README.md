# Flowrider

## WARNING: FOR PERSONAL USE ONLY, NOT PRODUCTION READY

## Overview
Inspired by MosaicML's `streaming` library (https://github.com/mosaicml/streaming), this library provides a PyTorch IterableDataset implementation that streams data from cloud storage.  It is distributed training compatible, and can cache data to disk.



## Testing

`cargo test --no-default-features --features auto-initialize`


## NOTE

- Logging has to use envlogger, even though there are ways to send logs to the Python logger.  This is because when sending logs to Python's logger, the GIL is required.  Since we have a background thread doing work (and potentially logging), that can create a minefield of either deadlocks or not allowing background threads to work.


