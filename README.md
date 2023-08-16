# Concurrent Composition
An implementation of Concurrent Composition for Interactive Differential Privacy with Adaptive Privacy-Loss Parameters. 
The implementation uses queryables to model odometers and filters.
A queryable is like a state machine: it answers queries that may change its internal state. 

An odometer queryable is a kind of queryable whose state consists of a sensitive dataset and a privacy accumulator.
When a measurement (a query) is submitted to the queryable, the queryable prepares an answer, accumulates the privacy loss within the state, and then returns the answer.
An answer may be noninteractive, or it may be interactive (another queryable).
This implementation uses _wrapping_ on any interactive outputs to enforce constraints necessary for the privacy properties of the compositor to hold.
For instance, if a sequential odometer spawns an interactive mechanism, then the interactive mechanism will be wrapped in another queryable that freezes itself, should another query be submitted to the parent sequential odometer.

`main.py` contains a constructor function that can be used to build sequential odometers (`make_sequential_odometer`) and similarly a constructor for concurrent odometers (`make_concurrent_odometer`).
Please compare the two implementations: since the concurrent odometer does not need to enforce sequentiality, `_new_sequentiality_wrapper` is not called, and thus answers don't get wrapped in another queryable that would freeze it.

This implementation also contains a function that can convert any odometer into a filter: `make_odometer_to_filter`.
Filter queryables also employ wrapping, to ensure that the child odometer does not violate the continuation rule.

All of these functions are demonstrated in the three `test_*` functions at the bottom of the file:
1. `test_concurrent_odometer` a concurrent compositor is built, and queries to an interactive child are interleaved.
2. `test_sequential_odometer` a sequential compositor is built, and the sequentiality constraint prevents query interleaving.
3. `test_concurrent_filter` a filter is built, and queries are submitted until the continuation rule prevents further queries.
