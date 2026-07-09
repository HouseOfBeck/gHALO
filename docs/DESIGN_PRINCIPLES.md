1. Preserve benchmark semantics over source compatibility.
2. GPU memory is the primary data location; avoid host staging whenever possible.
3. Measure steady-state communication, not initialization.
4. The slowest rank determines benchmark performance.
5. Separate benchmark logic from communication backends.
6. Diagnostics are a first-class feature, not an afterthought.
7. Every architectural decision should support scaling from a laptop to a large GPU cluster.

