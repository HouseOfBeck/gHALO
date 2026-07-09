# Design Principles

1. Preserve benchmark semantics over source compatibility.
2. Treat GPU memory as the long-term primary data location while preserving
   CPU/MPI reference modes.
3. Measure steady-state communication, not initialization.
4. Report the slowest rank as the primary application-visible performance
   metric.
5. Separate benchmark logic from communication backends.
6. Make diagnostics a first-class feature, not an afterthought.
7. Keep local development portable, even when production execution requires a
   remote HPC system.
8. Document architectural decisions before they become hidden assumptions.
