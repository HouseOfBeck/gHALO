# Versioning Strategy

gHALO should use semantic versioning with extra care while the project is below
`1.0.0`.

Before `1.0.0`, minor versions may introduce planned feature families and may
change internal APIs. User-facing output schemas, documented command-line
interfaces, and benchmark compatibility modes should still be changed
deliberately and documented in `CHANGELOG.md`.

## Proposed Milestones

- `0.1.x`: CPU reference implementation and MPI baseline.
- `0.2.x`: GPU-aware MPI and device-resident exchange paths.
- `0.3.x`: HIP foundations, kernels, and GPU buffer management.
- `0.4.x`: RCCL backend experiments.
- `0.5.x`: UCX backend experiments and lower-level transport diagnostics.
- `0.6.x`: Diagnostic reporting, heat maps, and topology health analysis.
- `1.0.0`: First production release with stable CLI, output schemas,
  compatibility mode, and documented validation expectations.

This extends the initial proposal by giving UCX and diagnostics explicit
pre-`1.0.0` space. That keeps `1.0.0` reserved for a production-quality tool
rather than the first release that merely contains all planned backends.

## Patch Releases

Patch releases should be used for:

- bug fixes;
- documentation corrections;
- CI and packaging fixes;
- non-breaking output additions;
- portability fixes.

## Breaking Changes

Before `1.0.0`, breaking changes are allowed but should be documented. After
`1.0.0`, breaking changes should require a major version bump.

Examples of breaking changes:

- changing the default benchmark methodology;
- changing the meaning of a reported metric;
- removing output fields;
- changing CSV or JSON schema semantics;
- changing CLI behavior in incompatible ways.
