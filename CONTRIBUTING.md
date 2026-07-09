# Contributing to gHALO

Thank you for helping build gHALO. The project is intended to become a
long-lived open-source HPC benchmark and diagnostic suite, so contributions
should favor clarity, reproducibility, and careful measurement over quick
feature growth.

## Project Goals

gHALO preserves the benchmark philosophy of Alan Wallcraft's HALO benchmark
while modernizing the implementation for GPU-based HPC systems.

Primary goals:

- Measure halo exchange latency, bandwidth, scaling, and topology health.
- Keep benchmark methodology explicit and reproducible.
- Isolate communication backends from benchmark orchestration.
- Support portable C++20 development on macOS workstations.
- Support remote Linux HPC builds for MPI, HIP, RCCL, UCX, and future backends.
- Grow from a benchmark into a diagnostic suite for cluster communication
  health.

## Development Workflow

1. Read the relevant design docs before changing behavior.
2. Keep changes focused and reviewable.
3. Add or update documentation when behavior, output, build requirements, or
   architecture changes.
4. Add tests for portable logic whenever practical.
5. Run the available local checks before opening a pull request.
6. Run MPI, HIP, RCCL, UCX, or GPU tests on an appropriate remote system when
   your change touches those areas.

Useful local commands:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Version 0 MPI builds require MPI:

```sh
cmake -S . -B build-mpi -DGHALO_ENABLE_MPI=ON
cmake --build build-mpi
ctest --test-dir build-mpi
```

## Coding Standards

- Use C++20.
- Prefer portable standard-library facilities in core code.
- Keep MPI, HIP, RCCL, UCX, and other platform-specific APIs behind backend
  boundaries.
- Prefer target-scoped CMake options, include directories, compile definitions,
  and link libraries.
- Keep benchmark timing methodology visible in code and documentation.
- Record output metadata needed to reproduce or compare results.
- Avoid introducing local build assumptions that require ROCm, HIP, MPI, or GPU
  hardware on macOS.

Formatting is governed by `.clang-format` and `.editorconfig`.

## Branch Strategy

- Use `main` for reviewed, working project history.
- Use short-lived feature branches for changes.
- Prefer branch names such as:
  - `codex/docs-contributing`
  - `feature/mpi-baseline`
  - `fix/cmake-mpi-detection`
  - `docs/frontier-notes`

Avoid mixing unrelated changes in one branch.

## Commit Messages

Use concise, imperative commit subjects:

```text
Add MPI backend skeleton
Document HALO timing methodology
Fix CSV header metadata
```

Recommended format:

```text
Short imperative subject

Optional body explaining why the change is needed, what tradeoffs were made,
and how it was validated.
```

For larger changes, mention the affected area in the subject when helpful:

```text
cmake: gate HIP language enablement
docs: add versioning strategy
mpi: preserve HALO grid factorization
```

## Pull Request Expectations

Pull requests should include:

- A short summary of the change.
- Motivation or linked issue.
- Notes on architecture or methodology decisions.
- Test and validation results.
- Documentation updates when applicable.
- Any limitations, follow-up work, or unsupported platforms.

Do not include generated build directories, local result files, or machine-local
configuration unless they are intentionally documented examples.

## Code Review Expectations

Review should focus on:

- Correctness and reproducibility.
- Preservation of benchmark semantics.
- Backend isolation.
- Portability across macOS development and Linux HPC execution.
- CMake feature detection and dependency boundaries.
- Test coverage appropriate to the risk of the change.
- Documentation clarity.

Reviewers should be specific, constructive, and respectful. Contributors should
respond by either updating the change or explaining the tradeoff.

## Architectural Decisions

Architectural decisions should be documented in the relevant design document or
in a new file under `docs/` when the decision affects long-term project shape.
At minimum, document:

- the problem being solved;
- the chosen approach;
- alternatives considered;
- compatibility implications;
- validation expectations.

## License

By contributing, you agree that your contributions will be licensed under the
project's MIT License.
