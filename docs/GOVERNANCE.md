# Governance

This document defines the long-term engineering governance of gHALO. It is not a
legal document. It describes the technical philosophy, decision process, and
project expectations that should guide gHALO over many years of open-source HPC
development.

The intended audience includes maintainers, contributors, reviewers, downstream
users, and AI coding assistants working in the repository.

## Mission

The mission of gHALO is to preserve the benchmark philosophy of Alan
Wallcraft's original HALO benchmark while modernizing its implementation for
modern GPU-based HPC systems.

gHALO exists to evaluate:

- communication latency;
- communication bandwidth;
- scalability;
- communication topology health.

The project is intended to become both a benchmark and a diagnostic tool. It
should measure performance, but it should also help explain communication
problems that affect real HPC applications.

## Project Principles

1. Preserve benchmark semantics.

   Compatibility with the benchmark methodology is more important than
   compatibility with the original implementation. gHALO should honor what the
   original benchmark measured, not reproduce legacy source structure for its
   own sake.

2. Benchmark real communication patterns.

   Avoid synthetic microbenchmarks whenever practical. gHALO should continue to
   represent halo exchanges that occur in production HPC applications, including
   staged nearest-neighbor communication, realistic synchronization, and
   application-visible completion costs.

3. Measure steady-state communication.

   Initialization, allocation, setup, request creation, runtime warm-up, and
   other one-time costs should not influence primary benchmark results. Those
   costs may be measured separately when useful, but they must be labeled
   separately.

4. The slowest participant determines application performance.

   In distributed applications, progress is limited by the rank, device, link,
   or communication path that completes last. Whenever appropriate, gHALO should
   report worst-case communication performance rather than averages alone.

5. Diagnostics are first-class features.

   Finding communication problems is as important as measuring performance.
   Output formats, metadata, topology information, and analysis tools should be
   designed to support diagnosis, not only headline benchmark numbers.

6. GPU memory is the primary data location.

   The long-term architecture should treat GPU-resident data as the normal case
   on modern accelerator systems. Avoid unnecessary host staging, and when host
   staging is used, make it explicit in the backend metadata and output.

7. Communication backends must be interchangeable.

   MPI, GPU-aware MPI, RCCL, UCX, SHMEM, and future implementations should
   preserve identical benchmark semantics. Backend differences should reveal
   communication behavior, not change the benchmark definition.

8. Performance should never compromise correctness.

   Optimizations must not change benchmark semantics, corrupt validation paths,
   hide synchronization requirements, or produce misleading measurements. Fast
   incorrect results are not useful.

9. Documentation is part of the software.

   Major architectural changes require documentation updates. If a design
   decision affects benchmark semantics, backend boundaries, output schemas,
   build requirements, or diagnostic interpretation, it should be documented
   before it becomes a hidden assumption.

## Architecture Governance

gHALO should remain modular enough to support new communication technologies
without rewriting benchmark semantics.

Expected boundaries:

- Benchmark logic must remain independent of communication backends.
- Communication backends must remain independent of output formatting.
- Output formatting must remain independent of diagnostics.
- Visualization tools should remain independent of benchmark execution.

These boundaries are intended to prevent accidental coupling:

- The benchmark core defines what is measured.
- Backends define how communication is performed.
- Output writers serialize results.
- Diagnostics interpret results.
- Visualization tools display results.

Backends may report metadata needed by output and diagnostics, but they should
not decide how results are formatted or visualized. Diagnostics may consume
benchmark output, but they should not be required for benchmark execution.

Avoid unnecessary coupling, global dependencies, and architecture that assumes a
single machine, scheduler, GPU vendor, MPI implementation, or network stack.

## Coding Philosophy

Prefer:

- modern C++20;
- RAII for resource lifetime;
- standard-library facilities where they are sufficient;
- clean interfaces with explicit ownership and synchronization behavior;
- unit testing for portable logic;
- small commits with focused intent.

Avoid:

- global state;
- hidden dependencies;
- duplicated code;
- premature optimization;
- backend-specific assumptions in the portable core;
- output behavior embedded inside communication code.

Performance work is welcome, but it should follow measurement and correctness.
When optimization changes the shape of the benchmark, synchronization behavior,
memory placement, or reported metrics, the change requires careful review and
documentation.

## Versioning

gHALO should use Semantic Versioning.

Expected progression:

- `0.x`: research, architecture, backend development, output schema iteration,
  and validation methodology.
- `1.x`: stable benchmark interface, stable compatibility mode, documented
  output schemas, and production-ready use by HPC centers and application
  teams.
- `2.x`: major new benchmark capabilities or intentionally breaking changes
  that expand the scope of what gHALO measures.

Before `1.0.0`, internal APIs may evolve quickly. Even during `0.x`
development, benchmark semantics and published output meaning should change
only deliberately and with documentation.

After `1.0.0`, changes that alter primary metrics, compatibility semantics,
command-line behavior, or output schema meaning should be treated as breaking
changes.

## Contribution Policy

Contributions are encouraged. gHALO should remain open to improvements from HPC
centers, national laboratories, universities, vendors, application teams, and
independent contributors.

Contributions are expected to provide:

- readable code;
- documentation for behavior, architecture, or workflow changes;
- tests where practical;
- enough information to reproduce benchmark results;
- clear separation between benchmark semantics and backend implementation.

Changes that affect benchmark methodology should explain how compatibility is
preserved. Changes that affect performance should explain how results were
measured. Changes that affect diagnostics should explain what problem they help
detect.

## Decision Process

For any significant architectural decision, contributors should first answer:

- Does this preserve benchmark semantics?
- Does this improve diagnostics?
- Does this improve portability?
- Does this improve maintainability?
- Does this improve scientific validity?

If the answer to any question is "no", the contributor should explain why the
change is still justified. A valid justification may include preserving
compatibility, enabling an important backend, reducing long-term complexity, or
making a tradeoff explicit for a specific mode.

Significant decisions should be documented in the relevant design document,
pull request, issue, or a new document under `docs/`. The larger the long-term
impact, the more durable the documentation should be.

## Long-Term Vision

The long-term objective is for gHALO to become a trusted benchmark and
diagnostic tool used by HPC centers, national laboratories, universities, and
industry.

gHALO should become a modern successor to the original HALO benchmark while
honoring its historical design philosophy: measure the communication behavior
that matters to real domain-decomposed applications, expose scaling and
topology problems clearly, and report results in a form that helps users make
sound technical decisions.

The project should remain conservative about benchmark semantics and ambitious
about diagnostics. Over time, gHALO should help users understand not only how
fast a system communicates, but where, when, and why communication performance
breaks down.
