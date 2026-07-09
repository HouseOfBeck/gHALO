# Historical Context

gHALO is inspired by Alan Wallcraft's HALO benchmark, developed for the Naval
Research Laboratory and described in the Fall 1999 NAVO MSRC Navigator.

The original HALO benchmark reflected a practical HPC concern that remains
important: many scientific applications spend meaningful time exchanging halo
regions between neighboring subdomains. Measuring that communication pattern
directly can reveal behavior that generic bandwidth tests may miss.

Modern HPC systems have changed substantially since the original benchmark was
described. GPU memory, GPU-aware MPI, accelerator interconnects, collective
communication libraries, and lower-level transport APIs now shape the cost and
reliability of distributed stencil-like workloads.

gHALO carries forward the original benchmark philosophy rather than the original
implementation. The project asks a contemporary version of the same question:
how healthy, scalable, and predictable is halo exchange communication on the
system where real applications are expected to run?

## Why a New Project?

A direct translation would risk preserving implementation assumptions from a
different generation of machines. gHALO is a new implementation path intended
for:

- GPU-resident data
- accelerator-aware communication stacks
- structured benchmark output
- diagnostic reporting
- topology and cluster-health analysis

The historical link is conceptual: measure the communication pattern that
matters, and make the result useful for people responsible for performance.
