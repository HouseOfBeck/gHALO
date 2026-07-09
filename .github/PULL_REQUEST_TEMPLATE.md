## Summary

Describe the change and why it is needed.

## Scope

- [ ] Documentation
- [ ] Build or CI
- [ ] Portable core
- [ ] MPI backend
- [ ] Future GPU/backend infrastructure
- [ ] Tests
- [ ] Other:

## Design Notes

Document architectural decisions, compatibility implications, or tradeoffs.

## Validation

List commands run and systems used.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Checklist

- [ ] I kept the change focused.
- [ ] I updated documentation where behavior or workflow changed.
- [ ] I preserved HALO benchmark methodology where applicable.
- [ ] I kept platform-specific dependencies isolated behind feature gates.
- [ ] I added or updated tests where practical.
