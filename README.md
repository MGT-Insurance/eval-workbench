# Eval Workbench


# PARALLEL Branch

---
# MGT Evaluation Implementations

This repository contains specific implementations built on top of the [Axion](https://github.com/ax-foundry/axion) or any other Evaluation Module. This architecture separates the core evaluation framework from individual evaluation implementations, enabling better tracking, ability to create custom sharable tooling, easier sharing, and a clear separation of concerns.


---


### pre-commit
Formatting is managed via pre-commit hooks.
```
# Run on all files
pre-commit run --all-files

# Install to run after every commit
pre-commit install
```