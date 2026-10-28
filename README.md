# Ascend Analyzers

Offline analysis tools for Ascend NPU toolchain reports (BiSheng / Triton compiler).
Python 3.9+, standard library only.

## Install

```bash
# From a local clone
pip install .

# Or directly from GitHub
pip install git+https://github.com/<you>/ascend-analyzers.git
```

## Tools

### UB Memory Analyzer

Analyzes `memory_info_aiv.json` (BiSheng MemoryDisplay JSON) and produces an
interactive HTML report, a machine-readable `analysis.json`, and a `timeline.csv`.

```bash
# CLI
ascend-ub-analyzer memory_info_aiv.json -o ub_report
ascend-ub-analyzer memory_info_aiv.json -o ub_report --capacity-bytes 196608

# Or via module
python -m ascend_analyzers.ub memory_info_aiv.json -o ub_report
```

Python API:

```python
from ascend_analyzers.ub import analyze_document, write_outputs
import json

report = analyze_document(json.load(open("memory_info_aiv.json")), capacity_bytes=None)
write_outputs(report, "ub_report")
```

## Tests

```bash
pip install -e .
python -m pytest tests/ -v
# or
python -m unittest discover -s tests -v
```

## Adding a new analyzer

1. Create a new subpackage under `src/ascend_analyzers/` (e.g. `src/ascend_analyzers/gm/`)
2. Add an `__init__.py` that re-exports the public API
3. Add a `__main__.py` if it has a CLI
4. Register a `console_scripts` entry point in `pyproject.toml`
5. Add tests under `tests/`
