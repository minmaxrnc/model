# Contributing

Thank you for your interest in contributing.

By submitting a contribution, you agree that your contribution may be included in this project under the PolyForm Noncommercial License 1.0.0.

You also agree to the Contributor License Agreement in `CLA.md`.

Do not submit code that you do not have the right to contribute.

Do not submit code copied from GPL, AGPL, LGPL, proprietary, or unknown-license projects unless you clearly identify it before submission and receive written approval from the maintainer.

## Running the tests

```bash
pytest
```

The backward-compatibility tests (`tests/test_backward_compat.py`) compare against reference
values recorded with 0.3.0 (`tests/fixtures/v0_3_0_reference.pt`). They check bit for bit only in
the environment where the fixture was recorded (same torch version, OS and CPU architecture), since
PyTorch does not guarantee bitwise-identical seeded initialisation and kernels across versions or
platforms. Elsewhere they use rounding tolerances, and the operation-trace test is skipped, so a
run on another setup reports one skipped test.

## Editing the README

`README.md` is written for GitHub and uses GitHub's math syntax: ``$`...`$`` for inline formulas and
```` ```math ```` blocks for display formulas. The PyPI project description, `README.pypi.md`, is
generated from it, because PyPI does not render math:

```bash
python scripts/build_pypi_readme.py           # sync version references, regenerate README.pypi.md
python scripts/build_pypi_readme.py --check   # verify both are up to date (also run by the tests)
```

The script also keeps version references in sync with `pyproject.toml`'s `version` (the single
place to change it): the BibTeX `version = {...}` field in `README.md` and `__version__` in
`src/minmaxrnc/__init__.py` are rewritten to match. Historical mentions ("New in 0.4.0") are left
alone; to sync a new kind of reference, add it to `VERSIONED` in the script.

The script converts formulas to HTML that PyPI renders (`<sub>`, `<sup>`, Unicode symbols) and
rewrites relative links to absolute GitHub URLs pinned to the release tag. It supports a small LaTeX
subset and fails on anything else; extend `SYMBOLS` / `FUNCTIONS` in the script when needed. Do not
edit `README.pypi.md` by hand.
