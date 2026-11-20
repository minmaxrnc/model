# Contributing

Thank you for your interest in contributing.

By submitting a contribution, you agree that your contribution may be included in this project under the PolyForm Noncommercial License 1.0.0.

You also agree to the Contributor License Agreement in `CLA.md`.

Do not submit code that you do not have the right to contribute.

Do not submit code copied from GPL, AGPL, LGPL, proprietary, or unknown-license projects unless you clearly identify it before submission and receive written approval from the maintainer.

## Editing the README

`README.md` is written for GitHub and uses GitHub's math syntax: ``$`...`$`` for inline formulas and
```` ```math ```` blocks for display formulas. The PyPI project description, `README.pypi.md`, is
generated from it, because PyPI does not render math:

```bash
python scripts/build_pypi_readme.py           # regenerate README.pypi.md
python scripts/build_pypi_readme.py --check   # verify it is up to date (also run by the tests)
```

The script converts formulas to HTML that PyPI renders (`<sub>`, `<sup>`, Unicode symbols) and
rewrites relative links to absolute GitHub URLs pinned to the release tag. It supports a small LaTeX
subset and fails on anything else; extend `SYMBOLS` / `FUNCTIONS` in the script when needed. Do not
edit `README.pypi.md` by hand.
