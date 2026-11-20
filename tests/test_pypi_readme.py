# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
README.pypi.md (the PyPI description) is generated from README.md by
scripts/build_pypi_readme.py. These tests keep the two in sync and pin down
the LaTeX → PyPI-safe HTML conversion.
"""

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location('build_pypi_readme', ROOT / 'scripts' / 'build_pypi_readme.py')
build_pypi_readme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_pypi_readme)

latex_to_html = build_pypi_readme.latex_to_html
convert       = build_pypi_readme.convert
LatexError    = build_pypi_readme.LatexError


class TestGeneratedReadme(unittest.TestCase):

    def test_pypi_readme_is_up_to_date(self):
        expected = build_pypi_readme.build()
        actual = (ROOT / 'README.pypi.md').read_text()
        self.assertEqual(
            actual, expected,
            "README.pypi.md is out of date: run python scripts/build_pypi_readme.py",
        )

    def test_version_references_in_sync_with_pyproject(self):
        """BibTeX version in README.md and __version__ follow pyproject.toml."""
        version = build_pypi_readme.project_version()
        for path, content in build_pypi_readme.synced_files().items():
            with self.subTest(file=path.name):
                self.assertEqual(
                    path.read_text(), content,
                    f"{path.name} has a stale version reference (pyproject.toml: {version}): "
                    f"run python scripts/build_pypi_readme.py",
                )

    def test_package_version_matches_pyproject(self):
        import minmaxrnc
        self.assertEqual(minmaxrnc.__version__, build_pypi_readme.project_version())

    def test_sync_version_rewrites_only_the_marked_reference(self):
        pattern = build_pypi_readme.VERSIONED[0][1]
        src = 'New in 0.4.0.\n```bibtex\n@software{x,\n  version = {0.3.0},\n}\n```\n'
        out = build_pypi_readme.sync_version(src, pattern, '1.2.3', 'test')
        self.assertIn('  version = {1.2.3},', out)
        self.assertIn('New in 0.4.0.', out)
        with self.assertRaises(RuntimeError):
            build_pypi_readme.sync_version('no reference here', pattern, '1.2.3', 'test')

    def test_pyproject_uses_generated_readme(self):
        self.assertIn('readme = {file = "README.pypi.md"', (ROOT / 'pyproject.toml').read_text())

    def test_no_math_syntax_or_relative_links_left(self):
        out = (ROOT / 'README.pypi.md').read_text()
        self.assertNotIn('$`', out)
        self.assertNotIn('```math', out)
        for target in build_pypi_readme.LINK.findall(out):
            link = target[1]
            self.assertRegex(link, r'^(https?:|#|mailto:)', f"relative link {link!r} would break on PyPI")

    def test_links_pinned_to_release_tag(self):
        self.assertRegex(build_pypi_readme.release_ref(), r'^v\d+\.\d+\.\d+')


class TestLatexToHtml(unittest.TestCase):

    CASES = [
        (r'k',                                   '<em>k</em>'),
        (r'k = T',                               '<em>k</em> = <em>T</em>'),
        (r'O(\log T)',                           '<em>O</em>(log <em>T</em>)'),
        (r'x_{t+1}',                             '<em>x</em><sub><em>t</em>+1</sub>'),
        (r'x_t \in \mathbb{R}^d',                '<em>x</em><sub><em>t</em></sub> ∈ ℝ<sup><em>d</em></sup>'),
        (r'\mathbb{R}^{d \times d}',             'ℝ<sup><em>d</em> × <em>d</em></sup>'),
        (r'(R \otimes x)_i = \max_j \min(R_{ij}, x_j)',
         '(<em>R</em> ⊗ <em>x</em>)<sub><em>i</em></sub> = max<sub><em>j</em></sub> '
         'min(<em>R</em><sub><em>ij</em></sub>, <em>x</em><sub><em>j</em></sub>)'),
        (r'\prod_i (I - \beta_i k_i k_i^\top)',
         '∏<sub><em>i</em></sub> (<em>I</em> − β<sub><em>i</em></sub> <em>k</em><sub><em>i</em></sub> '
         '<em>k</em><sub><em>i</em></sub><sup>⊤</sup>)'),
        (r'd > 1',                               '<em>d</em> &gt; 1'),
        (r'\text{num\_neurons} \cdot d',         'num_neurons · <em>d</em>'),
        (r'S_3',                                 '<em>S</em><sub>3</sub>'),
        # blackboard bold in every common spelling, and Unicode passed through upright
        (r'x \in \mathbb R^d',                   '<em>x</em> ∈ ℝ<sup><em>d</em></sup>'),
        (r'x \in \Bbb{R}^d',                     '<em>x</em> ∈ ℝ<sup><em>d</em></sup>'),
        (r'x \in \Bbb R^d',                      '<em>x</em> ∈ ℝ<sup><em>d</em></sup>'),
        (r'x_t \in ℝ^d',                         '<em>x</em><sub><em>t</em></sub> ∈ ℝ<sup><em>d</em></sup>'),
        (r'α + x',                               'α + <em>x</em>'),
    ]

    def test_cases(self):
        for tex, expected in self.CASES:
            with self.subTest(tex=tex):
                self.assertEqual(latex_to_html(tex), expected)

    def test_unsupported_command_raises(self):
        with self.assertRaises(LatexError):
            latex_to_html(r'\frac{a}{b}')

    def test_unbalanced_braces_raise(self):
        for tex in (r'x_{t', r'x}'):
            with self.subTest(tex=tex), self.assertRaises(LatexError):
                latex_to_html(tex)


class TestConvertDocument(unittest.TestCase):

    def test_inline_math_and_links(self):
        out = convert('A $`x_t`$ and [docs](docs/model.md#a), [site](https://x.org), [here](#b).\n', 'v9.9.9')
        self.assertIn('A <em>x</em><sub><em>t</em></sub> and', out)
        self.assertIn('(https://github.com/minmaxrnc/model/blob/v9.9.9/docs/model.md#a)', out)
        self.assertIn('(https://x.org)', out)
        self.assertIn('(#b)', out)

    def test_display_math_becomes_paragraph(self):
        out = convert('Before\n\n   ```math\n   x_{t+1} = s_t\n   ```\n\nAfter\n', 'v1.0.0')
        self.assertIn('\n   <em>x</em><sub><em>t</em>+1</sub> = <em>s</em><sub><em>t</em></sub>\n', out)
        self.assertNotIn('```', out)

    def test_code_is_left_untouched(self):
        src = '```python\nprice = "$`x_t`$"  # $5\n```\n\nUse `$HOME` here.\n'
        out = convert(src, 'v1.0.0')
        self.assertIn('price = "$`x_t`$"  # $5', out)
        self.assertIn('Use `$HOME` here.', out)

    def test_block_markers(self):
        src = ('- Common\n'
               '  <!-- github-only -->\n'
               '- GH item [d](docs/a.md).\n'
               '  <!-- /github-only -->\n'
               '  <!-- pypi-only\n'
               '  - PyPI item [r](https://x.org).\n'
               '  -->\n'
               'end\n')
        out = convert(src, 'v1.0.0')
        self.assertIn('- Common\n- PyPI item [r](https://x.org).\nend\n', out)
        self.assertNotIn('GH item', out)
        self.assertNotIn('-only', out)

    def test_pypi_block_content_less_indented_than_marker_is_rejected(self):
        """On GitHub, a less-indented line closes the list item the comment
        opened in, leaving it unterminated: it would hide the rest of the page."""
        src = '- A\n  <!-- pypi-only\n- B\n  -->\nrest\n'
        with self.assertRaises(LatexError):
            convert(src, 'v1.0.0')

    def test_inline_markers(self):
        src = ('- See <!-- github-only -->GH `code`<!-- /github-only -->'
               '<!-- pypi-only PyPI [r](https://x.org) --> here.\n')
        self.assertIn('- See PyPI [r](https://x.org) here.\n', convert(src, 'v1.0.0'))

    def test_inline_marker_at_line_start_is_rejected(self):
        """A line (or list item) beginning with <!-- is a raw HTML block in
        CommonMark: GitHub would stop rendering Markdown in it."""
        for src in ('- <!-- github-only -->GH `x`<!-- /github-only -->\n',
                    '<!-- pypi-only PyPI text -->\n',
                    '1. <!-- pypi-only PyPI text --> more\n'):
            with self.subTest(src=src), self.assertRaises(LatexError):
                convert(src, 'v1.0.0')

    def test_unmatched_marker_is_rejected(self):
        for src in ('Text <!-- github-only -->open\n', '  <!-- github-only -->\nopen block\n'):
            with self.subTest(src=src), self.assertRaises(LatexError):
                convert(src, 'v1.0.0')

    def test_plain_dollar_math_is_rejected(self):
        with self.assertRaises(LatexError):
            convert('Inline $x_t$ math.\n', 'v1.0.0')

    def test_escaped_dollar_is_allowed(self):
        self.assertIn(r'costs \$5', convert('It costs \\$5.\n', 'v1.0.0'))


if __name__ == '__main__':
    unittest.main()
