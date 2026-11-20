# SPDX-FileCopyrightText: 2026 Alessandro Ronca
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Build README.pypi.md (the PyPI project description) from README.md.

README.md is written for GitHub, which renders LaTeX math: inline as
$`...`$ and display as ```math fenced blocks. PyPI's renderer does not
render math, so this script converts every formula into markup both render
identically on PyPI — <em> for variables, <sub>/<sup> for indices and
exponents, Unicode for symbols (ℝ, ⊗, ⊕, ×, ∏, β, ⊤, …) — and rewrites
relative links to absolute GitHub URLs pinned to the release tag
(v<version from pyproject.toml>), since relative links do not resolve on PyPI.

Content can also differ between the two, with HTML comments (invisible on
GitHub) as markers, usable inline or across lines:

    <!-- github-only -->shown on GitHub only<!-- /github-only -->
    <!-- pypi-only shown on PyPI only -->

Only a small LaTeX subset is supported (see SYMBOLS / FUNCTIONS / FONTS);
anything else raises an error rather than producing broken output, and so
does plain $...$ math (use $`...`$, GitHub's unambiguous inline syntax).

Usage (from the repository root):

    python scripts/build_pypi_readme.py           # write README.pypi.md
    python scripts/build_pypi_readme.py --check   # exit 1 if README.pypi.md is stale

tests/test_pypi_readme.py runs the --check logic, so a README.md edit
without regenerating README.pypi.md fails the test suite.
"""

import argparse
import html
import pathlib
import re
import sys

ROOT       = pathlib.Path(__file__).resolve().parent.parent
SOURCE     = ROOT / 'README.md'
TARGET     = ROOT / 'README.pypi.md'
PYPROJECT  = ROOT / 'pyproject.toml'
REPO_URL   = 'https://github.com/minmaxrnc/model'

HEADER = (
    "<!-- Generated from README.md by scripts/build_pypi_readme.py; do not edit. -->\n"
)

# --- supported LaTeX ------------------------------------------------------

SYMBOLS = {
    'otimes': '⊗', 'oplus': '⊕', 'times': '×', 'cdot': '·', 'in': '∈',
    'notin': '∉', 'le': '≤', 'leq': '≤', 'ge': '≥', 'geq': '≥', 'ne': '≠',
    'neq': '≠', 'approx': '≈', 'to': '→', 'mapsto': '↦', 'infty': '∞',
    'prod': '∏', 'sum': '∑', 'top': '⊤', 'circ': '∘', 'ldots': '…',
    'dots': '…', 'cdots': '⋯', 'leadsto': '⇝', 'mid': '|', 'pm': '±',
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'lambda': 'λ', 'mu': 'μ', 'pi': 'π', 'sigma': 'σ', 'tau': 'τ',
    'phi': 'φ', 'omega': 'ω', 'Sigma': 'Σ', 'Delta': 'Δ',
    ',': ' ', ';': ' ', ' ': ' ', 'quad': ' ', 'qquad': ' ',
    '{': '{', '}': '}', '_': '_', '%': '%', '#': '#', '&': '&amp;',
    'left': '', 'right': '',
}
FUNCTIONS = {'max', 'min', 'log', 'exp', 'sup', 'inf', 'arg', 'argmax', 'argmin', 'deg', 'dim'}
BLACKBOARD = {'R': 'ℝ', 'N': 'ℕ', 'Z': 'ℤ', 'Q': 'ℚ', 'C': 'ℂ'}
FONTS = {'text', 'mathrm', 'operatorname', 'textrm'}          # upright text
PLAIN = set('0123456789()[],;:.!/|=+') | {"'"}
ESCAPED = {'<': '&lt;', '>': '&gt;', '-': '−', '*': '∗'}


class LatexError(ValueError):
    pass


class _Parser:
    def __init__(self, tex: str):
        self.s, self.i = tex, 0

    def error(self, msg):
        raise LatexError(f"{msg} at position {self.i} in LaTeX {self.s!r}")

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else None

    def group(self):
        """Body of a {...} group, or a single token, as raw LaTeX."""
        if self.peek() == '{':
            depth, start = 0, self.i
            while self.i < len(self.s):
                c = self.s[self.i]
                depth += (c == '{') - (c == '}')
                self.i += 1
                if depth == 0:
                    return self.s[start + 1:self.i - 1]
            self.error("unbalanced '{'")
        if self.peek() == '\\':
            start = self.i
            self.command_name()
            return self.s[start:self.i]
        if self.peek() is None:
            self.error("missing argument")
        self.i += 1
        return self.s[self.i - 1]

    def command_name(self):
        self.i += 1                                       # backslash
        m = re.match(r'[A-Za-z]+|.', self.s[self.i:])
        if not m:
            self.error("dangling backslash")
        self.i += len(m.group(0))
        return m.group(0)

    def convert(self) -> str:
        out = []
        while self.i < len(self.s):
            c = self.s[self.i]
            if c.isspace():
                while self.i < len(self.s) and self.s[self.i].isspace():
                    self.i += 1
                if out and not out[-1].endswith(' '):
                    out.append(' ')
            elif c.isalpha():
                m = re.match(r'[A-Za-z]+', self.s[self.i:])
                self.i += len(m.group(0))
                out.append(f'<em>{m.group(0)}</em>')
            elif c in '_^':
                self.i += 1
                tag = 'sub' if c == '_' else 'sup'
                out.append(f'<{tag}>{_Parser(self.group()).convert().strip()}</{tag}>')
            elif c == '{':
                out.append(_Parser(self.group()).convert())
            elif c == '}':
                self.error("unbalanced '}'")
            elif c == '\\':
                name = self.command_name()
                if name in SYMBOLS:
                    out.append(SYMBOLS[name])
                elif name in FUNCTIONS:
                    out.append(name)
                elif name == 'mathbb':
                    arg = self.group()
                    if arg not in BLACKBOARD:
                        self.error(f"unsupported \\mathbb{{{arg}}}")
                    out.append(BLACKBOARD[arg])
                elif name in FONTS:
                    out.append(html.escape(self.group().replace('\\_', '_'), quote=False))
                else:
                    self.error(f"unsupported LaTeX command \\{name}")
            elif c in PLAIN:
                self.i += 1
                out.append(c)
            elif c in ESCAPED:
                self.i += 1
                out.append(ESCAPED[c])
            elif ord(c) > 127:                            # Unicode symbols pass through
                self.i += 1
                out.append(c)
            else:
                self.error(f"unsupported character {c!r}")
        return ''.join(out)


def latex_to_html(tex: str) -> str:
    """Convert one formula (the supported LaTeX subset) to PyPI-safe HTML."""
    return _Parser(tex.strip()).convert().strip()


# --- document conversion ---------------------------------------------------

GITHUB_ONLY  = re.compile(r'<!--\s*github-only\s*-->.*?<!--\s*/github-only\s*-->', re.S)
PYPI_ONLY    = re.compile(r'<!--\s*pypi-only\s(.*?)-->', re.S)
INLINE_MATH  = re.compile(r'\$`(.+?)`\$')
CODE_SPAN    = re.compile(r'(`+).+?\1')
FENCE        = re.compile(r'^(\s*)(```+|~~~+)\s*(\w*)\s*$')
LINK         = re.compile(r'(\]\()([^)\s]+)(\))')


def _absolute_link(target: str, ref: str) -> str:
    if re.match(r'^[a-z][a-z0-9+.-]*:', target) or target.startswith('#'):
        return target
    return f'{REPO_URL}/blob/{ref}/{target.lstrip("./")}'


def _convert_line(line: str, ref: str, lineno: int) -> str:
    # Any '$' left once $`...`$ math and `code` spans are removed would be
    # GitHub's other math syntax ($...$), which we do not convert.
    rest = CODE_SPAN.sub('', INLINE_MATH.sub('', line))
    if re.search(r'(?<!\\)\$', rest):
        raise LatexError(
            f"README.md:{lineno}: unsupported '$' — use $`...`$ for inline math, "
            f"or escape a literal dollar sign as \\$"
        )
    try:
        line = INLINE_MATH.sub(lambda m: latex_to_html(m.group(1)), line)
    except LatexError as e:
        raise LatexError(f"README.md:{lineno}: {e}") from None
    return LINK.sub(lambda m: m.group(1) + _absolute_link(m.group(2), ref) + m.group(3), line)


def convert(markdown: str, ref: str) -> str:
    """README.md (GitHub flavour, LaTeX math) → README.pypi.md."""
    markdown = GITHUB_ONLY.sub('', markdown)
    markdown = PYPI_ONLY.sub(lambda m: m.group(1).strip(), markdown)
    if re.search(r'<!--\s*/?(github|pypi)-only', markdown):
        raise LatexError("README.md: unmatched github-only / pypi-only marker")
    out, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        fence = FENCE.match(line)
        if fence:
            indent, marker, lang = fence.groups()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(marker):
                j += 1
            if j == len(lines):
                raise LatexError(f"README.md:{i + 1}: unterminated code fence")
            if lang == 'math':                           # display math → its own paragraph
                tex = ' '.join(l.strip() for l in lines[i + 1:j])
                try:
                    out.append(f'{indent}{latex_to_html(tex)}')
                except LatexError as e:
                    raise LatexError(f"README.md:{i + 1}: {e}") from None
            else:                                        # other code blocks: verbatim
                out.extend(lines[i:j + 1])
            i = j + 1
            continue
        out.append(_convert_line(line, ref, i + 1))
        i += 1
    return HEADER + '\n'.join(out) + '\n'


def release_ref() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.M)
    if not m:
        raise RuntimeError("version not found in pyproject.toml")
    return f'v{m.group(1)}'


def build() -> str:
    return convert(SOURCE.read_text(), release_ref())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true',
                    help='exit with status 1 if README.pypi.md is out of date')
    args = ap.parse_args(argv)
    expected = build()
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != expected:
            print("README.pypi.md is out of date: run python scripts/build_pypi_readme.py",
                  file=sys.stderr)
            return 1
        print("README.pypi.md is up to date")
        return 0
    TARGET.write_text(expected)
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
