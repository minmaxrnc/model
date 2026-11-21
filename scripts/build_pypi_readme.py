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
GitHub) as markers. Block form — marker lines on their own, which the
generated file drops; indent them to the list-item content (e.g. 2 spaces)
to keep a list unbroken on GitHub:

    - Common item
      <!-- github-only -->
    - Item shown on GitHub only
      <!-- /github-only -->
      <!-- pypi-only
      - Item shown on PyPI only
      -->

Everything inside a pypi-only block must be indented at least as much as its
opening marker (that indentation is removed in the generated file): a less
indented line would close the enclosing list item on GitHub, leaving the
comment unterminated so that it hides the rest of the page. This is rejected.

Inline form, within a line:

    text <!-- github-only -->GitHub text<!-- /github-only --><!-- pypi-only PyPI text -->

An inline marker must not begin a line (or a list item's content): CommonMark
then treats the whole line as a raw HTML block and stops rendering Markdown in
it on GitHub, so this is rejected.

Only a small LaTeX subset is supported (see SYMBOLS / FUNCTIONS / FONTS);
anything else raises an error rather than producing broken output, and so
does plain $...$ math (use $`...`$, GitHub's unambiguous inline syntax).

Version references are kept in sync with pyproject.toml's `version`, the
single source of truth: before generating README.pypi.md, the script rewrites
the places listed in VERSIONED (the BibTeX `version = {...}` field in
README.md, `__version__` in the package) to the current version. Historical
mentions such as "New in 0.4.0" are deliberately not touched.

Usage (from the repository root):

    python scripts/build_pypi_readme.py           # sync versions, write README.pypi.md
    python scripts/build_pypi_readme.py --check   # exit 1 if anything is stale

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
        """Body of a {...} group, or a single token, as raw LaTeX (as in TeX,
        spaces before the argument are skipped: \\mathbb R == \\mathbb{R})."""
        while self.peek() is not None and self.peek().isspace():
            self.i += 1
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
            elif c.isascii() and c.isalpha():            # variables: ASCII letters only
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
                elif name in ('mathbb', 'Bbb'):
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

GITHUB_ONLY_BLOCK = re.compile(r'^[ \t]*<!--\s*github-only\s*-->[ \t]*\n.*?^[ \t]*<!--\s*/github-only\s*-->[ \t]*\n', re.S | re.M)
PYPI_ONLY_BLOCK   = re.compile(r'^[ \t]*<!--\s*pypi-only[ \t]*\n(.*?)^[ \t]*-->[ \t]*\n', re.S | re.M)
GITHUB_ONLY  = re.compile(r'<!--\s*github-only\s*-->.*?<!--\s*/github-only\s*-->', re.S)
PYPI_ONLY    = re.compile(r'<!--\s*pypi-only\s(.*?)-->', re.S)
MARKER_AT_LINE_START = re.compile(r'^[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]+)?<!--\s*/?(?:github|pypi)-only\b(?!\s*-->[ \t]*$)(?![ \t]*$)', re.M)
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


def _unwrap_pypi_block(m) -> str:
    """Content of a pypi-only block, without the opening marker's indentation."""
    indent = re.match(r'[ \t]*', m.group(0)).group(0)
    lines = m.group(1).splitlines(keepends=True)
    for line in lines:
        if line.strip() and not line.startswith(indent):
            raise LatexError(
                f"README.md: pypi-only block content {line.strip()!r} is indented less than "
                f"its '<!-- pypi-only' marker; on GitHub that closes the enclosing list "
                f"item and leaves the comment open, hiding the rest of the page"
            )
    return ''.join(l[len(indent):] if l.startswith(indent) else l for l in lines)


def convert(markdown: str, ref: str) -> str:
    """README.md (GitHub flavour, LaTeX math) → README.pypi.md."""
    markdown = GITHUB_ONLY_BLOCK.sub('', markdown)
    markdown = PYPI_ONLY_BLOCK.sub(_unwrap_pypi_block, markdown)
    bad = MARKER_AT_LINE_START.search(markdown)
    if bad:
        lineno = markdown.count('\n', 0, bad.start()) + 1
        raise LatexError(
            f"README.md: an inline github-only/pypi-only marker begins a line "
            f"(near generated line {lineno}); GitHub would render that line as raw "
            f"HTML. Put text before it, or use the block form."
        )
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


# --- version references ----------------------------------------------------

# (file, pattern, description): group 1 and 2 surround the version string.
VERSIONED = [
    (SOURCE,
     re.compile(r'^(\s*version\s*=\s*\{)[^}]*(\})', re.M),
     'BibTeX version field in README.md'),
    (ROOT / 'src' / 'minmaxrnc' / '__init__.py',
     re.compile(r'^(__version__\s*=\s*")[^"]*(")', re.M),
     '__version__ in src/minmaxrnc/__init__.py'),
]


def project_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.M)
    if not m:
        raise RuntimeError("version not found in pyproject.toml")
    return m.group(1)


def release_ref() -> str:
    return f'v{project_version()}'


def sync_version(text: str, pattern: re.Pattern, version: str, what: str) -> str:
    """Set every version reference matched by `pattern` in `text` to `version`."""
    new, n = pattern.subn(lambda m: f'{m.group(1)}{version}{m.group(2)}', text)
    if n == 0:
        raise RuntimeError(f"no {what} found to keep in sync with pyproject.toml")
    return new


def synced_files() -> dict:
    """{path: content with its version references synced} for every VERSIONED file."""
    version, contents = project_version(), {}
    for path, pattern, what in VERSIONED:
        contents[path] = sync_version(contents.get(path, path.read_text()), pattern, version, what)
    return contents


def build() -> str:
    """README.pypi.md content, from README.md with its version references synced."""
    return convert(synced_files()[SOURCE], release_ref())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true',
                    help='exit with status 1 if a version reference or README.pypi.md is out of date')
    args = ap.parse_args(argv)
    files = synced_files()
    files[TARGET] = convert(files[SOURCE], release_ref())
    stale = [p for p, c in files.items() if not p.exists() or p.read_text() != c]
    version = project_version()
    checked = (
        f"README.pypi.md matches README.md, and the version references "
        f"({'; '.join(what for _, _, what in VERSIONED)}) are at {version}"
    )

    def reason(p):
        if p == TARGET:
            return "regenerated from README.md"
        return f"version references set to {version} (pyproject.toml)"

    if args.check:
        for p in stale:
            need = ("must be regenerated from README.md" if p == TARGET
                    else f"version references must be set to {version} (pyproject.toml)")
            print(f"OUT OF DATE: {p.relative_to(ROOT)}: {need}; "
                  f"run python scripts/build_pypi_readme.py", file=sys.stderr)
        if not stale:
            print(f"OK: {checked}.")
        return 1 if stale else 0
    for p in stale:
        p.write_text(files[p])
        print(f"Updated {p.relative_to(ROOT)}: {reason(p)}.")
    if not stale:
        print(f"No changes needed: {checked}. Nothing was written.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
