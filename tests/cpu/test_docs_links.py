# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching test: the docs' links and anchors resolve and their Python examples parse.

Files in this repository move: test oracles went from ``python/pth_flash_attn``
to ``python/pth_testing`` and pages are renamed. A move that forgets
one inbound link leaves a reader (or a reviewer reading the repository cold) at
a dead end, and nothing else in CI would notice. This module checks three things.

1. Every markdown link ``[text](target)`` in the live documents (``README.md``,
   ``CONTRIBUTING.md``, ``SECURITY.md``, ``CHANGELOG.md``, ``docs/**/*.md`` and the
   READMEs under ``bench/``) names an existing file or directory when the target
   is relative.
2. Every ``#anchor`` of such a link (``#section`` within the file, ``file.md#section``
   across files) names a heading of the target markdown file, using GitHub's rule
   for heading slugs (lower case; punctuation other than hyphens and underscores
   removed; spaces become hyphens; a repeated heading gets ``-1``, ``-2``, ...),
   or an explicit ``<a id="...">``/``<a name="...">`` anchor.
3. Every fenced ``python`` block of the live documents compiles. Compiling does
   not run the code (many examples need a GPU and say so), but it catches the
   typos and stale syntax that make an example unusable.

Out of scope, on purpose:

* external links (``http:``, ``https:``, ``mailto:``, any ``scheme:``): checking
  them needs the network; anchors of non-markdown targets (``file.py#L10``) are
  not checked either.
* links inside fenced code blocks and inline code spans, which are examples,
  not links.

The negative controls write a markdown file with one broken link and one broken
anchor among valid ones, and a snippet with a syntax error, and check that
exactly those are reported, so a checker that never finds anything cannot pass.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[2]

# Pages whose Python examples are meant to be copied by users.
USER_PAGES = (
    "docs/api.md",
    "docs/architecture.md",
    "docs/build.md",
    "docs/faq.md",
    "docs/integration.md",
    "docs/troubleshooting.md",
    "docs/vllm.md",
)

# [text](target) or ![alt](target), with an optional "title". The text may be
# empty (it is after inline code is stripped from [`path`](path)).
_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
_FENCE = re.compile(r"^\s*(```|~~~)\s*([A-Za-z0-9_+-]*)")
_CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_INLINE_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]*>")
_HTML_ANCHOR = re.compile(r"""<a\s[^>]*?\b(?:id|name)\s*=\s*["']([^"']+)["']""")
_MARKDOWN_SUFFIXES = (".md", ".markdown")


def markdown_links(text: str) -> list[tuple[int, str]]:
    """Return (line number, target) of every link outside code."""
    links: list[tuple[int, str]] = []
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        opener = _FENCE.match(line)
        if opener:
            if fence is None:
                fence = opener.group(1)
            elif opener.group(1) == fence:
                fence = None
            continue
        if fence is not None:
            continue
        for match in _LINK.finditer(_CODE_SPAN.sub("", line)):
            links.append((number, match.group(1).strip("<>")))
    return links


def python_blocks(text: str) -> list[tuple[int, str]]:
    """Return (first line number, source) of every fenced ``python`` block."""
    blocks: list[tuple[int, str]] = []
    fence: str | None = None
    language = ""
    start = 0
    body: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        opener = _FENCE.match(line)
        if opener and fence is None:
            fence, language = opener.group(1), opener.group(2)
            start, body = number, []
        elif opener and opener.group(1) == fence and not opener.group(2):
            if language.lower() in ("python", "py"):
                blocks.append((start + 1, "\n".join(body) + "\n"))
            fence = None
        elif fence is not None:
            body.append(line)
    return blocks


def heading_slug(heading: str) -> str:
    """GitHub's anchor of a heading: lower case, no punctuation, spaces as ``-``."""
    text = _HTML_TAG.sub("", _INLINE_LINK.sub(r"\1", heading)).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors_of(text: str) -> set[str]:
    """Return every anchor of a markdown file: heading slugs and ``<a id>`` names."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    fence: str | None = None
    for line in text.splitlines():
        opener = _FENCE.match(line)
        if opener:
            if fence is None:
                fence = opener.group(1)
            elif opener.group(1) == fence:
                fence = None
            continue
        if fence is not None:
            continue
        anchors.update(_HTML_ANCHOR.findall(line))
        heading = _HEADING.match(line)
        if heading:
            slug = heading_slug(heading.group(2))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            anchors.add(slug if count == 0 else f"{slug}-{count}")
    return {a.lower() for a in anchors}


def broken_anchors(md_file: Path, root: Path) -> list[str]:
    """Return ``file:line: target`` for each ``#anchor`` that names no heading.

    Only markdown targets that exist are checked (a missing file is reported by
    :func:`broken_links`); external links are skipped.
    """
    problems = []
    cache: dict[Path, set[str]] = {}
    for number, target in markdown_links(md_file.read_text(encoding="utf-8")):
        if _SCHEME.match(target) or "#" not in target:
            continue
        path, _, fragment = target.partition("#")
        if not fragment:
            continue
        path = unquote(path)
        if not path:
            resolved = md_file
        elif path.startswith("/"):
            resolved = root / path.lstrip("/")
        else:
            resolved = md_file.parent / path
        if not resolved.is_file() or resolved.suffix not in _MARKDOWN_SUFFIXES:
            continue
        if resolved not in cache:
            cache[resolved] = anchors_of(resolved.read_text(encoding="utf-8"))
        if unquote(fragment).lower() not in cache[resolved]:
            problems.append(f"{md_file.relative_to(root)}:{number}: {target}")
    return problems


def broken_links(md_file: Path, root: Path) -> list[str]:
    """Return ``file:line: target`` for each relative link that does not resolve.

    A target starting with ``/`` is taken relative to ``root`` (the repository
    root, as on a code-hosting site); any other relative target relative to the
    file's directory.
    """
    problems = []
    for number, target in markdown_links(md_file.read_text(encoding="utf-8")):
        if target.startswith("#") or _SCHEME.match(target):
            continue
        path = unquote(target.split("#", 1)[0])
        if not path:
            continue
        if path.startswith("/"):
            resolved = root / path.lstrip("/")
        else:
            resolved = md_file.parent / path
        if not resolved.exists():
            problems.append(f"{md_file.relative_to(root)}:{number}: {target}")
    return problems


def unparsable_snippets(md_file: Path, root: Path) -> list[str]:
    """Return ``file:line: error`` for each ``python`` block that does not compile."""
    problems = []
    text = md_file.read_text(encoding="utf-8")
    for number, source in python_blocks(text):
        try:
            compile(source, str(md_file), "exec")
        except SyntaxError as err:
            problems.append(f"{md_file.relative_to(root)}:{number}: {err.msg}")
    return problems


def live_markdown_files() -> list[Path]:
    """The root documents, every markdown file below docs/ and the READMEs of bench/."""
    root_docs = ["README.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md"]
    return [
        *(REPO / name for name in root_docs),
        *sorted((REPO / "docs").rglob("*.md")),
        *sorted((REPO / "bench").rglob("*.md")),
    ]


def user_page_files() -> list[Path]:
    """The pages whose Python examples users copy."""
    return [REPO / "README.md", *(REPO / page for page in USER_PAGES)]


def test_scan_covers_the_entry_points():
    """The scan reads the documents a reader starts from."""
    files = set(live_markdown_files())
    for entry in ("README.md", "docs/README.md", "docs/architecture.md"):
        assert REPO / entry in files, f"{entry} is not scanned"
    assert REPO / "docs" / "architecture.md" in files
    assert REPO / "docs" / "build.md" in files
    for entry in ("CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "bench/README.md"):
        assert REPO / entry in files, f"{entry} is not scanned"


def test_every_relative_link_in_live_docs_resolves():
    """No live document links to a file that moved or was deleted."""
    problems = [p for f in live_markdown_files() for p in broken_links(f, REPO)]
    assert not problems, "broken relative links:\n" + "\n".join(problems)


def test_every_anchor_in_live_docs_resolves():
    """No live document links to a heading that was renamed or removed."""
    problems = [p for f in live_markdown_files() for p in broken_anchors(f, REPO)]
    assert not problems, "broken anchors:\n" + "\n".join(problems)


def test_python_examples_in_user_pages_compile():
    """Every fenced python block of the user-facing pages is valid Python."""
    files = user_page_files()
    missing = [f for f in files if not f.is_file()]
    assert not missing, f"user pages are missing: {missing}"
    problems = [p for f in files for p in unparsable_snippets(f, REPO)]
    assert not problems, "python examples that do not parse:\n" + "\n".join(problems)
    assert sum(len(python_blocks(f.read_text(encoding="utf-8"))) for f in files) >= 5


def test_every_python_example_in_live_docs_compiles():
    """Every fenced python block of every live document is valid Python."""
    problems = [p for f in live_markdown_files() for p in unparsable_snippets(f, REPO)]
    assert not problems, "python examples that do not parse:\n" + "\n".join(problems)


def test_broken_link_is_reported_negative_control(tmp_path):
    """A file with one broken link among valid ones reports exactly that link."""
    (tmp_path / "exists.md").write_text("# target\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    doc = tmp_path / "doc.md"
    doc.write_text(
        "\n".join(
            [
                "[ok](exists.md) and [ok dir](sub/) and [ok anchor](exists.md#target)",
                "[`code text`](exists.md) [root-relative](/exists.md)",
                "[web](https://example.com/missing.md) [mail](mailto:a@b.c) [top](#x)",
                "`[in code](missing-in-span.md)`",
                "```",
                "[in fence](missing-in-fence.md)",
                "```",
                "see [moved](old/place.md#section)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert broken_links(doc, tmp_path) == ["doc.md:8: old/place.md#section"]


def test_broken_anchor_is_reported_negative_control(tmp_path):
    """Wrong, renamed and cross-file anchors are reported; valid forms pass."""
    (tmp_path / "other.md").write_text(
        '# Other page\n\n## Same, same\n\n## Same, same\n\n<a id="named"></a>\n'
        "```\n# not a heading\n```\n",
        encoding="utf-8",
    )
    doc = tmp_path / "doc.md"
    doc.write_text(
        "\n".join(
            [
                "# Title: with `code` and [a link](x.md) (2)",
                "[ok](#title-with-code-and-a-link-2) [ok](other.md#other-page)",
                "[ok](other.md#same-same) [ok](other.md#same-same-1)",
                "[ok](other.md#named)",
                "[web](https://example.com/page.md#nothing) [file](doc.py#L10)",
                "[gone](#renamed) [wrong](other.md#same-same-2)",
                "[fence](other.md#not-a-heading)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert broken_anchors(doc, tmp_path) == [
        "doc.md:6: #renamed",
        "doc.md:6: other.md#same-same-2",
        "doc.md:7: other.md#not-a-heading",
    ]


def test_unparsable_snippet_is_reported_negative_control(tmp_path):
    """One bad python block among good ones and other languages is reported."""
    doc = tmp_path / "doc.md"
    doc.write_text(
        "\n".join(
            [
                "```python",
                "x = 1",
                "```",
                "```bash",
                "this is not python (",
                "```",
                "```python",
                "def broken(:",
                "    pass",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert [n for n, _ in python_blocks(doc.read_text(encoding="utf-8"))] == [2, 8]
    problems = unparsable_snippets(doc, tmp_path)
    assert len(problems) == 1 and problems[0].startswith("doc.md:8:")
