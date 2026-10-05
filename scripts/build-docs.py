#!/usr/bin/env python3
"""Build the bundled Markdown/HTML manual, with no third-party dependencies."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "manual"


def inline(value: str) -> str:
    escaped = html.escape(value, quote=True)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)", r'<a href="\2">\1</a>', escaped)
    return escaped


def render(markdown: str) -> tuple[str, str]:
    output: list[str] = []
    contents: list[str] = []
    lines = markdown.splitlines()
    ids: dict[str, int] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        heading = re.match(r"^(#{1,6}) (.+)$", line)
        if heading:
            level, text = len(heading[1]), heading[2]
            base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            ids[base] = ids.get(base, 0) + 1
            anchor = base if ids[base] == 1 else f"{base}-{ids[base]}"
            output.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            if level == 1:
                contents.append(f'<li><a href="#{anchor}">{inline(text)}</a></li>')
            index += 1
        elif line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            output.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
            index += 1
        elif line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[ :|\-]+\|$", lines[index + 1]):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            header = "<tr>" + "".join(f"<th scope=\"col\">{inline(cell)}</th>" for cell in rows[0]) + "</tr>"
            body = "".join("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>" for row in rows[2:])
            output.append('<div class="table-scroll"><table><thead>' + header + "</thead><tbody>" + body + "</tbody></table></div>")
        elif re.match(r"^(\d+\. |[-*] )", line):
            ordered = bool(re.match(r"^\d+\. ", line))
            pattern = r"^\d+\. (.+)$" if ordered else r"^[-*] (.+)$"
            entries = []
            while index < len(lines) and (entry := re.match(pattern, lines[index])):
                entries.append("<li>" + inline(entry[1]) + "</li>")
                index += 1
            tag = "ol" if ordered else "ul"
            output.append(f"<{tag}>" + "".join(entries) + f"</{tag}>")
        else:
            paragraph = []
            while index < len(lines) and lines[index].strip():
                paragraph.append(lines[index].strip())
                index += 1
            output.append("<p>" + inline(" ".join(paragraph)) + "</p>")
    return "\n".join(output), "\n".join(contents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated outputs differ")
    args = parser.parse_args()
    chapters = sorted((MANUAL / "chapters").glob("*.md"))
    if not chapters:
        parser.error("no manual chapters found")
    markdown = "\n\n".join(chapter.read_text(encoding="utf-8").strip() for chapter in chapters) + "\n"
    body, contents = render(markdown)
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IPDE Studio user manual</title><link rel="stylesheet" href="manual.css"></head>
<body><a class="skip" href="#manual-content">Skip to manual</a>
<header><a href="../help/index.html">IPDE Studio Help</a><span>User manual · 0.9 development series</span></header>
<main id="manual-content"><p class="print-note">Use your browser's Print command to print or save as PDF. <a href="manual.md" download>Download editable Markdown</a>.</p>
<nav aria-label="Manual chapters"><h2>Contents</h2><ol>{contents}</ol></nav>
{body}
</main><footer>IPDE · Development pre-release · <a href="https://github.com/gnaservicesinc/ipde">Repository</a> · <a href="https://github.com/gnaservicesinc/ipde/issues/new">Report a bug</a></footer>
</body></html>
'''
    outputs = {MANUAL / "manual.md": markdown, MANUAL / "index.html": page}
    stale = []
    for path, value in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != value:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(value, encoding="utf-8")
    if stale:
        print("Generated documentation is stale: " + ", ".join(stale))
        return 1
    print("Bundled manual is current." if args.check else "Built bundled HTML and Markdown manual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
