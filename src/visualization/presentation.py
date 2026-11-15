"""Safe display helpers; original annotated text and offsets remain unchanged."""

import html
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


class _DisplayText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style"}:
            self.suppressed += 1
        elif tag in {"p", "br", "h1", "h2", "h3", "h4", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.suppressed = max(0, self.suppressed - 1)
        elif tag in {"p", "h1", "h2", "h3", "h4", "div", "li"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.suppressed:
            self.parts.append(data)


def display_text(value: str) -> str:
    """Strip display markup only; do not rewrite frozen corpus or entity spans."""
    parser = _DisplayText()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


def safe_json(value: object) -> str:
    """Embed JSON as inert script data without permitting an HTML closing-script token."""
    return (
        json.dumps(value, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def safe_evidence_url(value: str | None) -> str:
    """Allow navigable citations only for HTTP(S), escaping attribute delimiters."""
    if not value:
        return "#"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "#"
    return html.escape(value, quote=True)


def write_viewer_assets(destination: Path) -> None:
    """Copy the packaged static behaviour alongside either standalone or unified HTML."""
    (destination / "viewer.js").write_text((Path(__file__).parent / "assets/viewer.js").read_text())
