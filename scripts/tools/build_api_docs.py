"""Generate API documentation via pdoc and emit HTML/PDF artifacts.

The script wraps ``pdoc`` so contributors only need to run one command to render HTML
documentation plus a lightweight PDF summary for smoke verification.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the API doc builder."""
    parser = argparse.ArgumentParser(
        description="Generate API reference docs using pdoc and write them to output/docs/api/."
    )
    parser.add_argument(
        "modules",
        nargs="*",
        default=["robot_sf"],
        help="Python modules/packages to document (defaults to 'robot_sf').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/docs/api"),
        help="Destination directory for generated artifacts (default: output/docs/api).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before generating docs.",
    )
    parser.add_argument(
        "--pdf-name",
        type=str,
        default="robot_sf_api.pdf",
        help="Name of the generated PDF summary file.",
    )
    return parser.parse_args()


def _run_pdoc(modules: list[str], html_dir: Path) -> None:
    """Invoke pdoc to build HTML documentation into ``html_dir``."""
    html_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pdoc",
        "--html",
        "--force",
        "--output-dir",
        str(html_dir),
        *modules,
    ]
    subprocess.run(cmd, check=True)


def _escape_pdf_text(text: str) -> str:
    """Escape parentheses and backslashes for literal PDF text."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_summary_pdf(pdf_path: Path, modules: list[str], html_dir: Path) -> None:
    """Write a tiny single-page PDF summarizing the generated HTML artifacts."""
    html_files: list[Path] = sorted(html_dir.rglob("*.html"))
    display_paths: list[str] = []
    base = html_dir.parent
    for path in html_files:
        try:
            display_paths.append(path.relative_to(base).as_posix())
        except ValueError:
            display_paths.append(path.as_posix())
    html_lines = [f" - {p}" for p in display_paths[:50]]
    if not html_lines:
        html_lines = [" - (no HTML files found)"]
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "Robot Social Force API Docs",
        f"Generated: {timestamp}",
        "",
        "Modules:",
        *[f" - {mod}" for mod in modules],
        "",
        "HTML files:",
        *html_lines,
        "",
        "See the HTML directory for full API content.",
    ]

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    stream_lines = ["BT", "/F1 12 Tf", "72 720 Td"]
    for idx, line in enumerate(lines):
        escaped = _escape_pdf_text(line)
        if idx == 0:
            stream_lines.append(f"({escaped}) Tj")
        else:
            stream_lines.append("T*")
            stream_lines.append(f"({escaped}) Tj")
    stream_lines.append("ET")
    stream_content = "\n".join(stream_lines).encode("utf-8")

    buffer = bytearray()
    buffer.extend(b"%PDF-1.4\n")
    offsets = [0]

    def add_object(content: bytes) -> int:
        obj_id = len(offsets)
        offsets.append(len(buffer))
        buffer.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        buffer.extend(content)
        if not content.endswith(b"\n"):
            buffer.extend(b"\n")
        buffer.extend(b"endobj\n")
        return obj_id

    add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_object(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add_object(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add_object(
        f"<< /Length {len(stream_content)} >>\nstream\n".encode("ascii")
        + stream_content
        + b"\nendstream"
    )

    xref_pos = len(buffer)
    buffer.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    buffer.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    buffer.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )

    pdf_path.write_bytes(buffer)


def main() -> None:
    """Entry point for the API doc generation helper."""
    args = _parse_args()
    output_dir = args.output_dir
    html_dir = output_dir / "html"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    _run_pdoc(args.modules, html_dir)
    pdf_path = output_dir / args.pdf_name
    _write_summary_pdf(pdf_path, args.modules, html_dir)

    print(f"HTML docs written to {html_dir}")
    print(f"PDF summary written to {pdf_path}")


if __name__ == "__main__":
    main()
