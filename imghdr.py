"""Compatibility shim for the removed stdlib ``imghdr`` module."""

from __future__ import annotations

from collections.abc import Callable
from typing import IO

__all__ = ["tests", "what"]

TestFunc = Callable[[bytes, str | None], str | None]


def _sniff(file: str | bytes | IO[bytes], h: bytes | None) -> tuple[bytes, str | None]:
    filename_hint = file if isinstance(file, str) else None
    if h is not None:
        return h, filename_hint
    if isinstance(file, str):
        with open(file, "rb") as handle:
            return handle.read(32), filename_hint
    if isinstance(file, bytes | bytearray):
        payload = bytes(file)
        return payload, filename_hint
    data = file.read(32)
    return data, filename_hint


def _test_jpeg(header: bytes, _f: str | None = None) -> str | None:
    if header[6:10] in (b"JFIF", b"Exif"):
        return "jpeg"
    if header.startswith(b"\xff\xd8"):
        return "jpeg"
    return None


def _test_png(header: bytes, _f: str | None = None) -> str | None:
    if header.startswith(b"\211PNG\r\n\032\n"):
        return "png"
    return None


def _test_gif(header: bytes, _f: str | None = None) -> str | None:
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def _test_tiff(header: bytes, _f: str | None = None) -> str | None:
    if header[:2] in (b"MM", b"II"):
        return "tiff"
    return None


def _test_rgb(header: bytes, _f: str | None = None) -> str | None:
    if header.startswith(b"\001\332"):
        return "rgb"
    return None


def _test_pbm(header: bytes, _f: str | None = None) -> str | None:
    if header.startswith(b"P4"):
        return "pbm"
    return None


def _test_pgm(header: bytes, _f: str | None = None) -> str | None:
    if header.startswith(b"P5"):
        return "pgm"
    return None


def _test_ppm(header: bytes, _f: str | None = None) -> str | None:
    if header.startswith(b"P6"):
        return "ppm"
    return None


def _test_rast(header: bytes, _f: str | None = None) -> str | None:
    if header.startswith(b"Y\xc6j\x95"):
        return "rast"
    return None


def _test_xbm(header: bytes, _f: str | None = None) -> str | None:
    if header[:9] == b"#define ":
        return "xbm"
    return None


def _test_bmp(header: bytes, _f: str | None = None) -> str | None:
    if header.startswith(b"BM"):
        return "bmp"
    return None


def _test_webp(header: bytes, _f: str | None = None) -> str | None:
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    return None


def _test_exr(header: bytes, _f: str | None = None) -> str | None:
    if header.startswith(b"\x76\x2f\x31\x01"):
        return "exr"
    return None


def _test_data_uri(header: bytes, _f: str | None = None) -> str | None:
    if header.startswith(b"data:image"):
        return "data"
    return None


tests: list[tuple[TestFunc, str | None]] = [
    (_test_jpeg, None),
    (_test_png, None),
    (_test_gif, None),
    (_test_tiff, None),
    (_test_rgb, None),
    (_test_pbm, None),
    (_test_pgm, None),
    (_test_ppm, None),
    (_test_rast, None),
    (_test_xbm, None),
    (_test_bmp, None),
    (_test_webp, None),
    (_test_exr, None),
    (_test_data_uri, None),
]


def what(file: str | bytes | IO[bytes], h: bytes | None = None) -> str | None:
    """Guess the type of an image file by sniffing its header."""

    header, filename_hint = _sniff(file, h)
    for test, declared in tests:
        outcome = test(header, filename_hint)
        if outcome:
            return outcome if declared is None else declared
    return None
