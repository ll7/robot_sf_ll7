"""Compatibility shim for the removed stdlib ``imghdr`` module."""

from __future__ import annotations

from collections.abc import Callable
from typing import IO

__all__ = ["what"]

TestFunc = Callable[[bytes], str | None]


def _test_jpeg(header: bytes) -> str | None:
    if header[6:10] in (b"JFIF", b"Exif"):
        return "jpeg"
    if header.startswith(b"\xff\xd8"):
        return "jpeg"
    return None


def _test_png(header: bytes) -> str | None:
    if header.startswith(b"\211PNG\r\n\032\n"):
        return "png"
    return None


def _test_gif(header: bytes) -> str | None:
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def _test_tiff(header: bytes) -> str | None:
    if header[:2] in (b"MM", b"II"):
        return "tiff"
    return None


def _test_rgb(header: bytes) -> str | None:
    if header.startswith(b"\001\332"):
        return "rgb"
    return None


def _test_pbm(header: bytes) -> str | None:
    if header.startswith(b"P4"):
        return "pbm"
    return None


def _test_pgm(header: bytes) -> str | None:
    if header.startswith(b"P5"):
        return "pgm"
    return None


def _test_ppm(header: bytes) -> str | None:
    if header.startswith(b"P6"):
        return "ppm"
    return None


def _test_rast(header: bytes) -> str | None:
    if header.startswith(b"Y\xc6j\x95"):
        return "rast"
    return None


def _test_xbm(header: bytes) -> str | None:
    if header[:9] == b"#define ":
        return "xbm"
    return None


def _test_bmp(header: bytes) -> str | None:
    if header.startswith(b"BM"):
        return "bmp"
    return None


def _test_webp(header: bytes) -> str | None:
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    return None


def _test_exr(header: bytes) -> str | None:
    if header.startswith(b"\x76\x2f\x31\x01"):
        return "exr"
    return None


def _test_data_uri(header: bytes) -> str | None:
    if header.startswith(b"data:image"):
        return "data"
    return None


def _load_header(file: str | bytes | IO[bytes], h: bytes | None) -> bytes:
    if h is not None:
        return h
    if isinstance(file, bytes) or isinstance(file, bytearray):
        return bytes(file)
    with open(file, "rb") as handle:
        return handle.read(32)


_TESTS: tuple[TestFunc, ...] = (
    _test_jpeg,
    _test_png,
    _test_gif,
    _test_tiff,
    _test_rgb,
    _test_pbm,
    _test_pgm,
    _test_ppm,
    _test_rast,
    _test_xbm,
    _test_bmp,
    _test_webp,
    _test_exr,
    _test_data_uri,
)


def what(file: str | bytes | IO[bytes], h: bytes | None = None) -> str | None:
    """Guess the type of an image file by sniffing its header."""

    header = _load_header(file, h)
    for test in _TESTS:
        result = test(header)
        if result:
            return result
    return None
