from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize


def _macos_sdkroot() -> str | None:
    import os
    import sys

    if sys.platform != "darwin":
        return None
    try:
        import subprocess

        sdkroot = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
        ).strip()
    except Exception:
        return None
    if sdkroot and os.path.exists(sdkroot):
        return sdkroot
    return None


def _ensure_macos_sdk() -> None:
    import os
    import sys

    if sys.platform != "darwin":
        return

    sdkroot = os.environ.get("SDKROOT")
    if sdkroot and os.path.exists(sdkroot):
        return

    try:
        import subprocess

        sdkroot = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
        ).strip()
    except Exception:
        return

    if sdkroot and os.path.exists(sdkroot):
        os.environ["SDKROOT"] = sdkroot
        os.environ.setdefault("CMAKE_OSX_SYSROOT", sdkroot)


class BuildRvo2Ext(_build_ext):
    """Builds RVO2 before our module."""

    def run(self):
        # Build RVO2
        import os
        import os.path
        import subprocess

        _ensure_macos_sdk()
        build_dir = os.path.abspath('build/RVO2')
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
            subprocess.check_call(['cmake', '../..', '-DCMAKE_CXX_FLAGS=-fPIC'],
                                  cwd=build_dir)
        subprocess.check_call(['cmake', '--build', '.'], cwd=build_dir)

        _build_ext.run(self)


_SDKROOT = _macos_sdkroot()
_SDKROOT_ARGS = ['-isysroot', _SDKROOT] if _SDKROOT else []

extensions = [
    Extension(
        'rvo2',
        ['src/*.pyx'],
        include_dirs=['src'],
        libraries=['RVO'],
        library_dirs=['build/RVO2/src'],
        extra_compile_args=['-fPIC', *_SDKROOT_ARGS],
        extra_link_args=[*_SDKROOT_ARGS],
    ),
]

setup(
    name="pyrvo2",
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': BuildRvo2Ext},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Cython',
        'Topic :: Games/Entertainment :: Simulation',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
