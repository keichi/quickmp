# quickmp

[![Run unit tests](https://github.com/keichi/quickmp/actions/workflows/python-package.yml/badge.svg)](https://github.com/keichi/quickmp/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/quickmp)](https://pypi.org/project/quickmp/)
[![Documentation Status](https://readthedocs.org/projects/quickmp/badge/?version=latest)](https://quickmp.readthedocs.io/en/latest/?badge=latest)

quickmp is a high-performance matrix profile library for time series data with CPU and NEC Vector
Engine (VE) backends. See the [documentation](https://quickmp.readthedocs.io/) for usage and the
API reference.

## Installation

Pre-built wheels (CPU backend only) are available for Linux (x86-64) and macOS 14 or later (Apple
silicon):

```bash
pip install quickmp
```

## Building from Source

```bash
git clone https://github.com/keichi/quickmp.git
cd quickmp
pip install -e .
```

The backend is selected at build time: if [VEDA](https://github.com/SX-Aurora/veda) is found, the
VE backend is built; otherwise the CPU backend is built.

- **CPU**: OpenMP is used if available. On macOS, install libomp with Homebrew
  (`brew install libomp`) beforehand; otherwise quickmp runs on a single thread.
- **VE**: Requires VEDA and the NEC compiler (`nc++`). Build on a host with the NEC SDK (e.g., a
  Vector Host or a front-end server of an SX-Aurora TSUBASA system).

Extra CMake options can be passed through `CMAKE_ARGS`, e.g.,
`CMAKE_ARGS="-DOpenMP_ROOT=/path/to/libomp" pip install -e .`. The default release flags
(`-march=native -ffast-math`) can be overridden with the `RELEASE_FLAGS` environment variable.

## Testing

```bash
pip install -e '.[test]'
pytest -v
```

On VE, run the tests on a host where Vector Engines are available (e.g., in a batch job).
