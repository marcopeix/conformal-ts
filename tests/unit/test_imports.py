"""Sanity test: package imports and version is set."""

import conformal_ts


def test_package_imports() -> None:
    assert conformal_ts.__version__
