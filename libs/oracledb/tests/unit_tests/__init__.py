# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""All unit tests (lightweight tests)."""

from typing import Any


def assert_all_importable(module: Any) -> None:
    for attr in module.__all__:
        getattr(module, attr)
