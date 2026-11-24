# Copyright (c) 2025, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import glob
import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def test_importable_all() -> None:
    for path in glob.glob("../langchain_oracledb/*"):
        relative_path = Path(path).parts[-1]
        if relative_path.endswith(".typed"):
            continue
        module_name = relative_path.split(".")[0]
        module = importlib.import_module("langchain_oracledb." + module_name)
        all_ = getattr(module, "__all__", [])
        for cls_ in all_:
            getattr(module, cls_)


@pytest.mark.parametrize(
    "module",
    [
        "langchain_oracledb.vectorstores.oraclevs",
        "langchain_oracledb.embeddings.oracleai",
    ],
)
def test_sql_imports_do_not_load_optional_vecdb_sdk(module):
    script = """
import importlib.abc
import sys

class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "oracle_vecdb" or fullname.startswith("oracle_vecdb."):
            raise AssertionError("SQL imports must not load the optional VecDB SDK")

sys.meta_path.insert(0, BlockSDK())
__import__(sys.argv[1])
"""
    subprocess.run([sys.executable, "-c", script, module], check=True)
