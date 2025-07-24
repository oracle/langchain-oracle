# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import glob
import importlib
from pathlib import Path


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
