# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import glob
import importlib
from pathlib import Path

import langchain_oci


def test_importable_all() -> None:
    for path in glob.glob("../langchain_oci/*"):
        relative_path = Path(path).parts[-1]
        if relative_path.endswith(".typed"):
            continue
        module_name = relative_path.split(".")[0]
        module = importlib.import_module("langchain_oci." + module_name)
        all_ = getattr(module, "__all__", [])
        for cls_ in all_:
            getattr(module, cls_)


def test_public_all_is_resolvable() -> None:
    """Every name advertised in the top-level ``__all__`` must resolve.

    ``from langchain_oci import *`` iterates ``__all__`` and resolves each name
    (via module globals or ``__getattr__``). A name that is listed but cannot be
    resolved makes ``import *`` raise ``AttributeError``. The guardrails agent
    middleware is the motivating case: it is absent on the Python 3.9 /
    langchain 0.3.x matrix (no ``langchain.agents.middleware``), so it must not
    appear in ``__all__`` there. Running on every supported version locks the
    contract — including the 3.9 CI job, where the middleware genuinely is gone.
    """
    for name in langchain_oci.__all__:
        getattr(langchain_oci, name)
