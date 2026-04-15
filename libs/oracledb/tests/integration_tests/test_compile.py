# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest


@pytest.mark.compile
def test_compile_imports() -> None:
    """CI compile check: imports should succeed without DB access."""
    from langchain_oracledb.vectorstores.oraclevs import OracleVS

    assert OracleVS is not None
