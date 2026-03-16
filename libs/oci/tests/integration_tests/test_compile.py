# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest


@pytest.mark.compile
def test_compile_imports() -> None:
    """CI compile check: imports should succeed without running real integrations."""
    from langchain_oci import OCIGenAIEmbeddings, create_deep_research_agent
    from langchain_oci.agents.datastores.vectorstores import ADB, OpenSearch

    assert OCIGenAIEmbeddings is not None
    assert create_deep_research_agent is not None
    assert ADB is not None
    assert OpenSearch is not None
