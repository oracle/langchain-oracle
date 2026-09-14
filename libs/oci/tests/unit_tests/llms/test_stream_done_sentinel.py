# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""OCIGenAI.stream() must skip the non-JSON ``data: [DONE]`` SSE frame."""

import json
from unittest.mock import MagicMock

from langchain_oci.llms.oci_generative_ai import OCIGenAI


def test_llm_stream_skips_done_sentinel() -> None:
    oci_client = MagicMock()
    llm = OCIGenAI(model_id="cohere.command", client=oci_client)
    response = MagicMock()
    response.data.events.return_value = [
        MagicMock(data=json.dumps({"text": "Hel"})),
        MagicMock(data=json.dumps({"text": "lo"})),
        MagicMock(data=""),
        MagicMock(data="[DONE]"),
    ]
    oci_client.generate_text.return_value = response

    chunks = list(llm.stream("hi"))

    assert "".join(c for c in chunks) == "Hello"
