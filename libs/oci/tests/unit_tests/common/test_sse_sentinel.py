# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""``is_sse_sentinel`` is the single guard shared by both sync stream loops."""

import pytest

from langchain_oci.common import is_sse_sentinel


@pytest.mark.parametrize("data", [None, "", "   ", "\n", "[DONE]", "  [DONE]\n"])
def test_sentinel_frames_are_skipped(data):
    assert is_sse_sentinel(data) is True


@pytest.mark.parametrize("data", ['{"text": "hi"}', "[DONE] trailing", "[done]", "0"])
def test_payload_frames_are_kept(data):
    assert is_sse_sentinel(data) is False


def test_both_stream_loops_use_the_shared_guard():
    import langchain_oci.chat_models.oci_generative_ai as chat_mod
    import langchain_oci.llms.oci_generative_ai as llm_mod

    assert chat_mod.is_sse_sentinel is is_sse_sentinel
    assert llm_mod.is_sse_sentinel is is_sse_sentinel
    assert not hasattr(chat_mod, "_is_sse_sentinel")
