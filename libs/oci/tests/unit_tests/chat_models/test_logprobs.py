# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""logprobs / top_logprobs support in ChatOCIGenAI (issue #265)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.chat_models.providers.cohere import CohereProvider
from langchain_oci.chat_models.providers.generic import (
    GenericProvider,
    OpenAIProvider,
    normalize_logprobs,
    normalize_logprobs_params,
)


class TestNormalizeLogprobsParams:
    def test_logprobs_true_requests_one_top_token(self) -> None:
        assert normalize_logprobs_params({"logprobs": True}) == {"log_probs": 1}

    def test_top_logprobs_sets_count(self) -> None:
        assert normalize_logprobs_params({"logprobs": True, "top_logprobs": 3}) == {
            "log_probs": 3
        }
        assert normalize_logprobs_params({"top_logprobs": 2}) == {"log_probs": 2}

    def test_false_or_absent_sends_nothing(self) -> None:
        assert normalize_logprobs_params({"logprobs": False, "max_tokens": 5}) == {
            "max_tokens": 5
        }
        assert normalize_logprobs_params({"max_tokens": 5}) == {"max_tokens": 5}

    def test_top_logprobs_with_logprobs_false_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="top_logprobs"):
            normalize_logprobs_params({"logprobs": False, "top_logprobs": 2})

    def test_explicit_log_probs_passes_through(self) -> None:
        assert normalize_logprobs_params({"log_probs": 4, "logprobs": True}) == {
            "log_probs": 4
        }

    def test_generic_and_openai_providers_apply_mapping(self) -> None:
        assert GenericProvider().normalize_params({"logprobs": True}) == {
            "log_probs": 1
        }
        assert OpenAIProvider().normalize_params(
            {"max_tokens": 9, "logprobs": True, "top_logprobs": 2}
        ) == {"max_completion_tokens": 9, "log_probs": 2}

    def test_cohere_rejects_logprobs_clearly(self) -> None:
        with pytest.raises(ValueError, match="Cohere .* do not support logprobs"):
            CohereProvider().normalize_params({"logprobs": True})
        with pytest.raises(ValueError, match="Cohere"):
            CohereProvider().normalize_params({"log_probs": 2})
        # logprobs=False is a no-op, not an error
        assert CohereProvider().normalize_params({"logprobs": False, "k": 1}) == {
            "k": 1
        }


class TestNormalizeLogprobs:
    def test_sdk_object(self) -> None:
        raw = SimpleNamespace(
            tokens=["Hi", "!"],
            token_logprobs=[-0.02, -0.03],
            top_logprobs=[{"Hi": "-0.02", "Hello": "-3.9"}, {"!": "-0.03"}],
            text_offset=[0, 2],
        )
        assert normalize_logprobs(raw) == {
            "tokens": ["Hi", "!"],
            "token_logprobs": [-0.02, -0.03],
            "top_logprobs": [{"Hi": -0.02, "Hello": -3.9}, {"!": -0.03}],
            "text_offset": [0, 2],
        }

    def test_camel_case_json(self) -> None:
        raw = {
            "tokens": ["Hi"],
            "tokenLogprobs": [-0.5],
            "topLogprobs": [{"Hi": "-0.5", "Hey": "-2.0"}],
            "textOffset": [0],
        }
        out = normalize_logprobs(raw)
        assert out is not None
        assert out["token_logprobs"] == [-0.5]
        assert out["top_logprobs"] == [{"Hi": -0.5, "Hey": -2.0}]
        assert out["text_offset"] == [0]

    def test_absent(self) -> None:
        assert normalize_logprobs(None) is None
        assert normalize_logprobs({}) is None
        assert normalize_logprobs(SimpleNamespace()) is None


def _llm() -> ChatOCIGenAI:
    return ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=MagicMock())


class TestGenerationInfo:
    def test_sync_response_surfaces_logprobs(self) -> None:
        provider = GenericProvider()
        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(reasoning_content=None),
            logprobs=SimpleNamespace(
                tokens=["a"],
                token_logprobs=[-0.1],
                top_logprobs=[{"a": "-0.1"}],
                text_offset=[0],
            ),
        )
        response = SimpleNamespace(
            data=SimpleNamespace(
                chat_response=SimpleNamespace(
                    choices=[choice], time_created="t", usage=None
                )
            )
        )
        info = provider.chat_generation_info(response)
        assert info["logprobs"]["tokens"] == ["a"]
        assert info["logprobs"]["top_logprobs"] == [{"a": -0.1}]

    def test_sync_response_without_logprobs_is_unchanged(self) -> None:
        provider = GenericProvider()
        choice = SimpleNamespace(finish_reason="stop", message=None, logprobs=None)
        response = SimpleNamespace(
            data=SimpleNamespace(
                chat_response=SimpleNamespace(
                    choices=[choice], time_created="t", usage=None
                )
            )
        )
        assert "logprobs" not in provider.chat_generation_info(response)

    def test_stream_event_without_logprobs(self) -> None:
        assert GenericProvider().chat_stream_generation_info(
            {"finishReason": "stop"}
        ) == {"finish_reason": "stop"}

    def test_stream_event_with_logprobs_is_normalised(self) -> None:
        info = GenericProvider().chat_stream_generation_info(
            {
                "finishReason": "stop",
                "logprobs": {"tokens": ["x"], "tokenLogprobs": [-1]},
            }
        )
        assert info["logprobs"]["token_logprobs"] == [-1.0]

    def test_async_dict_response_surfaces_logprobs(self) -> None:
        llm = _llm()
        info = llm._extract_generation_info(
            {
                "chatResponse": {
                    "finishReason": "stop",
                    "choices": [
                        {
                            "message": {"content": [{"type": "TEXT", "text": "hi"}]},
                            "logprobs": {
                                "tokens": ["hi"],
                                "tokenLogprobs": [-0.2],
                                "topLogprobs": [{"hi": "-0.2"}],
                            },
                        }
                    ],
                }
            }
        )
        assert info["logprobs"] == {
            "tokens": ["hi"],
            "token_logprobs": [-0.2],
            "top_logprobs": [{"hi": -0.2}],
            "text_offset": None,
        }

    def test_request_carries_log_probs(self) -> None:
        llm = ChatOCIGenAI(
            model_id="meta.llama-3.3-70b-instruct",
            client=MagicMock(),
            model_kwargs={"logprobs": True, "top_logprobs": 2},
        )
        request = llm._prepare_request(
            [HumanMessage(content="hi")],
            stop=None,
            stream=False,
        )
        assert request.chat_request.log_probs == 2
