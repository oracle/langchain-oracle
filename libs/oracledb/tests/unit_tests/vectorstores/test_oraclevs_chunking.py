# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Unit tests for document chunking support in OracleVS."""

from types import MethodType
from typing import Any, List, cast

from langchain_core.documents import Document

from langchain_oracledb.vectorstores.oraclevs import OracleVS


class _SimpleSplitter:
    def split_text(self, text: str) -> List[str]:
        return [part for part in text.split("|") if part]


def test_prepare_texts_from_documents_without_splitter() -> None:
    docs = [
        Document(page_content="alpha", metadata={"doc_id": "A"}),
        Document(page_content="beta", metadata={"doc_id": "B"}),
    ]

    texts, metadatas, source_doc_indices = OracleVS._prepare_texts_from_documents(docs)

    assert texts == ["alpha", "beta"]
    assert metadatas == [{"doc_id": "A"}, {"doc_id": "B"}]
    assert source_doc_indices == [0, 1]


def test_prepare_texts_from_documents_with_splitter_adds_chunk_metadata() -> None:
    docs = [Document(page_content="a1|a2", metadata={"doc_id": "A"})]
    splitter = _SimpleSplitter()

    texts, metadatas, source_doc_indices = OracleVS._prepare_texts_from_documents(
        docs, text_splitter=splitter, add_chunk_metadata=True
    )

    assert texts == ["a1", "a2"]
    assert source_doc_indices == [0, 0]
    assert metadatas == [
        {"doc_id": "A", "source_doc_index": 0, "chunk_index": 0},
        {"doc_id": "A", "source_doc_index": 0, "chunk_index": 1},
    ]


def test_add_documents_expands_ids_per_chunk() -> None:
    docs = [
        Document(page_content="a1|a2", metadata={"doc_id": "A"}),
        Document(page_content="b1", metadata={"doc_id": "B"}),
    ]
    splitter = _SimpleSplitter()
    captured: dict[str, Any] = {}

    def _fake_add_texts(self: OracleVS, texts, metadatas=None, *, ids=None, **kwargs):
        captured["texts"] = texts
        captured["metadatas"] = metadatas
        captured["ids"] = ids
        captured["kwargs"] = kwargs
        return ids

    vs = OracleVS.__new__(OracleVS)
    cast(Any, vs).add_texts = MethodType(_fake_add_texts, vs)

    returned_ids = vs.add_documents(
        docs, text_splitter=splitter, ids=["doc-A", "doc-B"]
    )

    assert captured["texts"] == ["a1", "a2", "b1"]
    assert captured["ids"] == ["doc-A#chunk-0", "doc-A#chunk-1", "doc-B#chunk-0"]
    assert returned_ids == ["doc-A#chunk-0", "doc-A#chunk-1", "doc-B#chunk-0"]


def test_add_documents_extracts_ids_from_doc_id() -> None:
    """When ids kwarg is not given, extract from Document.id (base class behaviour)."""
    docs = [
        Document(page_content="alpha", metadata={}, id="id-A"),
        Document(page_content="beta", metadata={}, id="id-B"),
    ]
    captured: dict[str, Any] = {}

    def _fake_add_texts(self: OracleVS, texts, metadatas=None, *, ids=None, **kwargs):
        captured["ids"] = ids
        return ids

    vs = OracleVS.__new__(OracleVS)
    cast(Any, vs).add_texts = MethodType(_fake_add_texts, vs)

    vs.add_documents(docs)

    assert captured["ids"] == ["id-A", "id-B"]


def test_add_documents_kwargs_ids_take_precedence_over_doc_id() -> None:
    """Explicit ids kwarg should win over Document.id."""
    docs = [
        Document(page_content="alpha", metadata={}, id="id-A"),
        Document(page_content="beta", metadata={}, id="id-B"),
    ]
    captured: dict[str, Any] = {}

    def _fake_add_texts(self: OracleVS, texts, metadatas=None, *, ids=None, **kwargs):
        captured["ids"] = ids
        return ids

    vs = OracleVS.__new__(OracleVS)
    cast(Any, vs).add_texts = MethodType(_fake_add_texts, vs)

    vs.add_documents(docs, ids=["explicit-A", "explicit-B"])

    assert captured["ids"] == ["explicit-A", "explicit-B"]
