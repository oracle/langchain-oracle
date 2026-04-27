# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OpenSearch datastore backed by LangChain vector store/retriever contracts."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict

from langchain_oci.datastores.vectorstores.base import VectorDataStore


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _coerce_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _looks_like_json_blob(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _extract_content(source: Mapping[str, Any]) -> str:
    direct_content = _coerce_text(source.get("content"))
    if direct_content:
        return direct_content

    text_content = _coerce_text(source.get("text"))
    if text_content:
        return text_content

    metadata = _coerce_metadata(source.get("metadata"))
    metadata_content = _coerce_text(metadata.get("content"))
    if metadata_content and not _looks_like_json_blob(metadata_content):
        return metadata_content

    return metadata_content or text_content


def _normalize_source(
    source: Mapping[str, Any],
    *,
    vector_field: str,
    document_id: str,
) -> dict[str, Any]:
    raw_source = dict(source)
    raw_source.pop(vector_field, None)
    metadata = _coerce_metadata(raw_source.pop("metadata", None))

    content = _extract_content(source)
    title = (
        _coerce_text(raw_source.get("title"))
        or _coerce_text(metadata.get("title"))
        or "Untitled"
    )
    source_path = (
        _coerce_text(raw_source.get("source"))
        or _coerce_text(raw_source.get("source_path"))
        or _coerce_text(metadata.get("source"))
        or _coerce_text(metadata.get("source_path"))
    )

    normalized = {**metadata, **raw_source}
    normalized.pop("text", None)
    normalized.pop("content", None)
    normalized["id"] = document_id
    normalized["title"] = title
    normalized["content"] = content
    normalized["source"] = source_path
    return normalized


class _OpenSearchVectorStore(VectorStore):
    """Minimal LangChain vector store adapter over an OpenSearch index."""

    def __init__(
        self,
        *,
        client: Any,
        embedding_model: Any,
        index_name: str,
        vector_field: str,
    ) -> None:
        self._client = client
        self._embedding_model = embedding_model
        self._index_name = index_name
        self._vector_field = vector_field

    @property
    def embeddings(self) -> Any:
        return self._embedding_model

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        texts_list = list(texts)
        metadata_list = metadatas or [{} for _ in texts_list]
        ids_list = ids or [str(uuid.uuid4()) for _ in texts_list]
        embeddings = self._embedding_model.embed_documents(texts_list)

        bulk_body: list[dict[str, Any]] = []
        for doc_id, text, metadata, embedding in zip(
            ids_list, texts_list, metadata_list, embeddings
        ):
            source = {**metadata, "content": text, self._vector_field: embedding}
            bulk_body.append(
                {
                    "index": {
                        "_index": self._index_name,
                        "_id": doc_id,
                    }
                }
            )
            bulk_body.append(source)

        self._client.bulk(body=bulk_body, refresh=True)
        return ids_list

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> "_OpenSearchVectorStore":
        client = kwargs.pop("client")
        index_name = kwargs.pop("index_name")
        vector_field = kwargs.pop("vector_field", "embedding")
        store = cls(
            client=client,
            embedding_model=embedding,
            index_name=index_name,
            vector_field=vector_field,
        )
        store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return None

        for document_id in ids:
            self._client.delete(
                index=self._index_name,
                id=str(document_id),
                refresh=True,
            )
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        docs: list[Document] = []
        for document_id in ids:
            try:
                response = self._client.get(index=self._index_name, id=str(document_id))
            except Exception:
                continue

            if not response.get("found"):
                continue

            normalized = _normalize_source(
                response.get("_source", {}),
                vector_field=self._vector_field,
                document_id=response["_id"],
            )
            docs.append(
                Document(
                    page_content=str(normalized.pop("content", "")),
                    metadata=normalized,
                )
            )
        return docs

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return [
            doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        embedding = self._embedding_model.embed_query(query)
        search_body = {
            "size": k,
            "query": {"knn": {self._vector_field: {"vector": embedding, "k": k}}},
            "_source": {"excludes": [self._vector_field]},
        }
        response = self._client.search(index=self._index_name, body=search_body)
        hits = response.get("hits", {}).get("hits", [])

        docs_and_scores = []
        for hit in hits:
            normalized = _normalize_source(
                hit.get("_source", {}),
                vector_field=self._vector_field,
                document_id=hit["_id"],
            )
            docs_and_scores.append(
                (
                    Document(
                        page_content=str(normalized.pop("content", "")),
                        metadata=normalized,
                    ),
                    hit.get("_score", 0.0),
                )
            )
        return docs_and_scores


class _OpenSearchKeywordRetriever(BaseRetriever):
    """Minimal keyword retriever over an OpenSearch index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any
    index_name: str
    search_fields: list[str]
    vector_field: str
    k: int = 4

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        search_body = {
            "size": self.k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": self.search_fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
            "_source": {"excludes": [self.vector_field]},
        }
        response = self.client.search(index=self.index_name, body=search_body)
        hits = response.get("hits", {}).get("hits", [])

        documents = []
        for hit in hits:
            normalized = _normalize_source(
                hit.get("_source", {}),
                vector_field=self.vector_field,
                document_id=hit["_id"],
            )
            normalized["score"] = hit.get("_score", 0.0)
            documents.append(
                Document(
                    page_content=str(normalized.pop("content", "")),
                    metadata=normalized,
                )
            )
        return documents


@dataclass
class OpenSearch(VectorDataStore):
    """OpenSearch vector datastore.

    Example:
        >>> from langchain_oci.datastores import OpenSearch, create_datastore_tools
        >>>
        >>> store = OpenSearch(
        ...     endpoint="https://opensearch.example.com:9200",
        ...     index_name="my-docs",
        ...     username="admin",
        ...     password="...",
        ...     datastore_description="company documentation, policies",
        ... )
        >>>
        >>> tools = create_datastore_tools(
        ...     stores={"docs": store},
        ...     compartment_id="ocid1.compartment...",
        ... )
    """

    endpoint: str
    index_name: str
    username: Optional[str] = None
    password: Optional[str] = None
    use_ssl: bool = True
    verify_certs: bool = True
    vector_field: str = "embedding"
    search_fields: list[str] = field(default_factory=lambda: ["title", "content"])
    datastore_description: str = ""

    _client: Any = field(default=None, repr=False)
    _embedding_model: Any = field(default=None, repr=False)
    _vectorstore: Any = field(default=None, repr=False)
    _keyword_retriever: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "opensearch"

    @property
    def vectorstore(self) -> VectorStore:
        if self._vectorstore is None:
            raise RuntimeError("OpenSearch datastore is not connected.")
        return self._vectorstore

    @property
    def keyword_retriever(self) -> Optional[BaseRetriever]:
        return self._keyword_retriever

    def connect(self, embedding_model: Any) -> None:
        try:
            from opensearchpy import OpenSearch as OpenSearchClient
            from opensearchpy import RequestsHttpConnection
        except ImportError as e:
            raise ImportError(
                "opensearch-py required: pip install opensearch-py"
            ) from e

        http_auth = None
        if self.username and self.password:
            http_auth = (self.username, self.password)

        self._client = OpenSearchClient(
            hosts=[self.endpoint],
            http_auth=http_auth,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            connection_class=RequestsHttpConnection,
            timeout=30,
        )
        self._embedding_model = embedding_model
        self._client.info()
        self._vectorstore = _OpenSearchVectorStore(
            client=self._client,
            embedding_model=embedding_model,
            index_name=self.index_name,
            vector_field=self.vector_field,
        )
        self._keyword_retriever = _OpenSearchKeywordRetriever(
            client=self._client,
            index_name=self.index_name,
            search_fields=self.search_fields,
            vector_field=self.vector_field,
        )

    def search(self, query: str, embedding: list[float], top_k: int) -> list[dict]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {
                "id": (doc.metadata or {}).get("id"),
                "score": score,
                "title": (doc.metadata or {}).get("title", ""),
                "content": doc.page_content,
                "source": (doc.metadata or {}).get("source", ""),
                **{
                    key: value
                    for key, value in (doc.metadata or {}).items()
                    if key not in {"id", "title", "source"}
                },
            }
            for doc, score in docs_and_scores
        ]

    def keyword_search(self, query: str, top_k: int) -> list[dict]:
        docs = self.keyword_search_documents(query=query, top_k=top_k)
        return [
            {
                "id": (doc.metadata or {}).get("id"),
                "score": (doc.metadata or {}).get("score", 0),
                "title": (doc.metadata or {}).get("title", ""),
                "content": doc.page_content,
                "source": (doc.metadata or {}).get("source", ""),
                **{
                    key: value
                    for key, value in (doc.metadata or {}).items()
                    if key not in {"id", "score", "title", "source"}
                },
            }
            for doc in docs
        ]

    def get(self, document_id: str | int) -> Optional[dict]:
        try:
            response = self._client.get(index=self.index_name, id=str(document_id))
            if response.get("found"):
                return _normalize_source(
                    response.get("_source", {}),
                    vector_field=self.vector_field,
                    document_id=response["_id"],
                )
        except Exception:
            pass
        return None

    def insert(
        self, title: str, content: str, source: str, embedding: list[float]
    ) -> str:
        doc = {
            "title": title,
            "content": content,
            "source": source,
            self.vector_field: embedding,
        }
        response = self._client.index(index=self.index_name, body=doc, refresh=True)
        return response.get("_id", "")

    def bulk_insert(self, documents: list[dict], embeddings: list[list[float]]) -> int:
        bulk_body: list[dict] = []
        for doc, embedding in zip(documents, embeddings):
            bulk_body.append({"index": {"_index": self.index_name}})
            doc_entry: dict = {
                "title": doc.get("title", "Untitled"),
                "content": doc.get("content", ""),
                "source": doc.get("source", "bulk_insert"),
                self.vector_field: embedding,
            }
            bulk_body.append(doc_entry)
        response = self._client.bulk(body=bulk_body, refresh=True)
        if response.get("errors"):
            failed = sum(
                1 for item in response["items"] if "error" in item.get("index", {})
            )
            return len(documents) - failed
        return len(documents)

    def update(
        self,
        document_id: str | int,
        title: Optional[str],
        content: Optional[str],
        source: Optional[str],
        embedding: Optional[list[float]],
    ) -> bool:
        update_fields: dict[str, Any] = {}
        if title is not None:
            update_fields["title"] = title
        if content is not None:
            update_fields["content"] = content
        if source is not None:
            update_fields["source"] = source
        if embedding is not None:
            update_fields[self.vector_field] = embedding
        if not update_fields:
            return False
        try:
            self._client.update(
                index=self.index_name,
                id=str(document_id),
                body={"doc": update_fields},
                refresh=True,
            )
            return True
        except Exception:
            return False

    def delete(self, document_id: str | int) -> bool:
        try:
            response = self._client.delete(
                index=self.index_name, id=str(document_id), refresh=True
            )
            return response.get("result") == "deleted"
        except Exception:
            return False

    def stats(self) -> dict:
        stats = self._client.indices.stats(index=self.index_name)
        index_stats = stats.get("indices", {}).get(self.index_name, {})
        primaries = index_stats.get("primaries", {})
        return {
            "store": self.name,
            "index": self.index_name,
            "document_count": primaries.get("docs", {}).get("count", 0),
            "size_bytes": primaries.get("store", {}).get("size_in_bytes", 0),
        }
