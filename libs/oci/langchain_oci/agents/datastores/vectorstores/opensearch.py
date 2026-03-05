# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OpenSearch vector store datastore."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_oci.agents.datastores.vectorstores.base import VectorDataStore


@dataclass
class OpenSearch(VectorDataStore):
    """OpenSearch vector datastore.

    Example:
        >>> from langchain_oci.agents import OpenSearch, create_datastore_tools
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

    @property
    def name(self) -> str:
        return "opensearch"

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

    def search(self, query: str, embedding: list[float], top_k: int) -> list[dict]:
        search_body = {
            "size": top_k,
            "query": {"knn": {self.vector_field: {"vector": embedding, "k": top_k}}},
            "_source": {"excludes": [self.vector_field]},
        }
        response = self._client.search(index=self.index_name, body=search_body)
        hits = response.get("hits", {}).get("hits", [])
        return [
            {"id": hit["_id"], "score": hit.get("_score", 0), **hit.get("_source", {})}
            for hit in hits
        ]

    def keyword_search(self, query: str, top_k: int) -> list[dict]:
        search_body = {
            "size": top_k,
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
        response = self._client.search(index=self.index_name, body=search_body)
        hits = response.get("hits", {}).get("hits", [])
        return [
            {"id": hit["_id"], "score": hit.get("_score", 0), **hit.get("_source", {})}
            for hit in hits
        ]

    def get(self, document_id: str | int) -> Optional[dict]:
        try:
            response = self._client.get(index=self.index_name, id=str(document_id))
            if response.get("found"):
                source = response.get("_source", {})
                source.pop(self.vector_field, None)
                return {"id": response["_id"], **source}
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
