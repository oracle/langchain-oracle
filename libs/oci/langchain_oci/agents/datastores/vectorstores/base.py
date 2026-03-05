# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Abstract base class for vector store datastores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class VectorDataStore(ABC):
    """Abstract base class for vector datastores.

    Implement this class to create custom datastore backends.
    Built-in implementations: OpenSearch, ADB (Oracle Autonomous Database).

    Example:
        >>> from langchain_oci.agents import VectorDataStore
        >>>
        >>> class MyDataStore(VectorDataStore):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_store"
        ...
        ...     # ... implement other methods
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Store identifier."""
        ...

    @property
    def datastore_description(self) -> str:
        """Description of documents in this store.

        Used for auto-routing in multi-datastore configurations. The agent
        uses semantic similarity between the query and datastore descriptions
        to select the most relevant datastore(s) to search.

        Examples:
            "legal contracts, clauses, compliance documentation"
            "incident reports, runbooks, system diagnostics"
            "medical research papers, clinical trials, drug information"
        """
        return ""

    @abstractmethod
    def connect(self, embedding_model: Any) -> None:
        """Initialize connection to the store."""
        ...

    @abstractmethod
    def search(self, query: str, embedding: list[float], top_k: int) -> list[dict]:
        """Perform vector similarity search."""
        ...

    @abstractmethod
    def keyword_search(self, query: str, top_k: int) -> list[dict]:
        """Perform keyword/text search."""
        ...

    @abstractmethod
    def get(self, document_id: str | int) -> Optional[dict]:
        """Get a document by ID."""
        ...

    @abstractmethod
    def insert(
        self, title: str, content: str, source: str, embedding: list[float]
    ) -> str:
        """Insert a document. Returns document ID."""
        ...

    @abstractmethod
    def bulk_insert(self, documents: list[dict], embeddings: list[list[float]]) -> int:
        """Bulk insert documents. Returns count inserted."""
        ...

    @abstractmethod
    def update(
        self,
        document_id: str | int,
        title: Optional[str],
        content: Optional[str],
        source: Optional[str],
        embedding: Optional[list[float]],
    ) -> bool:
        """Update a document. Returns success."""
        ...

    @abstractmethod
    def delete(self, document_id: str | int) -> bool:
        """Delete a document. Returns success."""
        ...

    @abstractmethod
    def stats(self) -> dict:
        """Get store statistics."""
        ...
