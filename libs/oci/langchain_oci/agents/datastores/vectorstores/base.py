# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Abstract base class for datastore adapters built on LangChain standards."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


class VectorDataStore(ABC):
    """Abstract base class for datastore adapters.

    Implementations should expose standard LangChain primitives internally:
    a ``VectorStore`` for semantic search and, optionally, a ``BaseRetriever``
    for keyword search. The higher-level OCI agent tooling works against those
    contracts instead of calling backend-specific search methods directly.

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
    def logger(self) -> logging.Logger:
        """Logger scoped to the concrete datastore class."""
        return logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

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

    @property
    @abstractmethod
    def vectorstore(self) -> VectorStore:
        """LangChain vector store used for semantic retrieval."""

    @property
    def keyword_retriever(self) -> Optional[BaseRetriever]:
        """Optional LangChain retriever used for keyword/text search."""
        return None

    def search_documents(self, query: str, top_k: int) -> list[Document]:
        """Perform semantic search through the configured vector store."""
        return self.vectorstore.similarity_search(query, k=top_k)

    def search_documents_with_scores(
        self, query: str, top_k: int
    ) -> list[tuple[Document, float]]:
        """Perform semantic search with scores when the vector store supports it."""
        search_with_score = getattr(
            self.vectorstore, "similarity_search_with_score", None
        )
        if search_with_score is None:
            return [(doc, 0.0) for doc in self.search_documents(query, top_k)]
        return search_with_score(query, k=top_k)

    def keyword_search_documents(self, query: str, top_k: int) -> list[Document]:
        """Perform keyword search through the configured retriever."""
        retriever = self.keyword_retriever
        if retriever is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not provide a keyword retriever."
            )

        original_k = getattr(retriever, "k", None)
        if original_k is not None:
            setattr(retriever, "k", top_k)

        try:
            return retriever.invoke(query)
        finally:
            if original_k is not None:
                setattr(retriever, "k", original_k)

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
