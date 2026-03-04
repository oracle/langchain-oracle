# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Store selector for routing queries to datastores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from langchain_oci.agents.datastores.vectorstores import VectorDataStore


class StoreSelector:
    """Routes queries to the best datastore using semantic similarity.

    When multiple datastores are available, the selector compares the query
    embedding against pre-computed embeddings of each store's description
    to find the most relevant store.
    """

    def __init__(
        self,
        stores: dict[str, VectorDataStore],
        embedding_model: Any,
        default_store: str,
    ) -> None:
        self.stores = stores
        self.embedding_model = embedding_model
        self.default_store = default_store
        self._description_embeddings: dict[str, np.ndarray] = {}
        self._precompute_embeddings()

    def _precompute_embeddings(self) -> None:
        """Pre-compute embeddings for store descriptions."""
        for name, store in self.stores.items():
            description_text = store.datastore_description or name
            embedding = self.embedding_model.embed_query(description_text)
            self._description_embeddings[name] = np.array(embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def route(self, query: str) -> str:
        """Route a query to the most relevant store."""
        if len(self.stores) == 1:
            return next(iter(self.stores.keys()))

        query_embedding = np.array(self.embedding_model.embed_query(query))
        best_store = self.default_store
        best_score = -1.0

        for name, description_embedding in self._description_embeddings.items():
            score = self._cosine_similarity(query_embedding, description_embedding)
            if score > best_score:
                best_score = score
                best_store = name

        return best_store

    def get_store(self, name: str) -> VectorDataStore:
        """Get a store by name."""
        return self.stores[name]

    def list_stores(self) -> list[str]:
        """List all available store names."""
        return list(self.stores.keys())
