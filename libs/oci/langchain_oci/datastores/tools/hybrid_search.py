# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Hybrid search tool for datastores."""

from typing import ClassVar

from pydantic import BaseModel

from langchain_oci.datastores.tools.base import DatastoreTool
from langchain_oci.datastores.tools.schemas import SearchInput


class HybridSearchTool(DatastoreTool):
    """Hybrid search combining semantic and keyword results via RRF."""

    name: str = "search"
    args_schema: type[BaseModel] = SearchInput
    description: str = (
        "Search for documents combining meaning and exact terms. "
        "Works for conceptual queries ('cancer treatment outcomes') "
        "and technical terms ('NullPointerException', 'connection_exhaustion')."
    )
    usage_hint: ClassVar[str] = (
        "Returns Doc IDs, titles, and content snippets. "
        "ALWAYS cite Doc IDs in your output."
    )

    top_k: int = 5

    def _run(self, query: str) -> str:
        store_name = self.selector.route(query)
        store = self.selector.get_store(store_name)
        self._log_start(
            "hybrid search",
            query=query,
            store_name=store_name,
            top_k=self.top_k,
        )

        try:
            results_with_scores = store.hybrid_search_documents(
                query=query, top_k=self.top_k
            )
            documents = [doc for doc, _ in results_with_scores]
            scores = [score for _, score in results_with_scores]
            results = self._parse_documents(documents, scores)
            self._log_success(
                "hybrid search",
                store_name=store_name,
                result_count=len(results),
            )
            return self.formatter.format_search_results(results, store_name, "hybrid")
        except Exception as e:
            self._log_error(
                "hybrid search",
                query=query,
                store_name=store_name,
                error=e,
            )
            return self.formatter.format_error("hybrid search", e)
