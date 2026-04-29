# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Semantic search tool for datastores."""

from typing import ClassVar

from pydantic import BaseModel

from langchain_oci.datastores.tools.base import DatastoreTool
from langchain_oci.datastores.tools.schemas import SearchInput


class SearchTool(DatastoreTool):
    """Semantic search - find documents by meaning."""

    name: str = "search"
    args_schema: type[BaseModel] = SearchInput
    base_description: ClassVar[str] = (
        "Semantic search - find documents by meaning and concept. "
        "Best for broad research queries like 'cancer treatment outcomes' "
        "or 'database performance issues'."
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
            "semantic search",
            query=query,
            store_name=store_name,
            top_k=self.top_k,
        )

        try:
            docs_and_scores = store.search_documents_with_scores(
                query=query,
                top_k=self.top_k,
            )
            documents = [doc for doc, _ in docs_and_scores]
            scores = [score for _, score in docs_and_scores]
            results = self._parse_documents(documents, scores)
            self._log_success(
                "semantic search",
                store_name=store_name,
                result_count=len(results),
            )
            return self.formatter.format_search_results(results, store_name, "semantic")
        except Exception as e:
            self._log_error(
                "semantic search",
                query=query,
                store_name=store_name,
                error=e,
            )
            return self.formatter.format_error("semantic search", e)
