# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Semantic search tool for datastores."""

from typing import ClassVar

from pydantic import BaseModel

from langchain_oci.agents.datastores.tools.base import DatastoreTool
from langchain_oci.agents.datastores.tools.schemas import SearchInput


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

        try:
            embedding = self.selector.embedding_model.embed_query(query)
            raw_results = store.search(query, embedding, self.top_k)
            results = self._parse_results(raw_results)
            return self.formatter.format_search_results(results, store_name, "semantic")
        except Exception as e:
            return self.formatter.format_error("semantic search", e)
