# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Keyword search tool for datastores."""

from typing import ClassVar

from pydantic import BaseModel

from langchain_oci.datastores.tools.base import DatastoreTool
from langchain_oci.datastores.tools.schemas import SearchInput


class KeywordSearchTool(DatastoreTool):
    """Keyword search - find documents containing specific terms."""

    name: str = "keyword_search"
    args_schema: type[BaseModel] = SearchInput
    base_description: ClassVar[str] = (
        "Keyword search - find documents containing exact terms. "
        "Best for specific technical terms like 'connection_exhaustion', "
        "'metastasis', or 'NullPointerException'."
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
            "keyword search",
            query=query,
            store_name=store_name,
            top_k=self.top_k,
        )

        try:
            documents = store.keyword_search_documents(query=query, top_k=self.top_k)
            results = self._parse_documents(documents)
            self._log_success(
                "keyword search",
                store_name=store_name,
                result_count=len(results),
            )
            return self.formatter.format_search_results(results, store_name, "keyword")
        except Exception as e:
            self._log_error(
                "keyword search",
                query=query,
                store_name=store_name,
                error=e,
            )
            return self.formatter.format_error("keyword search", e)
