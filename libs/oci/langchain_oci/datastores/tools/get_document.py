# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Get document tool for datastores."""

from typing import ClassVar, Optional

from pydantic import BaseModel

from langchain_oci.datastores.tools.base import DatastoreTool
from langchain_oci.datastores.tools.schemas import GetDocumentInput


class GetDocumentTool(DatastoreTool):
    """Retrieve full document content by ID."""

    name: str = "get_document"
    args_schema: type[BaseModel] = GetDocumentInput
    base_description: ClassVar[str] = (
        "Retrieve full document content by ID. Use after search or "
        "keyword_search to get complete text of relevant documents."
    )
    usage_hint: ClassVar[str] = (
        "Provide a Doc ID from search results. "
        "ALWAYS cite the Doc ID when using this content."
    )

    def _run(
        self,
        document_id: str,
        store: Optional[str] = None,
    ) -> str:
        store_name = store or self.selector.default_store

        if store_name not in self.selector.stores:
            return f"Unknown datastore: '{store_name}'. Available: {self.store_list}"

        try:
            raw_doc = self.selector.get_store(store_name).get(document_id)
            if not raw_doc:
                return (
                    f"Document '{document_id}' not found in '{store_name}'. "
                    "Check the Doc ID from your search results."
                )

            result = self._parse_results([raw_doc])[0]
            return self.formatter.format_document(result, store_name)
        except Exception as e:
            return self.formatter.format_error("document retrieval", e)
