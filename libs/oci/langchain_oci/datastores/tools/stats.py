# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Stats tool for datastore information."""

from typing import ClassVar, Optional

from pydantic import BaseModel

from langchain_oci.datastores.tools.base import DatastoreTool, StoreStats
from langchain_oci.datastores.tools.schemas import StatsInput


class StatsTool(DatastoreTool):
    """Get statistics about available datastores."""

    name: str = "stats"
    args_schema: type[BaseModel] = StatsInput
    base_description: ClassVar[str] = (
        "START HERE - Get statistics about available datastores including "
        "document counts and descriptions. Call this first to understand "
        "what data is available for research."
    )

    def _run(self, store: Optional[str] = None) -> str:
        if store and store not in self.selector.stores:
            return f"Unknown datastore: '{store}'. Available: {self.store_list}"

        stores_to_check = [store] if store else self.selector.list_stores()

        stats_list = []
        for name in stores_to_check:
            try:
                s = self.selector.get_store(name)
                raw_stats = s.stats()
                stats_list.append(
                    StoreStats(
                        name=name,
                        description=s.datastore_description or "No description",
                        document_count=raw_stats.get("document_count", 0),
                        extra={
                            k: v for k, v in raw_stats.items() if k != "document_count"
                        },
                    )
                )
            except Exception as e:
                stats_list.append(
                    StoreStats(
                        name=name,
                        description=f"Error: {e}",
                        document_count=0,
                    )
                )

        return self.formatter.format_stats(stats_list)
