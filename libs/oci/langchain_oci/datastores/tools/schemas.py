# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Input schemas for datastore tools."""

from typing import Optional

from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Input schema for search tools."""

    query: str = Field(
        description="The search query - a question or topic to find documents about"
    )


class GetDocumentInput(BaseModel):
    """Input schema for get_document tool."""

    document_id: str = Field(
        description="The document ID from a previous search result"
    )
    store: Optional[str] = Field(
        default=None,
        description="Datastore name (optional, uses default if not specified)",
    )


class StatsInput(BaseModel):
    """Input schema for stats tool."""

    store: Optional[str] = Field(
        default=None,
        description="Datastore name, or omit to get stats for all stores",
    )
