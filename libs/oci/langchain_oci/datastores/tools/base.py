# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Base classes and types for datastore tools."""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field, PrivateAttr


@dataclass
class SearchResult:
    """A single search result from a datastore."""

    id: str
    title: str
    content: str
    score: float = 0.0
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoreStats:
    """Statistics about a datastore."""

    name: str
    description: str
    document_count: int
    extra: dict[str, Any] = field(default_factory=dict)


class ResultFormatter:
    """Formats tool results for LLM consumption with citation guidance."""

    CITATION_REMINDER = "IMPORTANT: Cite Doc IDs when using this information."
    DOCUMENT_METADATA_KEYS = {"category", "type", "runbook_category", "source_path"}

    @staticmethod
    def format_search_results(
        results: list[SearchResult],
        store_name: str,
        search_type: str = "semantic",
    ) -> str:
        """Format search results with citation instructions."""
        if not results:
            return f"No results found in '{store_name}' datastore."

        lines = [
            f"Found {len(results)} {search_type} matches from '{store_name}':",
            ResultFormatter.CITATION_REMINDER,
            "",
        ]

        for result in results:
            lines.append(
                f"[Doc ID: {result.id}]"
                + (f" (relevance: {result.score:.3f})" if result.score else "")
            )
            lines.append(f"  Title: {result.title}")
            content_preview = (
                result.content[:500] + "..."
                if len(result.content) > 500
                else result.content
            )
            lines.append(f"  Content: {content_preview}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_document(
        result: SearchResult,
        store_name: str,
    ) -> str:
        """Format a full document with citation markers."""
        lines = [
            f"=== FULL DOCUMENT [Doc ID: {result.id}] ===",
            f"Source Datastore: '{store_name}'",
            f"Title: {result.title}",
        ]

        if result.source:
            lines.append(f"Original Source: {result.source}")

        for key, value in result.metadata.items():
            if key in ResultFormatter.DOCUMENT_METADATA_KEYS and value not in (
                "",
                None,
            ):
                lines.append(f"{key.replace('_', ' ').title()}: {value}")

        lines.extend(
            [
                "",
                "Content:",
                result.content,
                "",
                f"=== END [Doc ID: {result.id}] - CITE THIS ID ===",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def format_stats(stores: list[StoreStats]) -> str:
        """Format datastore statistics."""
        lines = [
            "=== DATASTORE STATISTICS ===",
            "Available datastores for search, keyword_search, get_document.",
            "",
        ]

        for store in stores:
            lines.append(f"Datastore: '{store.name}'")
            lines.append(f"  Description: {store.description}")
            lines.append(f"  Documents: {store.document_count:,}")
            for key, value in store.extra.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_error(operation: str, error: Exception) -> str:
        """Format an error message."""
        return f"Error during {operation}: {type(error).__name__}: {error}"


class DatastoreTool(BaseTool, ABC):
    """Base class for datastore tools with common configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Tool metadata - override in subclasses
    name: str = "datastore_tool"
    base_description: ClassVar[str] = "A datastore tool."
    usage_hint: ClassVar[str] = ""

    # Injected dependencies
    selector: Any = Field(default=None, exclude=True)
    formatter: Any = Field(default=None, exclude=True)
    store_list: str = Field(default="", exclude=True)
    _logger: logging.Logger = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.formatter is None:
            object.__setattr__(self, "formatter", ResultFormatter())
        object.__setattr__(
            self,
            "_logger",
            logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}"),
        )

    @property
    def full_description(self) -> str:
        """Build full description with usage hints and store list."""
        parts = [self.base_description]
        if self.usage_hint:
            parts.append(self.usage_hint)
        if self.store_list:
            parts.append(f"Available stores: {self.store_list}")
        return " ".join(parts)

    def _parse_results(self, raw_results: list[dict]) -> list[SearchResult]:
        """Convert raw store results to SearchResult objects."""
        return [
            SearchResult(
                id=str(r.get("id", "unknown")),
                title=r.get("title", "Untitled"),
                content=str(r.get("content", "")),
                score=float(r.get("score", 0)),
                source=r.get("source"),
                metadata={
                    k: v
                    for k, v in r.items()
                    if k not in ("id", "title", "content", "score", "source")
                },
            )
            for r in raw_results
        ]

    def _parse_documents(
        self,
        documents: list[Document],
        scores: Optional[list[float]] = None,
    ) -> list[SearchResult]:
        """Convert LangChain documents into SearchResult objects."""
        results: list[SearchResult] = []
        for index, doc in enumerate(documents):
            metadata = dict(doc.metadata or {})
            document_id = getattr(doc, "id", None) or metadata.get("id", "unknown")
            results.append(
                SearchResult(
                    id=str(document_id),
                    title=str(metadata.get("title", "Untitled")),
                    content=str(doc.page_content or ""),
                    score=float(scores[index]) if scores is not None else 0.0,
                    source=metadata.get("source"),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ("id", "title", "source")
                    },
                )
            )
        return results

    def _log_start(
        self,
        operation: str,
        *,
        query: str,
        store_name: str,
        top_k: int,
    ) -> None:
        """Emit a consistent start log for datastore tool operations."""
        self._logger.info(
            "Starting datastore %s store=%s top_k=%s query=%r",
            operation,
            store_name,
            top_k,
            query,
        )

    def _log_success(
        self,
        operation: str,
        *,
        store_name: str,
        result_count: int,
    ) -> None:
        """Emit a consistent success log for datastore tool operations."""
        self._logger.info(
            "Datastore %s succeeded store=%s results=%s",
            operation,
            store_name,
            result_count,
        )

    def _log_error(
        self,
        operation: str,
        *,
        query: str,
        store_name: str,
        error: Exception,
    ) -> None:
        """Emit a consistent error log for datastore tool operations."""
        self._logger.exception(
            "Datastore %s failed store=%s query=%r error=%s: %s",
            operation,
            store_name,
            query,
            type(error).__name__,
            error,
        )
