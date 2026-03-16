# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Oracle Autonomous Database vector store datastore."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document

from langchain_oci.agents.datastores.vectorstores.base import VectorDataStore


@dataclass
class ADB(VectorDataStore):
    """Oracle Autonomous Database vector datastore.

    Uses ``langchain_oracledb.vectorstores.OracleVS`` for vector datastore
    operations.

    Example:
        >>> from langchain_oci.agents import ADB, create_datastore_tools
        >>>
        >>> store = ADB(
        ...     dsn="mydb_low",
        ...     user="ADMIN",
        ...     password="...",
        ...     wallet_location="~/.oracle-wallet",
        ...     datastore_description="sales data, revenue, customers",
        ... )
        >>>
        >>> tools = create_datastore_tools(
        ...     stores={"sales": store},
        ...     compartment_id="ocid1.compartment...",
        ... )
    """

    dsn: str
    user: str
    password: str
    wallet_location: Optional[str] = None
    wallet_password: Optional[str] = None
    table_name: str = "VECTOR_DOCUMENTS"
    datastore_description: str = ""
    chunk_on_write: bool = True
    chunking_params: Optional[dict[str, Any]] = None

    _connection: Any = field(default=None, repr=False)
    _embedding_model: Any = field(default=None, repr=False)
    _oraclevs: Any = field(default=None, repr=False)
    _text_retriever: Any = field(default=None, repr=False)
    _write_text_splitter: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "adb"

    @property
    def vectorstore(self) -> Any:
        if self._oraclevs is None:
            raise RuntimeError("ADB datastore is not connected.")
        return self._oraclevs

    @property
    def keyword_retriever(self) -> Any:
        return self._text_retriever

    def connect(self, embedding_model: Any) -> None:
        try:
            import oracledb
        except ImportError as e:
            raise ImportError("oracledb required: pip install oracledb") from e

        config_dir = None
        if self.wallet_location:
            config_dir = os.path.expanduser(self.wallet_location)

        self.logger.info(
            "Connecting ADB datastore table=%s dsn=%s wallet=%s user=%s",
            self.table_name,
            self.dsn,
            config_dir or "<none>",
            self.user,
        )

        self._connection = oracledb.connect(
            user=self.user,
            password=self.password,
            dsn=self.dsn,
            config_dir=config_dir,
            wallet_location=config_dir,
            wallet_password=self.wallet_password or self.password,
        )
        self._embedding_model = embedding_model
        self._initialize_oraclevs_backend()
        self.logger.info(
            "ADB datastore connected table=%s dsn=%s",
            self.table_name,
            self.dsn,
        )

    def _initialize_oraclevs_backend(self) -> None:
        try:
            from langchain_community.vectorstores.utils import DistanceStrategy
            from langchain_oracledb.document_loaders.oracleai import (
                OracleTextSplitter,
            )
            from langchain_oracledb.retrievers import (
                OracleTextSearchRetriever,
            )
            from langchain_oracledb.vectorstores.oraclevs import (
                OracleVS,
            )
        except ImportError as e:
            raise ImportError(
                "langchain-oracledb required for ADB datastore integration. "
                "Install with: pip install langchain-oracledb"
            ) from e

        self.logger.debug(
            (
                "Initializing ADB OracleVS backend table=%s "
                "chunk_on_write=%s chunking_params=%s"
            ),
            self.table_name,
            self.chunk_on_write,
            self.chunking_params,
        )

        distance_strategy = getattr(
            DistanceStrategy,
            "COSINE_DISTANCE",
            getattr(DistanceStrategy, "COSINE"),
        )

        self._oraclevs = OracleVS(
            client=self._connection,
            embedding_function=self._embedding_model,
            table_name=self.table_name,
            distance_strategy=distance_strategy,
            mutate_on_duplicate=True,
        )
        self._text_retriever = OracleTextSearchRetriever(vector_store=self._oraclevs)
        if self.chunk_on_write:
            params = self.chunking_params or {
                "split": "sentence",
                "max": 20,
                "normalize": "all",
            }
            self._write_text_splitter = OracleTextSplitter(
                conn=self._connection,
                params=params,
            )
        self.logger.debug("ADB OracleVS backend initialized table=%s", self.table_name)

    def _ingest_document(self, document: Document, doc_id: str) -> None:
        if self._write_text_splitter is not None:
            self._oraclevs.add_documents(
                [document],
                text_splitter=self._write_text_splitter,
                ids=[doc_id],
            )
            return

        self._oraclevs.add_texts(
            texts=[document.page_content],
            metadatas=[document.metadata],
            ids=[doc_id],
        )

    def _read_text(self, value: Any) -> str:
        if hasattr(value, "read"):
            return value.read()
        return str(value) if value else ""

    def search(self, query: str, embedding: list[float], top_k: int) -> list[dict]:
        docs_and_scores = self.search_documents_with_scores(query=query, top_k=top_k)
        self.logger.debug(
            "ADB search requested table=%s top_k=%s query=%r embedding_dims=%s",
            self.table_name,
            top_k,
            query,
            len(embedding),
        )
        docs_and_scores = self.search_documents_with_scores(query=query, top_k=top_k)
        return [
            {
                "id": (doc.metadata or {}).get("id"),
                "title": (doc.metadata or {}).get("title", ""),
                "content": (doc.page_content or "")[:1000],
                "source": (doc.metadata or {}).get("source", ""),
                "score": 1 - score,
            }
            for doc, score in docs_and_scores
        ]

    def keyword_search(self, query: str, top_k: int) -> list[dict]:
        self.logger.debug(
            "ADB keyword search requested table=%s top_k=%s query=%r",
            self.table_name,
            top_k,
            query,
        )
        docs = self.keyword_search_documents(query=query, top_k=top_k)
        return [
            {
                "id": (doc.metadata or {}).get("id"),
                "title": (doc.metadata or {}).get("title", ""),
                "content": (doc.page_content or "")[:1000],
                "source": (doc.metadata or {}).get("source", ""),
            }
            for doc in docs
        ]

    def get(self, document_id: str | int) -> Optional[dict]:
        cursor = self._connection.cursor()
        cursor.execute(
            f"""
            SELECT text, metadata
            FROM {self.table_name}
            WHERE JSON_VALUE(metadata, '$.id') = :doc_id
            """,
            {"doc_id": str(document_id)},
        )
        rows = cursor.fetchall()
        cursor.close()
        if not rows:
            return None

        parsed = []
        for text_value, metadata in rows:
            if not isinstance(metadata, dict):
                metadata = {}
            parsed.append((self._read_text(text_value), metadata))

        parsed.sort(key=lambda row: row[1].get("chunk_index", 0))
        content = "\n".join([p[0] for p in parsed])
        metadata = parsed[0][1]
        return {
            "id": metadata.get("id", str(document_id)),
            "title": metadata.get("title", ""),
            "content": content,
            "source": metadata.get("source", ""),
            "created_at": None,
        }

    def insert(
        self, title: str, content: str, source: str, embedding: list[float]
    ) -> str:
        doc_id = str(uuid.uuid4())
        self._ingest_document(
            Document(
                page_content=content,
                metadata={"id": doc_id, "title": title, "source": source},
            ),
            doc_id,
        )
        return doc_id

    def bulk_insert(self, documents: list[dict], embeddings: list[list[float]]) -> int:
        for doc in documents:
            doc_id = str(doc.get("id") or uuid.uuid4())
            self._ingest_document(
                Document(
                    page_content=str(doc.get("content", "")),
                    metadata={
                        "id": doc_id,
                        "title": str(doc.get("title", "Untitled")),
                        "source": str(doc.get("source", "bulk_insert")),
                    },
                ),
                doc_id,
            )
        return len(documents)

    def update(
        self,
        document_id: str | int,
        title: Optional[str],
        content: Optional[str],
        source: Optional[str],
        embedding: Optional[list[float]],
    ) -> bool:
        current = self.get(document_id)
        if not current:
            return False
        new_title = title if title is not None else str(current.get("title", ""))
        new_content = (
            content if content is not None else str(current.get("content", ""))
        )
        new_source = source if source is not None else str(current.get("source", ""))
        self.delete(document_id)
        self._ingest_document(
            Document(
                page_content=new_content,
                metadata={
                    "id": str(document_id),
                    "title": new_title,
                    "source": new_source,
                },
            ),
            str(document_id),
        )
        return True

    def delete(self, document_id: str | int) -> bool:
        cursor = self._connection.cursor()
        cursor.execute(
            (
                f"DELETE FROM {self.table_name} "
                "WHERE JSON_VALUE(metadata, '$.id') = :doc_id"
            ),
            {"doc_id": str(document_id)},
        )
        self._connection.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted

    def stats(self) -> dict:
        cursor = self._connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]

        # Try to get sources - handle both metadata (JSON) and source (VARCHAR) columns
        try:
            cursor.execute(
                f"""
                SELECT JSON_VALUE(metadata, '$.source') as source, COUNT(*) as cnt
                FROM {self.table_name}
                GROUP BY JSON_VALUE(metadata, '$.source')
                ORDER BY cnt DESC
                FETCH FIRST 10 ROWS ONLY
            """
            )
            sources = {
                (row[0] if row[0] is not None else "unknown"): row[1]
                for row in cursor.fetchall()
            }
        except Exception:
            # Fallback to SOURCE column if metadata doesn't exist
            try:
                cursor.execute(
                    f"""
                    SELECT source, COUNT(*) as cnt
                    FROM {self.table_name}
                    GROUP BY source
                    ORDER BY cnt DESC
                    FETCH FIRST 10 ROWS ONLY
                """
                )
                sources = {
                    (row[0] if row[0] is not None else "unknown"): row[1]
                    for row in cursor.fetchall()
                }
            except Exception:
                sources = {}

        cursor.close()
        return {
            "store": self.name,
            "table": self.table_name,
            "document_count": count,
            "sources": sources,
        }
