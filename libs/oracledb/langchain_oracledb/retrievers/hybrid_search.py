# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Hybrid search utilities for Oracle Database.

This module provides:
- OracleVectorizerPreference: create/drop DBMS_VECTOR_CHAIN vectorizer preferences used
    by hybrid vector indexes.
- create_hybrid_index/acreate_hybrid_index: build a HYBRID VECTOR INDEX over a text
    column using a vectorizer preference.
- OracleVSHybridSearchRetriever: a retriever that calls DBMS_HYBRID_VECTOR.SEARCH
    to perform keyword, semantic, or hybrid retrieval against an OracleVS table.

References:
- Hybrid search overview and guidance: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/understand-hybrid-search.html#GUID-310D2298-90F4-4AFE-AF03-F3B81E55F84C__GUID-03905981-A6E9-4D2C-A0DC-0807A95AA3F3
- Create vectorizer preference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create_preference.html
- Create hybrid vector index: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html
- Search API (DBMS_HYBRID_VECTOR.SEARCH): https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/search.html
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

import oracledb
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import field_validator

from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.vectorstores.oraclevs import INTERNAL_ID_KEY, OracleVS
from langchain_oracledb.vectorstores.utils import (
    _aget_connection,
    _ahandle_exceptions,
    _aindex_exists,
    _get_connection,
    _get_index_name,
    _handle_exceptions,
    _index_exists,
    _quote_indentifier,
    adrop_index,  # noqa: F401
    drop_index,  # noqa: F401
    output_type_string_handler,
)

if TYPE_CHECKING:
    from oracledb import (
        AsyncConnection,
        AsyncConnectionPool,
        Connection,
        ConnectionPool,
    )


logger = logging.getLogger(__name__)


def _validate_parameters(
    embeddings: OracleEmbeddings,
    params: dict[str, Any],
):
    """Validate that provided preference parameters are consistent with the
    OracleEmbeddings bound to the vector store.

    Supports two mutually exclusive ways to specify the model configuration:
    - model: database-resident embedding model name
    - embedder_spec: JSON spec for external embedding providers

    Raises:
        ValueError: if parameters do not match the embeddings configuration.

    Returns:
        bool: True if a model configuration was explicitly provided; False otherwise.
    """
    if "model" in params:
        model_name = params.get("model")

        if (
            embeddings.params.get("provider") != "database"
            or embeddings.params.get("model") != model_name
        ):
            raise ValueError(
                f"Mismatch between embedding and provided params: "
                f"OracleEmbeddings expects provider='database' and "
                f"model='{embeddings.params.get('model')}', but received "
                f"model='{model_name}'."
            )

        return True

    if "embedder_spec" in params:
        embedder_spec = params.get("embedder_spec")

        if not (
            json.dumps(embeddings.params, sort_keys=True)
            == json.dumps(embedder_spec, sort_keys=True)
        ):
            raise ValueError(
                "Mismatch between embedding and provided params: "
                "embedder_spec must exactly match OracleEmbeddings.params "
                "(after JSON normalization)."
            )

        return True

    return False


class OracleVectorizerPreference:
    """Manage DBMS_VECTOR_CHAIN vectorizer preferences for hybrid search.

    A vectorizer preference encapsulates embedding configuration used by
    hybrid vector indexes. This class derives the correct preference parameters
    from the OracleEmbeddings attached to the target OracleVS instance.

    Reference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create_preference.html

    Example:
        Basic synchronous usage

        from langchain_oracledb.embeddings import OracleEmbeddings
        from langchain_oracledb.vectorstores.oraclevs import OracleVS
        from langchain_oracledb.retrievers.hybrid_search import (
            OracleVectorizerPreference,
            create_hybrid_index,
            OracleVSHybridSearchRetriever,
        )
        import oracledb

        # Connect and prepare embeddings and vector store
        client = oracledb.connect(user="scott", password="tiger", dsn="dbhost/service")
        embeddings = OracleEmbeddings(
            params={
                "provider": "database",
                "model": "DB_MODEL"
            }
        )
        vs = OracleVS(client=client, table_name="DOCS", embedding_function=embeddings)

        # Create a vectorizer preference and hybrid index
        pref = OracleVectorizerPreference.create_preference(
            vs,
            preference_name="PREF_DOCS"
        )
        create_hybrid_index(
            client,
            idx_name="IDX_DOCS_HYB",
            vectorizer_preference=pref,
            params={
                "parallel": 4,
            },
        )

        # Build a retriever and search
        retriever = OracleVSHybridSearchRetriever(vector_store=vs, idx_name="IDX_DOCS_HYB", k=5)
        docs = retriever.invoke("refund policy for premium plan")

        # Cleanup when needed
        pref.drop_preference()

    Example:
        Async usage

        import asyncio

        async def main():
            async_client = dict(user="scott", password="tiger", dsn="dbhost/service")  # accepted by _aget_connection
            embeddings = OracleEmbeddings(params={"provider": "database", "model": "YOUR_DB_MODEL"})
            vs = OracleVS(client=async_client, table_name="DOCS", embedding_function=embeddings)

            pref = await OracleVectorizerPreference.acreate_preference(vs, preference_name="PREF_DOCS_A")
            await acreate_hybrid_index(async_client, "IDX_DOCS_HYB_A", pref)

            retriever = OracleVSHybridSearchRetriever(vector_store=vs, idx_name="IDX_DOCS_HYB_A", k=3)
            results = await retriever.ainvoke("latest SLA")

            await pref.adrop_preference()

        asyncio.run(main())
    """  # noqa E501

    params: Optional[dict[str, Any]]
    vs: OracleVS
    preference_name: str

    PREFERENCE_STR = """
    begin
    DBMS_VECTOR_CHAIN.CREATE_PREFERENCE(
        :1,
        dbms_vector_chain.vectorizer,
        json(:2));
    end;"""

    def _get_preference_parameters(self) -> dict:
        preference_params = self.params.copy() if self.params else {}
        embeddings = self.vs.embedding_function
        if not isinstance(embeddings, OracleEmbeddings):
            raise ValueError(
                "Only OracleEmbeddings can be used to create a vectorizer preference; "
                f"received type {type(embeddings).__name__}."
            )

        has_model_config = _validate_parameters(embeddings, preference_params)

        if not has_model_config:
            if embeddings.params.get("provider") == "database":
                preference_params["model"] = embeddings.params.get("model")
            else:
                preference_params["embedder_spec"] = embeddings.params

        return preference_params

    @classmethod
    def create_preference(
        cls,
        vector_store: OracleVS,
        preference_name: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> OracleVectorizerPreference:
        """Create a DBMS_VECTOR_CHAIN vectorizer preference for hybrid indexing.

        Parameters are inferred from the OracleEmbeddings attached to the provided
        OracleVS unless explicitly overridden via params.

        Args:
            vector_store: OracleVS whose OracleEmbeddings define the vectorizer config.
            preference_name: Optional explicit preference name.
                A random name is generated if omitted.
            params: Optional dict with additional options.

        Returns:
            OracleVectorizerPreference: handle containing the created preference name.

        Reference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create_preference.html
        """
        self = cls.__new__(cls)
        self.params = params
        self.vs = vector_store
        self.preference_name = (
            preference_name or "pref" + str(uuid.uuid4()).replace("-", "")[0:15]
        )

        preference_params = self._get_preference_parameters()

        with _get_connection(vector_store.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    OracleVectorizerPreference.PREFERENCE_STR,
                    [self.preference_name, json.dumps(preference_params)],
                )
                logger.info(f"Preference {self.preference_name} created.")

        return self

    @classmethod
    async def acreate_preference(
        cls,
        vector_store: OracleVS,
        preference_name: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> OracleVectorizerPreference:
        """Async variant of create_preference.

        Creates a DBMS_VECTOR_CHAIN vectorizer preference using the async
        connection/cursor APIs.

        Args:
            vector_store: OracleVS whose OracleEmbeddings define the vectorizer config.
            preference_name: Optional explicit preference name.
            params: Optional dict with additional options.

        Returns:
            OracleVectorizerPreference

        Reference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create_preference.html
        """
        self = cls.__new__(cls)
        self.params = params
        self.vs = vector_store
        self.preference_name = preference_name or _get_index_name("pref")[0:30]

        preference_params = self._get_preference_parameters()

        async with _aget_connection(vector_store.client) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    OracleVectorizerPreference.PREFERENCE_STR,
                    [self.preference_name, json.dumps(preference_params)],
                )
                logger.info(f"Preference {self.preference_name} created.")

        return self

    def drop_preference(self):
        """Drop this vectorizer preference using DBMS_VECTOR_CHAIN.DROP_PREFERENCE."""
        with _get_connection(self.vs.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "begin DBMS_VECTOR_CHAIN.DROP_PREFERENCE (:preference_name); end;",
                    preference_name=self.preference_name,
                )

    async def adrop_preference(self):
        """Async variant of drop_preference."""
        async with _aget_connection(self.vs.client) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "begin DBMS_VECTOR_CHAIN.DROP_PREFERENCE (:preference_name); end;",
                    preference_name=self.preference_name,
                )


def drop_preference(connection, preference_name):
    """Drop a DBMS_VECTOR_CHAIN preference by name.

    Args:
        connection: oracledb connection.
        preference_name: Preference to drop.
    """
    with _get_connection(connection) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "begin DBMS_VECTOR_CHAIN.DROP_PREFERENCE (:preference_name); end;",
                preference_name=preference_name,
            )


async def adrop_preference(connection, preference_name):
    """Async variant of drop_preference."""
    async with _aget_connection(connection) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "begin DBMS_VECTOR_CHAIN.DROP_PREFERENCE (:preference_name); end;",
                preference_name=preference_name,
            )


def _get_hybrid_index_ddl(
    vectorizer_preference: OracleVectorizerPreference,
    idx_name: str,
    params: dict[str, Any],
):
    """Build the CREATE HYBRID VECTOR INDEX DDL statement.

    The vectorizer is set via the provided OracleVectorizerPreference. Additional
    index parameters can be supplied in params["parameters"]. Some fields are
    reserved and must not be present in params["parameters"]:
    - model, embedder_spec, vector_idxtype, vectorizer

    Optional clauses:
    - filter_by: list[str] -> FILTER BY clause
    - order_by: list[str] with order_by_asc -> ORDER BY ... ASC|DESC
    - parallel: int -> PARALLEL N

    Args:
        vectorizer_preference: Preference that defines the embedding configuration.
        idx_name: Name of the hybrid index (quoted as needed).
        params: dict of options including
            "parameters", "filter_by", "order_by", "order_by_asc", "parallel".

    Returns:
        str: DDL statement text.

    Reference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html
    """
    index_parameters = params.get("parameters", {}).copy()
    if any(
        key.lower() in ["model", "embedder_spec", "vector_idxtype", "vectorizer"]
        for key in index_parameters
    ):
        raise ValueError(
            "Vectorization parameters must be given with OracleVectorizerPreference: "
            "do not include any of {model, embedder_spec, vector_idxtype, vectorizer} "
            "under params['parameters']."
        )

    params_str, filter_by_str, order_by_str, parallel_str = "", "", "", ""

    params_str = f"vectorizer {vectorizer_preference.preference_name} "
    for k, v in index_parameters.items():
        params_str += f"{k} {v} "

    filter_by = params.get("filter_by", None)
    if filter_by:
        filter_by_str = "FILTER BY " + ",".join(filter_by) + " "

    order_by = params.get("order_by", None)
    order_by_asc = params.get("order_by_asc", True)
    if order_by:
        order_by_str = (
            "ORDER BY " + ",".join(order_by) + f" {'ASC' if order_by_asc else 'DESC'} "
        )

    parallel = params.get("parallel", None)
    if parallel:
        if not isinstance(parallel, int):
            raise ValueError("parallel must be int")
        parallel_str = f"PARALLEL {parallel} "

    def oracle_string_literal(value: str) -> str:
        return value.replace("'", "''")

    create_index_query = f"""
    CREATE HYBRID VECTOR INDEX {idx_name} ON 
    {vectorizer_preference.vs.table_name}(text) 
    PARAMETERS ('{oracle_string_literal(params_str)}') {filter_by_str} {order_by_str} {parallel_str}
    """  # noqa E501

    return create_index_query


@_handle_exceptions
def create_hybrid_index(
    client: Union[Connection, ConnectionPool],
    idx_name: str,
    vectorizer_preference: OracleVectorizerPreference,
    params: Optional[dict[str, Any]] = None,
) -> None:
    """Create a HYBRID VECTOR INDEX if it does not already exist.

    The index uses the provided OracleVectorizerPreference to supply the
    vectorizer configuration. Additional options are accepted via params:
    - parameters: dict of INDEX PARAMETERS (excluding model/embedder_spec/vectorizer)
    - filter_by: list[str] -> FILTER BY clause
    - order_by: list[str], order_by_asc: bool -> ORDER BY ... ASC|DESC
    - parallel: int -> PARALLEL N

    Args:
        client: oracledb connection or connection parameters
            accepted by _get_connection.
        idx_name: Index name to create (quoted automatically).
        vectorizer_preference: OracleVectorizerPreference to reference in the index.
        params: Optional dict of index options.

    Reference:
        https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html
    """
    idx_name = _quote_indentifier(idx_name)
    ddl = _get_hybrid_index_ddl(vectorizer_preference, idx_name, params or {})

    with _get_connection(client) as connection:
        if not _index_exists(connection, idx_name, vectorizer_preference.vs.table_name):
            with connection.cursor() as cur:
                cur.execute(ddl)
                logger.info(f"Index {idx_name} created successfully...")
        else:
            logger.info(f"Index {idx_name} already exists...")


@_ahandle_exceptions
async def acreate_hybrid_index(
    client: Union[AsyncConnection, AsyncConnectionPool],
    idx_name: str,
    vectorizer_preference: OracleVectorizerPreference,
    params: Optional[dict[str, Any]] = None,
) -> None:
    """Async variant of create_hybrid_index.

    Creates the HYBRID VECTOR INDEX if it does not exist, using async APIs.

    Args:
        client: oracledb connection or connection parameters.
        idx_name: Index name to create (quoted automatically).
        vectorizer_preference: OracleVectorizerPreference to reference in the index.
        params: Optional dict of index options.

    Reference:
        https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html
    """
    idx_name = _quote_indentifier(idx_name)
    ddl = _get_hybrid_index_ddl(vectorizer_preference, idx_name, params or {})

    async with _aget_connection(client) as connection:
        if not await _aindex_exists(
            connection, idx_name, vectorizer_preference.vs.table_name
        ):
            async with connection.cursor() as cur:
                await cur.execute(ddl)
                logger.info(f"Index {idx_name} created successfully...")
        else:
            logger.info(f"Index {idx_name} already exists...")


class OracleVSHybridSearchRetriever(BaseRetriever):
    """LangChain retriever that executes DBMS_HYBRID_VECTOR.SEARCH against an OracleVS table.

    Modes:
    - "semantic": vector-only using the vectorizer.
    - "keyword": text-only using Oracle Text.
    - "hybrid": combined vector + text ranking.

    Prerequisites:
    - A vectorizer preference created with DBMS_VECTOR_CHAIN.CREATE_PREFERENCE.
    - A HYBRID VECTOR INDEX created on the target table/column.

    Fields:
    - vector_store: OracleVS pointing to the table with text and metadata columns.
    - idx_name: Name of the hybrid index to query.
    - search_mode: "keyword" | "semantic" | "hybrid" (default "hybrid").
    - k: Number of results to return (default 4).
    - params: Additional DBMS_HYBRID_VECTOR.SEARCH parameters; merged per call.
    - return_scores: If True, include overall, text, and vector scores in metadata.

    Example:
        Synchronous retrieval

        retriever = OracleVSHybridSearchRetriever(
            vector_store=vs,
            idx_name="IDX_DOCS_HYB",
            search_mode="hybrid",
            k=5,
            return_scores=True,
        )
        docs = retriever.invoke("how do I reset my password?")
        for d in docs:
            print(d.page_content, d.metadata.get("score"))

    Example:
        Async retrieval

        results = await OracleVSHybridSearchRetriever(
            vector_store=vs,
            idx_name="IDX_DOCS_HYB",
            search_mode="semantic",
            k=3,
        ).ainvoke("refund policy for premium plan")

    References:
    - Hybrid search overview: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/understand-hybrid-search.html#GUID-310D2298-90F4-4AFE-AF03-F3B81E55F84C__GUID-03905981-A6E9-4D2C-A0DC-0807A95AA3F3
    - Search API: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/search.html
    """  # noqa E501

    vector_store: OracleVS
    """OracleVS VectorStore"""
    idx_name: str
    """Hybrid Index name"""
    search_mode: Optional[Literal["keyword", "hybrid", "semantic"]] = (
        "hybrid"  # keyword, hybrid, semantic
    )
    k: Optional[int] = 4
    """Number of documents to return."""
    filter: Optional[dict[str, Any]] = None
    """search filter"""
    params: Optional[dict[str, Any]] = {}
    """Search parameters"""
    return_scores: Optional[bool] = False

    # apply function to field
    @field_validator("idx_name")
    def quote_name(cls, v):
        return _quote_indentifier(v)

    # apply function to field
    @field_validator("params")
    def validate_params(cls, search_params):
        search_params = search_params or {}
        if "search_text" in search_params:
            raise ValueError(
                "Cannot provide search_text as a parameter at the top level; "
                "it is derived from the query."
            )

        vec = search_params.get("vector") or {}
        if ("search_text" in vec) or ("search_vector" in vec):
            raise ValueError(
                "Cannot provide search_text as a parameter in params['vector']; "
                "it is derived from the query."
            )

        txt = search_params.get("text") or {}
        if ("search_text" in txt) or ("search_vector" in txt) or ("contains" in txt):
            raise ValueError(
                "Cannot provide search_text as a parameter in params['text']; "
                "it is derived from the query."
            )

        if "return" in search_params:
            raise ValueError(
                "Cannot provide return as a parameter in params; "
                "it is handled internally. Use `return_scores` "
                "parameter to get the scores."
            )

        return search_params

    def _get_search_params(self, query: str, **kwargs: Any) -> dict:
        """Build the JSON-serializable parameter dict for DBMS_HYBRID_VECTOR.SEARCH.

        Behavior:
        - Sets "hybrid_index_name" to the configured idx_name.
        - Populates "vector.search_text" and/or "text.search_text" based on search_mode.
        - Sets "return.topN", "return.values" and "return.format" ("JSON").
        - Merges any user-provided "params" from kwargs into the base params.

        Args:
            query: Natural language query string.
            **kwargs: May include "params" (dict) to merge and "k" (int) to override.

        Returns:
            dict: Parameters suitable for json(:search_params)
                in DBMS_HYBRID_VECTOR.SEARCH.
        """
        search_params = dict(self.params or {})
        search_params.update(kwargs.get("params", {}))
        search_params["hybrid_index_name"] = self.idx_name

        if "search_text" in search_params:
            raise ValueError(
                "Cannot provide search_text as a parameter at the top level; "
                "it is derived from the query."
            )
        if "return" in search_params:
            raise ValueError(
                "Cannot provide return as a parameter in params; "
                "it is handled internally. Use `return_scores` "
                "parameter to get the scores."
            )

        if self.search_mode == "hybrid" or self.search_mode == "semantic":
            search_params["vector"] = dict(search_params.get("vector") or {})
            if (
                "search_text" in search_params["vector"]
                or "search_vector" in search_params["vector"]
            ):
                raise ValueError(
                    "Cannot provide search_text as a parameter in params['vector']; "
                    "it is derived from the query."
                )
            search_params["vector"]["search_text"] = query
        if self.search_mode == "hybrid" or self.search_mode == "keyword":
            search_params["text"] = dict(search_params.get("text") or {})
            if (
                "search_text" in search_params["text"]
                or "search_vector" in search_params["text"]
                or "contains" in search_params["text"]
            ):
                raise ValueError(
                    "Cannot provide search_text as a parameter in params['text']; "
                    "it is derived from the query."
                )
            search_params["text"]["search_text"] = query

        search_params["return"] = {}
        search_params["return"]["topN"] = kwargs.get("k", None) or self.k or 4
        search_params["return"]["values"] = [
            "rowid",
            "score",
            "vector_score",
            "text_score",
        ]
        search_params["return"]["format"] = "JSON"

        return search_params

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Execute a synchronous hybrid search via DBMS_HYBRID_VECTOR.SEARCH.

        Uses the configured "search_mode" to decide whether to send vector, text,
        or both signals. The procedure returns a list of (rowid, score, text_score,
        vector_score). For each rowid, the corresponding row is fetched from the
        underlying OracleVS table and converted to a LangChain Document. When
        return_scores is True, overall and component scores are added to metadata.

        Args:
            query: Natural language query string.
            **kwargs: Optional overrides including "params" (dict) and "k" (int).

        Returns:
            List[Document]: Top-k documents sorted by the hybrid score.

        Reference:
            https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/search.html
        """

        search_params = self._get_search_params(query, **kwargs)

        docs = []

        with _get_connection(self.vector_store.client) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(search_params=oracledb.DB_TYPE_JSON)
                cursor.execute(
                    "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:search_params))",
                    search_params=search_params,
                )
                res = cursor.fetchall()
                res = json.loads(res[0][0])

                rowids = []
                scores = []
                text_scores = []
                vector_scores = []
                for row in res:
                    rowids.append((row["rowid"],))
                    scores.append(row["score"])
                    text_scores.append(row["text_score"])
                    vector_scores.append(row["vector_score"])

                res = []
                cursor.outputtypehandler = output_type_string_handler
                for rid_tuple in rowids:
                    rid = rid_tuple[0]
                    cursor.execute(
                        f"SELECT text, metadata FROM {self.vector_store.table_name} "
                        "WHERE rowid = :1",
                        [rid],
                    )
                    res.extend(cursor.fetchall())

                for i, row in enumerate(res):
                    metadata = row[1]
                    doc_id = metadata.pop(INTERNAL_ID_KEY, None)
                    if self.return_scores:
                        metadata["score"] = scores[i]
                        metadata["text_score"] = text_scores[i]
                        metadata["vector_score"] = vector_scores[i]
                    doc = Document(page_content=row[0], metadata=metadata, id=doc_id)
                    docs.append(doc)

        return docs

    async def _aget_relevant_documents(
        self, query: str, **kwargs: Any
    ) -> List[Document]:
        """Async variant of _get_relevant_documents using async connection APIs.

        Args:
            query: Natural language query string.
            **kwargs: Optional overrides including "params" (dict) and "k" (int).

        Returns:
            List[Document]: Top-k documents sorted by the hybrid score.

        Reference:
            https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/search.html
        """
        search_params = self._get_search_params(query, **kwargs)

        docs = []

        async with _aget_connection(self.vector_store.client) as connection:
            async with connection.cursor() as cursor:
                cursor.setinputsizes(search_params=oracledb.DB_TYPE_JSON)
                await cursor.execute(
                    "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:search_params))",
                    search_params=search_params,
                )
                res = await cursor.fetchall()
                res = json.loads(res[0][0])

                rowids = []
                scores = []
                text_scores = []
                vector_scores = []
                for row in res:
                    rowids.append((row["rowid"],))
                    scores.append(row["score"])
                    text_scores.append(row["text_score"])
                    vector_scores.append(row["vector_score"])

                cursor.outputtypehandler = output_type_string_handler
                rows = []
                for rid_tuple in rowids:
                    rid = rid_tuple[0]
                    await cursor.execute(
                        f"SELECT text, metadata FROM {self.vector_store.table_name} "
                        "WHERE rowid = :1",
                        [rid],
                    )
                    rows.extend(await cursor.fetchall())

                for i, row in enumerate(rows):
                    metadata = row[1]
                    doc_id = metadata.pop(INTERNAL_ID_KEY, None)
                    if self.return_scores:
                        metadata["score"] = scores[i]
                        metadata["text_score"] = text_scores[i]
                        metadata["vector_score"] = vector_scores[i]
                    doc = Document(page_content=row[0], metadata=metadata, id=doc_id)
                    docs.append(doc)

        return docs
