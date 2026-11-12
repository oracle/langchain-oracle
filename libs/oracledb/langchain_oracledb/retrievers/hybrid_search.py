# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Hybrid search utilities for Oracle Database.

This module provides:
- OracleVectorizerPreference: create/drop DBMS_VECTOR_CHAIN vectorizer preferences used
    by hybrid vector indexes.
- create_hybrid_index/acreate_hybrid_index: build a HYBRID VECTOR INDEX over a text
    column using a vectorizer preference.
- OracleHybridSearchRetriever: a retriever that calls DBMS_HYBRID_VECTOR.SEARCH
    to perform keyword, semantic, or hybrid retrieval against an OracleVS table.

References:
- Hybrid search overview and guidance: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/understand-hybrid-search.html#GUID-310D2298-90F4-4AFE-AF03-F3B81E55F84C__GUID-03905981-A6E9-4D2C-A0DC-0807A95AA3F3
- Create vectorizer preference: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create_preference.html
- Create hybrid vector index: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html
- Search API (DBMS_HYBRID_VECTOR.SEARCH): https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/search.html
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from copy import copy
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union, cast

import oracledb
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import field_validator

from langchain_oracledb._telemetry import tag_sql
from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.vectorstores.oraclevs import INTERNAL_ID_KEY, OracleVS
from langchain_oracledb.vectorstores.utils import (
    _aget_connection,
    _ahandle_exceptions,
    _aindex_exists,
    _get_connection,
    _handle_exceptions,
    _index_exists,
    _quote_indentifier,
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


def _preference_client(store: OracleVS) -> Any:
    if store._vecdb is not None:
        return store._vecdb.native_transport("vectorizer preference management").client
    return store.client


def _bound_vecdb_store(store: OracleVS, transport: Any) -> OracleVS:
    """Keep temporary preference and index DDL on one native checkout."""
    bound = copy(store)
    bound.client = transport.client
    bound._vecdb = copy(store._vecdb)
    bound._vecdb.transport = transport
    return bound


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
            OracleHybridSearchRetriever,
        )
        import os
        import oracledb

        # Connect and prepare embeddings and vector store
        client = oracledb.connect(
            dsn=os.environ["ORACLE_DB_DSN"]
        )
        embeddings = OracleEmbeddings(
            conn=client,
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
        retriever = OracleHybridSearchRetriever(vector_store=vs, idx_name="IDX_DOCS_HYB", k=5)
        docs = retriever.invoke("refund policy for premium plan")

        # Cleanup when needed
        pref.drop_preference()
    """  # noqa E501

    params: Optional[dict[str, Any]]
    vs: OracleVS
    preference_name: str

    PREFERENCE_STR = """
    begin
    dbms_vector_chain.CREATE_PREFERENCE(
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

        if self.vs._vecdb is not None:
            from ..vectorstores._vecdb_adapter import TEXT_KEY

            preference_params.setdefault(
                "paths", [{"type": "STRING", "path_list": [f"$.{TEXT_KEY}"]}]
            )
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

        client = _preference_client(vector_store)
        preference_params = self._get_preference_parameters()

        with _get_connection(client) as connection:
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
        self.preference_name = (
            preference_name or "pref" + str(uuid.uuid4()).replace("-", "")[0:15]
        )

        client = _preference_client(vector_store)
        preference_params = self._get_preference_parameters()

        async with _aget_connection(client) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    OracleVectorizerPreference.PREFERENCE_STR,
                    [self.preference_name, json.dumps(preference_params)],
                )
                logger.info(f"Preference {self.preference_name} created.")

        return self

    def drop_preference(self):
        """Drop this vectorizer preference using DBMS_VECTOR_CHAIN.DROP_PREFERENCE."""
        with _get_connection(_preference_client(self.vs)) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "begin DBMS_VECTOR_CHAIN.DROP_PREFERENCE (:preference_name); end;",
                    preference_name=self.preference_name,
                )

    async def adrop_preference(self):
        """Async variant of drop_preference."""
        async with _aget_connection(_preference_client(self.vs)) as connection:
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
    metadata_parameters: Optional[str] = None,
    metadata_column: Optional[str] = None,
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

    def quote_filter_order_identifier(value: str, field_name: str) -> str:
        value = value.strip()
        simple_identifier = r"[A-Za-z][A-Za-z0-9_$#]*"
        reg = (
            rf'^(?:"{simple_identifier}"|{simple_identifier})'
            rf'(?:\.(?:"{simple_identifier}"|{simple_identifier}))*$'
        )
        if not re.fullmatch(reg, value):
            raise ValueError(f"{field_name} contains an invalid identifier")

        pattern_match = rf'"({simple_identifier})"|({simple_identifier})'
        groups = re.findall(pattern_match, value)
        quoted_groups = [
            f'"{quoted}"' if quoted else f'"{unquoted.upper()}"'
            for quoted, unquoted in groups
        ]
        return ".".join(quoted_groups)

    def quote_identifier_list(values: Any, field_name: str) -> str:
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{field_name} must be a list of column names")
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"{field_name} must contain only column names")
        return ",".join(
            quote_filter_order_identifier(value, field_name) for value in values
        )

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

    if metadata_parameters is not None:
        from langchain_oracledb.vectorstores._vecdb import (
            DATAGUIDE_CALLBACK,
            table_identifier,
        )

        table_identifier(vectorizer_preference.preference_name)
        if re.search(
            r"\bSYNC\s*\(\s*ON\s+COMMIT\s*\)", metadata_parameters, re.IGNORECASE
        ):
            raise ValueError(
                "VecDB hybrid replacement cannot inherit SYNC (ON COMMIT); "
                "use the original scheduled data-guide synchronization"
            )

        for key in index_parameters:
            if key.lower() == "dataguide":
                if (
                    str(index_parameters[key]).upper().split()
                    != DATAGUIDE_CALLBACK.split()
                ):
                    raise ValueError(
                        "VecDB indexing must retain its data-guide callback"
                    )
                index_parameters = {
                    k: v for k, v in index_parameters.items() if k != key
                }
                break
        if any(k.lower() == "search_on" for k in index_parameters):
            raise ValueError("SEARCH_ON is not a hybrid vector index parameter")
        if any(k.lower() == "sync" for k in index_parameters):
            raise ValueError(
                "VecDB hybrid indexing preserves the existing synchronization "
                "schedule; do not override sync"
            )
        for key in ("memory", "maintenance"):
            if any(k.lower() == key for k in index_parameters):
                metadata_parameters = re.sub(
                    rf"\b{key}\s+\S+", "", metadata_parameters, flags=re.IGNORECASE
                )

    params_str, filter_by_str, order_by_str, parallel_str = "", "", "", ""

    params_str = f"vectorizer {vectorizer_preference.preference_name} "
    if metadata_parameters is not None:
        params_str += metadata_parameters + " "
    for k, v in index_parameters.items():
        params_str += f"{k} {v} "

    filter_by = params.get("filter_by", None)
    if filter_by:
        filter_by_str = (
            "FILTER BY " + quote_identifier_list(filter_by, "filter_by") + " "
        )

    order_by = params.get("order_by", None)
    order_by_asc = params.get("order_by_asc", True)
    if not isinstance(order_by_asc, bool):
        raise ValueError("order_by_asc must be a boolean")
    if order_by:
        order_by_str = (
            "ORDER BY "
            + quote_identifier_list(order_by, "order_by")
            + f" {'ASC' if order_by_asc else 'DESC'} "
        )

    parallel = params.get("parallel", None)
    if parallel is not None:
        if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel <= 0:
            raise ValueError("parallel must be a positive integer")
        parallel_str = f"PARALLEL {parallel} "

    column = "text"
    if vectorizer_preference.vs._vecdb is not None:
        from langchain_oracledb.vectorstores._vecdb import CURRENT_VERSION

        transport = vectorizer_preference.vs._vecdb.native_transport()
        column = (
            "CONTENT_METADATA" if transport.version >= CURRENT_VERSION else "METADATA"
        )

    if metadata_column is not None:
        if metadata_column not in {"METADATA", "CONTENT_METADATA"}:
            raise ValueError("Unsupported VecDB metadata column")
        column = metadata_column

    from langchain_oracledb.vectorstores._vecdb_adapter import hybrid_index_ddl

    return (
        hybrid_index_ddl(
            idx_name, vectorizer_preference.vs.table_name, column, params_str
        )
        + " "
        + filter_by_str
        + order_by_str
        + parallel_str
    )


def _vecdb_hybrid_builder(
    preference: OracleVectorizerPreference, params: Optional[dict[str, Any]]
):
    def build(index: str, metadata_parameters: str, column: str) -> str:
        return _get_hybrid_index_ddl(
            preference,
            index,
            params or {},
            metadata_parameters=metadata_parameters,
            metadata_column=column,
        )

    return build


@_handle_exceptions
def create_hybrid_index(
    client: Union[Connection, ConnectionPool],
    idx_name: str,
    vectorizer_preference: Optional[OracleVectorizerPreference] = None,
    vector_store: Optional[OracleVS] = None,
    params: Optional[dict[str, Any]] = None,
) -> None:
    """Create a HYBRID VECTOR INDEX if it does not already exist.

    For a VecDB store, this replaces the metadata guide/text index while retaining
    its physical name, data-guide callback, and compatible settings. VecDB
    retrieval discovers that physical index; its idx_name is ignored. Existing HVI
    indexes are reused. Replacement requires a connection without pending work.

    The index uses either the provided OracleVectorizerPreference or, if
    vector_store is given, a temporary preference derived from its
    OracleEmbeddings to supply the vectorizer configuration. Additional options
    are accepted via params:
    - parameters: dict of INDEX PARAMETERS (excluding model/embedder_spec/vectorizer)
    - filter_by: list[str] -> FILTER BY clause
    - order_by: list[str], order_by_asc: bool -> ORDER BY ... ASC|DESC
    - parallel: int -> PARALLEL N

    Args:
        client: oracledb connection or connection parameters accepted
            by _get_connection.
        idx_name: Index name to create (quoted automatically).
        vectorizer_preference: Existing OracleVectorizerPreference to
            reference in the index. Mutually exclusive with vector_store.
        vector_store: OracleVS instance. If provided, a temporary vectorizer preference
            is created from its OracleEmbeddings for the duration of index creation,
            then dropped. Mutually exclusive with vectorizer_preference.
        params: Optional dict of index options.

    Reference:
        https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html
    """
    if (not vector_store and not vectorizer_preference) or (
        vector_store and vectorizer_preference
    ):
        raise ValueError(
            "Exactly one of 'vector_store' or 'vectorizer_preference' must be provided."
        )

    target = vector_store or (
        vectorizer_preference.vs if vectorizer_preference else None
    )
    if target is not None and target._vecdb is not None:
        from langchain_oracledb.vectorstores._vecdb import _is_native_pool

        transport = target._vecdb.native_transport("hybrid index creation")
        if _is_native_pool(transport.client):
            with transport.session() as bound:
                store = _bound_vecdb_store(target, bound)
                preference = copy(vectorizer_preference)
                if preference is not None:
                    preference.vs = store
                return create_hybrid_index(
                    bound.client,
                    idx_name,
                    vectorizer_preference=preference,
                    vector_store=store if vector_store is not None else None,
                    params=params,
                )
        if getattr(transport.client, "transaction_in_progress", False) is True:
            raise RuntimeError("Index DDL requires a connection without pending work")

    drop = False
    if vector_store:
        vectorizer_preference = OracleVectorizerPreference.create_preference(
            vector_store
        )
        drop = True

    vectorizer_preference = cast(
        OracleVectorizerPreference,
        vectorizer_preference,
    )

    try:
        if vectorizer_preference.vs._vecdb is not None:
            backend = vectorizer_preference.vs._vecdb
            backend.native_transport().create_metadata_index(
                backend.name,
                idx_name,
                hybrid=True,
                build=_vecdb_hybrid_builder(vectorizer_preference, params),
            )
            return
        idx_name = _quote_indentifier(idx_name)
        ddl = _get_hybrid_index_ddl(vectorizer_preference, idx_name, params or {})
        with _get_connection(client) as connection:
            if not _index_exists(
                connection, idx_name, vectorizer_preference.vs.table_name
            ):
                with connection.cursor() as cur:
                    cur.execute(ddl)
                    logger.info(f"Index {idx_name} created successfully...")
            else:
                logger.info(f"Index {idx_name} already exists...")
    finally:
        if drop:
            vectorizer_preference.drop_preference()


@_ahandle_exceptions
async def acreate_hybrid_index(
    client: Union[AsyncConnection, AsyncConnectionPool],
    idx_name: str,
    vectorizer_preference: Optional[OracleVectorizerPreference] = None,
    vector_store: Optional[OracleVS] = None,
    params: Optional[dict[str, Any]] = None,
) -> None:
    """Async variant of create_hybrid_index.

    Creates the HYBRID VECTOR INDEX if it does not exist, using async APIs.
    VecDB guide/text indexes are replaced with their name and compatible settings
    retained; sync native clients are dispatched to a worker thread.
    For VecDB, serialize the entire helper across maintenance callers/processes.
    A returned submission or completed DDL does not guarantee text sync freshness.

    Args:
        client: oracledb async connection or async connection pool.
        idx_name: Index name to create (quoted automatically).
        vectorizer_preference: Existing OracleVectorizerPreference to reference in the
            index. Mutually exclusive with vector_store.
        vector_store: OracleVS instance. If provided, a temporary vectorizer preference
            is created from its OracleEmbeddings for the duration of index creation,
            then dropped. Mutually exclusive with vectorizer_preference.
        params: Optional dict of index options.

    Reference:
        https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/create-hybrid-vector-index.html
    """
    target = vector_store or (
        vectorizer_preference.vs if vectorizer_preference else None
    )
    if target is not None and target._vecdb is not None:
        from langchain_oracledb.vectorstores._vecdb import (
            AsyncNativeTransport,
            _is_native_pool,
        )

        transport = target._vecdb.native_transport("hybrid index creation")
        if isinstance(transport, AsyncNativeTransport) and _is_native_pool(
            transport.client, asynchronous=True
        ):
            async with transport.asession() as bound:
                store = _bound_vecdb_store(target, bound)
                preference = copy(vectorizer_preference)
                if preference is not None:
                    preference.vs = store
                return await acreate_hybrid_index(
                    bound.client,
                    idx_name,
                    vectorizer_preference=preference,
                    vector_store=store if vector_store is not None else None,
                    params=params,
                )
        if not isinstance(
            target._vecdb.native_transport("hybrid index creation"),
            AsyncNativeTransport,
        ):
            await asyncio.to_thread(
                create_hybrid_index,
                target._vecdb.native_transport().client,
                idx_name,
                vectorizer_preference,
                vector_store,
                params,
            )
            return
        if getattr(transport.client, "transaction_in_progress", False) is True:
            raise RuntimeError("Index DDL requires a connection without pending work")
    if (not vector_store and not vectorizer_preference) or (
        vector_store and vectorizer_preference
    ):
        raise ValueError(
            "Exactly one of 'vector_store' or 'vectorizer_preference' must be provided."
        )

    drop = False
    if vector_store:
        vectorizer_preference = await OracleVectorizerPreference.acreate_preference(
            vector_store
        )
        drop = True

    vectorizer_preference = cast(
        OracleVectorizerPreference,
        vectorizer_preference,
    )

    try:
        if vectorizer_preference.vs._vecdb is not None:
            backend = vectorizer_preference.vs._vecdb
            await backend.native_transport().acreate_metadata_index(
                backend.name,
                idx_name,
                hybrid=True,
                build=_vecdb_hybrid_builder(vectorizer_preference, params),
            )
            return
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
    finally:
        if drop:
            if vectorizer_preference.vs._vecdb is not None:
                from langchain_oracledb.vectorstores._vecdb import _complete_cleanup

                await _complete_cleanup(vectorizer_preference.adrop_preference())
            else:
                await vectorizer_preference.adrop_preference()


class OracleHybridSearchRetriever(BaseRetriever):
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

        retriever = OracleHybridSearchRetriever(
            vector_store=vs,
            idx_name="IDX_DOCS_HYB",
            search_mode="hybrid",
            k=5,
            return_scores=True,
        )
        docs = retriever.invoke("how do I rotate my database credentials?")
        for d in docs:
            print(d.page_content, d.metadata.get("score"))

    Example:
        Async retrieval

        results = await OracleHybridSearchRetriever(
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
    """SQL hybrid index name. VecDB ignores this and discovers its physical index."""
    search_mode: Optional[Literal["keyword", "hybrid", "semantic"]] = (
        "hybrid"  # keyword, hybrid, semantic
    )
    k: Optional[int] = 4
    """Number of documents to return."""
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
        from langchain_oracledb.vectorstores._vecdb_adapter import (
            validate_retriever_options,
        )

        return validate_retriever_options(search_params or {})

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
        from langchain_oracledb.vectorstores._vecdb_adapter import (
            validate_retriever_options,
        )

        search_params = dict(self.params or {})
        search_params.update(kwargs.get("params", {}))
        validate_retriever_options(search_params)
        search_params["hybrid_index_name"] = self.idx_name
        if self.search_mode in {"hybrid", "semantic"}:
            search_params["vector"] = dict(search_params.get("vector") or {})
            search_params["vector"]["search_text"] = query
        if self.search_mode in {"hybrid", "keyword"}:
            search_params["text"] = dict(search_params.get("text") or {})
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

    def _get_vecdb_search_params(self, query: str, **kwargs: Any) -> dict:
        from langchain_oracledb.vectorstores._vecdb_adapter import (
            retriever_hybrid_options,
        )

        return retriever_hybrid_options(
            query,
            self.search_mode or "hybrid",
            self.idx_name,
            kwargs.get("k", self.k or 4),
            self.params or {},
            kwargs.get("params", {}),
        )

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
        if self.vector_store._vecdb is not None:
            self.vector_store._vecdb.native_transport("hybrid search")
            search_params = self._get_vecdb_search_params(query, **kwargs)
            return self.vector_store._vecdb.hybrid_search(
                search_params, bool(self.return_scores)
            )

        search_params = self._get_search_params(query, **kwargs)

        docs = []

        with _get_connection(self.vector_store.client) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(search_params=oracledb.DB_TYPE_JSON)
                cursor.execute(
                    tag_sql(
                        "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:search_params))",
                        self.search_mode or "hybrid",
                    ),
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
        if self.vector_store._vecdb is not None:
            self.vector_store._vecdb.native_transport("hybrid search")
            search_params = self._get_vecdb_search_params(query, **kwargs)
            return await self.vector_store._vecdb.run_async(
                "hybrid_search", search_params, bool(self.return_scores)
            )
        search_params = self._get_search_params(query, **kwargs)

        docs = []

        async with _aget_connection(self.vector_store.client) as connection:
            async with connection.cursor() as cursor:
                cursor.setinputsizes(search_params=oracledb.DB_TYPE_JSON)
                await cursor.execute(
                    tag_sql(
                        "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:search_params))",
                        self.search_mode or "hybrid",
                    ),
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
