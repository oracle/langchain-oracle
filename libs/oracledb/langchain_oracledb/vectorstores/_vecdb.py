# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""VecDB backend operations over native or REST transports."""

from __future__ import annotations

import asyncio
import logging
import math
import warnings
from copy import copy
from decimal import Decimal
from functools import wraps
from typing import Any, Literal, Optional, cast

import numpy as np
from langchain_core.documents import Document

from .._annotations import TableType, table_annotations
from .._vecdb_sdk.parameter_validation import validate_operation_arguments
from . import _vecdb_schema as _schema
from . import utils as _db
from ._index_params import (
    _validate_index_type,
    _validate_int_param,
    _validate_vector_index_common,
    normalize_index_params,
)
from ._vecdb_adapter import (
    CURRENT_VERSION,
    LOGICAL_ID_KEY,
    TEXT_KEY,
    RestTransport,
    document,
    rows,
    search_request,
    storage_id,
)
from ._vecdb_capabilities import sdk_index_state, sdk_index_status
from ._vecdb_filters import SQLFilterRequired
from ._vecdb_native import (
    AsyncNativeTransport,
    NativeTransport,
    _is_native_pool,
    _NativeMethod,
)
from .utils import DistanceStrategy, _get_distance_function

logger = logging.getLogger(__name__)


def _with_native_backend_session(method: _NativeMethod) -> _NativeMethod:
    """Keep composed operations (prepare, batches and migration retries) together."""

    @wraps(method)
    def wrapped(self: VecDBBackend, *args: Any, **kwargs: Any) -> Any:
        transport = self.transport
        if not isinstance(transport, NativeTransport) or not _is_native_pool(
            transport.client
        ):
            return method(self, *args, **kwargs)
        with transport.session() as bound:
            backend = copy(self)
            backend.transport = bound
            return method(backend, *args, **kwargs)

    return cast(_NativeMethod, wrapped)


def _with_async_native_backend_session(method: _NativeMethod) -> _NativeMethod:
    @wraps(method)
    async def wrapped(self: VecDBBackend, *args: Any, **kwargs: Any) -> Any:
        transport = self.transport
        if not isinstance(transport, AsyncNativeTransport) or not _is_native_pool(
            transport.client, asynchronous=True
        ):
            return await method(self, *args, **kwargs)
        async with transport.asession() as bound:
            backend = copy(self)
            backend.transport = bound
            return await method(backend, *args, **kwargs)

    return cast(_NativeMethod, wrapped)


class VecDBBackend:
    def __init__(
        self,
        transport: Any,
        name: str,
        metric: DistanceStrategy,
        *,
        table_type: TableType = "vector_store",
        vector_index_params: Optional[dict] = None,
        embedding: Any = None,
        mutate_on_duplicate: Optional[bool] = False,
    ):
        self.transport = transport
        self._selection_name = name
        self.table_type = table_type
        self.quoted, api_name = _schema.table_identifier(name)
        # Native calls must retain component boundaries, including literal dots.
        self.name = self.quoted if isinstance(transport, NativeTransport) else api_name
        self.metric = _get_distance_function(
            DistanceStrategy.DOT_PRODUCT
            if metric == DistanceStrategy.MAX_INNER_PRODUCT
            else metric
        )
        self._creation_index_params = self.index_params(
            vector_index_params, operation="create_table"
        )
        self._constructor_index_requested = (
            vector_index_params is not None
            and vector_index_params.get("auto_index") is not False
        )
        self._embed_params = self._managed_embed_params(embedding, mutate_on_duplicate)
        validate_operation_arguments(
            "create_vector_table",
            {
                "index_params": self._creation_index_params,
                "embed_params": self._embed_params,
            },
        )

    def _managed_embed_params(
        self, embedding: Any, mutate: Optional[bool]
    ) -> Optional[dict]:
        from ..embeddings import OracleEmbeddings

        if not mutate or not isinstance(embedding, OracleEmbeddings):
            return None
        if (
            isinstance(self.transport, NativeTransport)
            and self.transport.version < CURRENT_VERSION
        ):
            return None
        params = embedding.params
        # Extra provider options and proxy settings retain their existing path.
        if (
            params.get("provider", "database") != "database"
            or set(params) - {"provider", "model"}
            or embedding.proxy
        ):
            return None
        model = params.get("model")
        if not isinstance(model, str) or not model.strip():
            return None
        return {"model": model, "embed_metadata_jsonpath": TEXT_KEY}

    @property
    def managed_embeddings(self) -> bool:
        return self._embed_params is not None

    def _validate_embedding_config(self, description: dict) -> None:
        expected = self._embed_params
        if expected is None:
            return
        stored = description.get("embed_params")

        def model_name(value: Any) -> Optional[str]:
            if not isinstance(value, str):
                return None
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1]
            return value.upper()

        if (
            not isinstance(stored, dict)
            or model_name(stored.get("model")) != model_name(expected["model"])
            or stored.get("embed_metadata_jsonpath") != TEXT_KEY
        ):
            raise ValueError(
                "Existing VecDB table embedding configuration does not match "
                "OracleEmbeddings model and content path; reopen with its original "
                "configuration or use a new table. BYOV tables are not converted "
                "to managed embeddings automatically."
            )

    def _table_creation_params(self) -> dict:
        params = {
            "name": self.name,
            "table_params": {"auto_generate_id": False},
            "annotations": table_annotations(self.table_type, "vecdb"),
        }
        if self._creation_index_params is not None:
            params["index_params"] = self._creation_index_params
        if self._embed_params is not None:
            params["embed_params"] = dict(self._embed_params)
        return params

    def _ensure_constructor_index(self) -> None:
        """Create a missing requested index without reconfiguring an existing one."""
        if self._constructor_index_requested and not self._vector_index_exists():
            _, request = self._index_request(self._creation_index_params, False)
            validate_operation_arguments("create_index", request)
            self.transport.call("create_index", **request)

    @_with_native_backend_session
    def prepare(
        self, *, columns: Optional[set[str]] = None, description: Optional[dict] = None
    ) -> None:
        if isinstance(self.transport, NativeTransport):
            if columns is None:
                description = description if description is not None else {}
                with _db._get_connection(self.transport.client) as connection:
                    layout: set[str] = set()
                    _schema.native_selection(
                        connection,
                        self._selection_name,
                        "vecdb",
                        description=description,
                        layout=layout,
                    )
                columns = layout if description else set()
            created = not columns
            if created:
                self.transport.call(
                    "create_vector_table", **self._table_creation_params()
                )
                columns = _schema._default_columns(self.transport.version)
            elif self.managed_embeddings:
                self._validate_embedding_config(
                    description
                    if description
                    else self.transport.call("describe_vector_table", name=self.name)
                )
            self.transport._remember_table(
                self.name, _schema._table_info(columns, None)
            )
            if not created:
                self._ensure_constructor_index()
            logger.info(
                "Table %s %s...",
                self.quoted,
                "created successfully" if created else "already exists",
            )
            return
        try:
            description = self.transport.call("describe_vector_table", name=self.name)
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                raise
            self.transport.call(
                "create_vector_table",
                **self._table_creation_params(),
            )
            logger.info("Table %s created successfully...", self.quoted)
        else:
            self._validate_embedding_config(description)
            self._ensure_constructor_index()
            logger.info("Table %s already exists...", self.quoted)

    @_with_native_backend_session
    def query(self, operation: str, **params: Any) -> dict:
        try:
            return self.transport.call(operation, **params)
        except Exception as exc:
            if not _schema.layout_error(exc):
                raise
            self.transport.migrate(self.name)
            return self.transport.call(operation, **params)

    def records(
        self,
        texts: list[str],
        vectors: Optional[list[list[float]]],
        metadatas: list[dict],
        ids: list[str],
    ) -> list[dict]:
        if not texts:
            return []
        if vectors is None and not self.managed_embeddings:
            raise ValueError("Explicit vectors are required for a BYOV table")
        if vectors is not None and len(texts) != len(vectors):
            raise ValueError("Text and embedding batch sizes must match")
        records = []
        for offset, (text, metadata, id_) in enumerate(zip(texts, metadatas, ids)):
            if LOGICAL_ID_KEY in metadata:
                raise ValueError(
                    f"Document metadata key {LOGICAL_ID_KEY!r} is reserved"
                )
            if TEXT_KEY in metadata:
                raise ValueError(
                    f"Document metadata key {TEXT_KEY!r} is reserved for content"
                )
            record: dict[str, Any] = {
                "id": storage_id(id_),
                "metadata": {**metadata, TEXT_KEY: text, LOGICAL_ID_KEY: id_},
            }
            if vectors is not None:
                record["dense_vector"] = vectors[offset]
            records.append(record)
        return records

    def add(
        self,
        texts: list[str],
        vectors: Optional[list[list[float]]],
        metadatas: list[dict],
        ids: list[str],
        mutate: bool,
    ) -> list[str]:
        records = self.records(texts, vectors, metadatas, ids)
        if not records:
            return []
        if not mutate:
            if vectors is None:
                raise ValueError("Managed VecDB embeddings require upsert writes")
            if isinstance(self.transport, RestTransport):
                raise ValueError(
                    "oracle-vecdb REST SDK does not support insert-only writes; "
                    "set mutate_on_duplicate=True to enable upserts "
                    "(existing IDs will be overwritten)"
                )
            inserted = self.transport.insert(self.name, records)
            logical_ids = {record["id"]: id_ for record, id_ in zip(records, ids)}
            return [logical_ids[id_] for id_ in inserted]
        self.transport.call("upsert_vectors", table_name=self.name, vectors=records)
        return ids

    def search(
        self,
        embedding: list[float],
        k: int,
        filter: Optional[dict],
        include_vectors: bool = False,
        **kwargs: Any,
    ) -> list:
        request, fallback = self._search_request(
            embedding, k, filter, include_vectors, kwargs
        )
        response = (
            self.transport.filter_search(self.name, filter, request)
            if fallback
            else self.query("search", **request)
        )
        return self.search_results(response, include_vectors)

    def _search_request(
        self, embedding: list, k: int, filter: Any, include: bool, options: dict
    ) -> tuple:
        try:
            return search_request(
                self.name, self.metric, embedding, k, filter, include, options
            ), False
        except SQLFilterRequired:
            if not isinstance(self.transport, NativeTransport):
                raise
            # Revalidate all search options even when filter translation cannot
            # use the package. Only the typed translation signal selects SQL.
            request = search_request(
                self.name, self.metric, embedding, k, None, include, options
            )
            logger.debug(
                "Using native SQL filter fallback for VecDB table %s", self.name
            )
            return request, True

    def search_results(self, response: dict, include_vectors: bool) -> list:
        result = []
        for row in rows(response, "results"):
            distance = row.get("distance")
            if not isinstance(distance, (int, float, Decimal)) or not math.isfinite(
                distance
            ):
                raise ValueError("VecDB returned an invalid distance")
            hit: tuple[Any, ...] = (document(row), float(distance))
            if include_vectors:
                if not isinstance(row.get("vector"), list):
                    raise ValueError("VecDB did not return requested embeddings")
                hit += (np.asarray(row["vector"], dtype=np.float32),)
            result.append(hit)
        return result

    @_with_native_backend_session
    def get(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        result: list[Document] = []
        for offset in range(0, len(ids), 100):
            batch = [storage_id(id_) for id_ in ids[offset : offset + 100]]
            response = self.query(
                "list_vectors",
                table_name=self.name,
                ids=batch,
                limit=len(batch),
                offset=0,
            )
            result.extend(document(row) for row in rows(response, "vectors"))
        return result

    def delete(self, ids: Optional[list[str]]) -> None:
        self.transport.call(
            "delete_vectors",
            table_name=self.name,
            ids=None if ids is None else [storage_id(id_) for id_ in ids],
        )

    def text_search(self, query: str, k: int, scores: bool = False) -> list[Document]:
        return self.transport.text_search(self.name, query, k, scores)

    def native_transport(
        self, operation: str = "SQL-based metadata index operations"
    ) -> NativeTransport:
        if isinstance(self.transport, RestTransport):
            raise ValueError(f"oracle-vecdb REST SDK does not support {operation}")
        return self.transport

    def hybrid_search(self, params: dict, scores: bool) -> list[Document]:
        return self.transport.hybrid_search(self.name, params, scores)

    def create_text_index(self, index: str) -> None:
        self.native_transport("text index creation").create_text_index(self.name, index)

    def describe_index(
        self, kind: str = "vector", job_name: Optional[str] = None
    ) -> dict:
        if kind not in {"vector", "metadata"}:
            raise ValueError("Index kind must be 'vector' or 'metadata'")
        if isinstance(self.transport, RestTransport):
            if kind == "metadata":
                raise ValueError(
                    "oracle-vecdb REST SDK does not support metadata index status"
                )
            response = self.transport.call(
                "describe_index_job" if job_name is not None else "describe_index",
                **(
                    {"index_job_name": job_name}
                    if job_name is not None
                    else {"table_name": self.name}
                ),
            )
            return sdk_index_status(response, job_name is not None)
        if job_name is not None:
            raise ValueError(
                "Native VecDB job status is unsupported; "
                "describe the physical index instead"
            )
        return self.transport.describe_index(self.name, kind)

    def index_params(
        self,
        params: Optional[dict],
        *,
        operation: Literal["create_table", "create_index"] = "create_index",
    ) -> Optional[dict]:
        """Map table policy separately from explicit index maintenance."""
        if operation == "create_table" and params is None:
            # Preserve OracleVS's opt-in vector-index creation.
            if (
                isinstance(self.transport, NativeTransport)
                and self.transport.version < CURRENT_VERSION
            ):
                return {"indexing": "manual"}
            return {"vector_index_params": {"auto_index": False}}
        if (
            operation != "create_table"
            and isinstance(params, dict)
            and "auto_index" in params
        ):
            raise ValueError(
                "auto_index is a table-creation policy and is unsupported for "
                "explicit index creation or rebuild"
            )
        params = normalize_index_params(params, metric=self.metric)
        if params is not None and "idx_name" in params:
            params.pop("idx_name")
            warnings.warn(
                "idx_name is ignored for VecDB because VecDB assigns physical "
                "index names. Use store.describe_vecdb_index() to discover "
                "the physical index name.",
                UserWarning,
                stacklevel=3,
            )
        if not params:
            if operation != "create_table":
                # An empty maintenance request only ensures an index exists.
                return None
            params = {"idx_type": "HNSW", "parallel": 8}
        automatic_supplied = "auto_index" in params
        automatic = params.pop("auto_index", None)
        if automatic_supplied and type(automatic) is not bool:
            raise ValueError("vector_index_params.auto_index must be a boolean")
        params = dict(params or {})
        if params.get("distance_metric", self.metric) != self.metric:
            raise ValueError("Index distance_metric must match distance_strategy")
        kind = params.get("idx_type", "HNSW")
        kind = "IVF" if isinstance(kind, str) and kind.upper() == "IVF" else "HNSW"
        _validate_index_type(params, kind)
        params.setdefault("idx_name", kind)
        _validate_vector_index_common(params)
        params.pop("idx_type", None)
        params.pop("idx_name", None)
        accuracy = params.pop("accuracy", 90)
        parallel = params.pop("parallel", None)
        if kind == "IVF" and "partitions" in params:
            params.setdefault("neighbor_part", params.pop("partitions"))
        mapped: dict[str, Any] = {
            "organization": "INMEMORY GRAPH" if kind == "HNSW" else "PARTITIONS",
            "distance_metric": self.metric,
            "accuracy": accuracy,
        }
        # Materialize OracleVS construction defaults for explicit requests:
        # supported packages can emit invalid DDL when these are omitted.
        advanced = (
            {"neighbors": 32, "efConstruction": 200}
            if kind == "HNSW"
            else {"partitions": 32}
        )
        ranges = (
            {"neighbors": (2, 2048), "efConstruction": (1, 65535)}
            if kind == "HNSW"
            else {"neighbor_part": (1, 10000000)}
        )
        for key, (low, high) in ranges.items():
            _validate_int_param(params, key, low, high)
            if key in params:
                advanced["partitions" if key == "neighbor_part" else key] = params.pop(
                    key
                )
        mapped.update(params)  # The SDK/database validates additional options.
        if mapped.get("distribute_params") is None:
            mapped.pop("distribute_params", None)
        if advanced:
            supplied = mapped.get("advanced_params")
            if supplied is None:
                mapped["advanced_params"] = advanced
            elif isinstance(supplied, dict):
                mapped["advanced_params"] = {**advanced, **supplied}
            # Leave malformed SDK options for its validation to reject.
        if getattr(self.transport, "version", CURRENT_VERSION) < CURRENT_VERSION:
            if automatic_supplied:
                mapped["indexing"] = "auto" if automatic else "manual"
            if parallel is not None:
                mapped["parallel_creation"] = parallel
            return mapped
        if automatic_supplied:
            mapped["auto_index"] = automatic
        mapped["organization"] = "INMEMORY GRAPH" if kind == "HNSW" else "PARTITIONS"
        result: dict[str, Any] = {"vector_index_params": mapped}
        if parallel is not None:
            result["parallel_creation"] = parallel
        return result

    @staticmethod
    def _index_present(response: dict) -> bool:
        state = sdk_index_state(response)
        if state == "VECTOR INDEX IS NOT CREATED":
            return False
        if state in {
            "READY",
            "VALID",
            "INDEX CREATION COMPLETED",
            "BUILDING",
            "PENDING",
            "FAILED",
            "UNUSABLE",
            "EXISTS",
        }:
            return True
        raise ValueError(
            f"Cannot determine VecDB vector-index existence from status {state!r}"
        )

    def _vector_index_exists(self) -> bool:
        if isinstance(self.transport, NativeTransport):
            status = self.transport.vector_index_status(self.name)
            try:
                return self._index_present(status)
            except ValueError:
                # Older packages can return unstructured progress text. Preserve
                # existing/building indexes when the API cannot prove membership.
                return bool(
                    self.transport.describe_index(self.name)["details"]["indexes"]
                )
        try:
            status = self.describe_index()
        except Exception as error:
            if getattr(error, "status", None) != 404:
                raise
            self.transport.call("describe_vector_table", name=self.name)
            return False
        exists = self._index_present(status["details"])
        if not exists:
            self.transport.call("describe_vector_table", name=self.name)
        return exists

    async def _avector_index_exists(self) -> bool:
        status = await self.transport.avector_index_status(self.name)
        try:
            return self._index_present(status)
        except ValueError:
            return bool(
                (await self.transport.adescribe_index(self.name))["details"]["indexes"]
            )

    def _index_request(self, mapped: Optional[dict], exists: bool) -> tuple[str, dict]:
        params: dict[str, Any] = {"table_name": self.name}
        operation = "rebuild_index" if exists else "create_index"
        if not exists and mapped is None:
            # Match SQL's default HNSW configuration and the store's metric.
            # Decide this only after existence is known so empty requests never
            # rebuild an existing index, including an unusable one.
            mapped = self.index_params({}, operation="create_table")
        if not exists and isinstance(self.transport, RestTransport):
            # ORDS otherwise leaves a stored manual policy in effect, even for
            # an explicit CREATE. Enable only this missing-index creation;
            # never send it to REBUILD.
            mapped = {
                **(mapped or {}),
                "vector_index_params": {
                    **(mapped or {}).get("vector_index_params", {}),
                    "auto_index": True,
                },
            }
        if mapped is not None:
            params["index_params"] = (
                {**mapped, "index_type": "vector"} if exists else mapped
            )
        return operation, params

    @_with_native_backend_session
    def create_index(
        self, params: Optional[dict], *, ensure: bool = False
    ) -> Optional[dict]:
        mapped = self.index_params(params)
        validate_operation_arguments("create_index", {"index_params": mapped})
        exists = self._vector_index_exists()
        if exists and (ensure or mapped is None):
            return None
        operation, request = self._index_request(mapped, exists)
        return self.transport.call(operation, **request)

    @_with_async_native_backend_session
    async def aquery(self, operation: str, **params: Any) -> dict:
        try:
            return await self.transport.acall(operation, **params)
        except Exception as exc:
            if not _schema.layout_error(exc):
                raise
            await self.transport.amigrate(self.name)
            return await self.transport.acall(operation, **params)

    @_with_async_native_backend_session
    async def adescribe_index(
        self, kind: str = "vector", job_name: Optional[str] = None
    ) -> dict:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(self.describe_index, kind, job_name)
        if job_name is not None:
            raise ValueError(
                "Native VecDB job status is unsupported; "
                "describe the physical index instead"
            )
        return await self.transport.adescribe_index(self.name, kind)

    @_with_async_native_backend_session
    async def acreate_index(self, params: Optional[dict]) -> Optional[dict]:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(self.create_index, params)
        mapped = self.index_params(params)
        validate_operation_arguments("create_index", {"index_params": mapped})
        exists = await self._avector_index_exists()
        if exists and mapped is None:
            return None
        operation, request = self._index_request(mapped, exists)
        return await self.transport.acall(operation, **request)

    @_with_async_native_backend_session
    async def acreate_text_index(self, index: str) -> None:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(self.create_text_index, index)
        await self.transport.acreate_text_index(self.name, index)
        return

    @_with_async_native_backend_session
    async def aprepare(
        self, *, columns: Optional[set[str]] = None, description: Optional[dict] = None
    ) -> None:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(
                self.prepare, columns=columns, description=description
            )
        if columns is None:
            description = description if description is not None else {}
            async with _db._aget_connection(self.transport.client) as connection:
                layout: set[str] = set()
                await _schema.anative_selection(
                    connection,
                    self._selection_name,
                    "vecdb",
                    description=description,
                    layout=layout,
                )
            columns = layout if description else set()
        created = not columns
        if created:
            await self.transport.acall(
                "create_vector_table", **self._table_creation_params()
            )
            columns = _schema._default_columns(self.transport.version)
        elif self.managed_embeddings:
            self._validate_embedding_config(
                description
                if description
                else await self.transport.acall("describe_vector_table", name=self.name)
            )
        self.transport._remember_table(self.name, _schema._table_info(columns, None))
        if not created and self._constructor_index_requested:
            if not await self._avector_index_exists():
                _, request = self._index_request(self._creation_index_params, False)
                validate_operation_arguments("create_index", request)
                await self.transport.acall("create_index", **request)
        logger.info(
            "Table %s %s...",
            self.quoted,
            "created successfully" if created else "already exists",
        )
        return

    @_with_async_native_backend_session
    async def aadd(
        self,
        texts: list[str],
        vectors: Optional[list[list[float]]],
        metadatas: list[dict],
        ids: list[str],
        mutate: bool,
    ) -> list[str]:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(
                self.add, texts, vectors, metadatas, ids, mutate
            )
        records = self.records(texts, vectors, metadatas, ids)
        if not records:
            return []
        if not mutate:
            if vectors is None:
                raise ValueError("Managed VecDB embeddings require upsert writes")
            inserted = await self.transport.ainsert(self.name, records)
            logical_ids = {record["id"]: id_ for record, id_ in zip(records, ids)}
            return [logical_ids[id_] for id_ in inserted]
        await self.transport.acall(
            "upsert_vectors", table_name=self.name, vectors=records
        )
        return ids

    @_with_async_native_backend_session
    async def adelete(self, ids: Optional[list[str]]) -> None:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(self.delete, ids)
        await self.transport.acall(
            "delete_vectors",
            table_name=self.name,
            ids=None if ids is None else [storage_id(id_) for id_ in ids],
        )
        return

    @_with_async_native_backend_session
    async def aget(self, ids: list[str]) -> list[Document]:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(self.get, ids)
        result: list[Document] = []
        for offset in range(0, len(ids), 100):
            batch = [storage_id(id_) for id_ in ids[offset : offset + 100]]
            response = await self.aquery(
                "list_vectors",
                table_name=self.name,
                ids=batch,
                limit=len(batch),
                offset=0,
            )
            result.extend(document(row) for row in rows(response, "vectors"))
        return result

    @_with_async_native_backend_session
    async def asearch(
        self,
        embedding: list[float],
        k: int,
        filter: Optional[dict],
        include_vectors: bool = False,
        **kwargs: Any,
    ) -> list:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(
                self.search, embedding, k, filter, include_vectors, **kwargs
            )
        request, fallback = self._search_request(
            embedding, k, filter, include_vectors, kwargs
        )
        if fallback:
            # Only a filter requiring SQL translation can select this path.
            assert filter is not None
            response = await self.transport.afilter_search(self.name, filter, request)
        else:
            response = await self.aquery("search", **request)
        return self.search_results(response, include_vectors)

    @_with_async_native_backend_session
    async def ahybrid_search(self, params: dict, scores: bool) -> list[Document]:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(self.hybrid_search, params, scores)
        return await self.transport.ahybrid_search(self.name, params, scores)

    @_with_async_native_backend_session
    async def atext_search(
        self, query: str, k: int, scores: bool = False
    ) -> list[Document]:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(self.text_search, query, k, scores)
        return await self.transport.atext_search(self.name, query, k, scores)
