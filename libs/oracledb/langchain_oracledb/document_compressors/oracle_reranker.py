# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Rerank LangChain documents with Oracle database or REST inference."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence

import oracledb
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, Field, model_validator

from .._telemetry import tag_sql
from ..vectorstores._vecdb import RestTransport, is_sdk_client
from ..vectorstores.utils import _get_connection


def _ranked_indices(response: Any, count: int) -> List[int]:
    """Validate Oracle's zero-based indices before mapping to original documents."""
    if hasattr(response, "read"):
        response = response.read()
    if isinstance(response, (str, bytes)):
        response = json.loads(response)
    if hasattr(response, "to_dict"):
        response = response.to_dict()
    if isinstance(response, dict):
        response = response.get("items")
    if not isinstance(response, list):
        raise ValueError("Malformed Oracle rerank response: expected a result array")

    scores = {}
    for row in response:
        if not isinstance(row, dict):
            raise ValueError(
                "Malformed Oracle rerank response: expected result objects"
            )
        index = row.get("index")
        if isinstance(index, str) and re.fullmatch(r"[0-9]+", index):
            index = int(index)
        if type(index) is not int or not 0 <= index < count or index in scores:
            raise ValueError(
                "Malformed Oracle rerank response: invalid or duplicate index"
            )
        score = row.get("score")
        if score is None or isinstance(score, bool):
            raise ValueError("Malformed Oracle rerank response: invalid score")
        try:
            score = float(score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Malformed Oracle rerank response: invalid score") from exc
        if not math.isfinite(score):
            raise ValueError("Malformed Oracle rerank response: score must be finite")
        scores[index] = score

    # Original input order breaks ties consistently across native and REST results.
    return sorted(scores, key=lambda index: (-scores[index], index))


class OracleReranker(BaseDocumentCompressor):
    """Select the most relevant documents using an Oracle reranking model.

    Native connections and pools call DBMS_VECTOR_CHAIN.UTL_TO_RERANK. Provider
    parameters follow that API, including ``provider`` and ``model``. REST SDK
    clients call their public ``rerank`` method with a loaded database model.
    API errors propagate without version checks or fallback. Connections, pools,
    and SDK clients remain caller-owned.

    Returned documents retain their original content, IDs, and metadata. Scores
    are used for ordering only. The inherited async API runs synchronous work in
    an executor; it requires a synchronous connection, pool, or REST SDK client.
    """

    conn: Any = Field(exclude=True, repr=False)
    params: Dict[str, Any]
    top_n: int = Field(default=3, gt=0, strict=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @staticmethod
    def load_onnx_model(
        conn: Any,
        dir: Optional[str] = None,
        onnx_file: Optional[str] = None,
        model_name: Optional[str] = None,
        *,
        url: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load an ONNX reranker from a database directory or object storage URL.

        Accepts the same connection and source arguments as
        OracleEmbeddings.load_onnx_model. URL loading uses the database API or
        REST SDK load_model and defaults model metadata to function=regression;
        model_params can supply custom metadata and a cloud credential name.
        Directory loading replaces an existing model and expects ONNX inputs
        first_input/second_input, database attributes FIRST_INPUT/SECOND_INPUT,
        and regression output output. URL loading requires a new model name.
        Clients remain caller-owned and errors propagate without fallback.
        """
        from .._onnx import load_onnx_model

        load_onnx_model(
            conn,
            dir,
            onnx_file,
            model_name,
            url=url,
            model_params=model_params,
            directory_metadata={
                "function": "regression",
                "regressionOutput": "output",
                "input": {
                    "first_input": ["FIRST_INPUT"],
                    "second_input": ["SECOND_INPUT"],
                },
            },
            url_metadata={"function": "regression"},
        )

    @model_validator(mode="after")
    def _validate_parameters(self) -> OracleReranker:
        for name in ("provider", "model"):
            if (
                not isinstance(self.params.get(name), str)
                or not self.params[name].strip()
            ):
                raise ValueError(f"params must contain a nonempty {name}")
        if self.conn is None:
            raise ValueError("conn is required")
        if is_sdk_client(self.conn):
            RestTransport(self.conn)
            if self.params["provider"] != "database":
                raise ValueError("REST reranking requires provider='database'")
        return self

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,
    ) -> Sequence[Document]:
        """Return up to top_n documents in descending relevance order."""
        if not documents:
            return []
        texts = [document.page_content for document in documents]
        params = dict(self.params)
        # top_n is authoritative even when provider parameters specify a limit.
        params.pop("top_n", None)
        params.pop("topN", None)
        limit_name = (
            "topN" if params["provider"] in {"vertexai", "ocigenai"} else "top_n"
        )
        # Fetch all candidate scores so ties at the cutoff use original input order.
        params[limit_name] = len(documents)

        if is_sdk_client(self.conn):
            model = params.pop("model")
            # The REST SDK selects the database provider through model_name.
            params.pop("provider")
            response = self.conn.rerank(
                query=query, documents=texts, model_name=model, model_params=params
            )
        else:
            with (
                _get_connection(self.conn) as connection,
                connection.cursor() as cursor,
            ):
                cursor.setinputsizes(
                    query=oracledb.DB_TYPE_CLOB,
                    documents=oracledb.DB_TYPE_CLOB,
                    params=oracledb.DB_TYPE_CLOB,
                )
                cursor.execute(
                    tag_sql(
                        "SELECT JSON_SERIALIZE(DBMS_VECTOR_CHAIN.UTL_TO_RERANK("
                        ":query, JSON(:documents), JSON(:params)) RETURNING CLOB) "
                        "FROM dual",
                        "rerank",
                    ),
                    query=query,
                    documents=json.dumps({"documents": texts}),
                    params=json.dumps(params),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError("Malformed Oracle rerank response: missing result")
                response = row[0]
                # Read LOBs while their cursor and pool checkout remain alive.
                if hasattr(response, "read"):
                    response = response.read()

        indices = _ranked_indices(response, len(documents))
        return [documents[index] for index in indices[: self.top_n]]
