# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
oracleai.py

Implements OracleEmbeddings for generating and handling
vector embeddings with Oracle AI Vector Search.

Authors:
    - Harichandan Roy (hroy)
    - David Jiang (ddjiang)
"""

from __future__ import annotations

import json
import logging
import math
import traceback
from typing import Any, Dict, List, Optional

import oracledb
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict

from .._telemetry import tag_sql
from ..vectorstores._vecdb import RestTransport, is_sdk_client
from ..vectorstores._vecdb_adapter import as_dict

logger = logging.getLogger(__name__)

"""OracleEmbeddings class"""


class OracleEmbeddings(BaseModel, Embeddings):
    """Generate embeddings through native Oracle SQL or a REST VecDB SDK client.

    REST clients use a loaded database model named by params['model']; provider
    defaults to database. REST supports only provider/model parameters and does
    not use the native session proxy setting. Clients remain caller-owned.
    """

    # Oracle connection
    conn: Any = None
    # embedding parameters
    params: Dict[str, Any]
    # proxy
    proxy: Optional[str] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    model_config = ConfigDict(
        extra="forbid",
    )

    """
    1 - user needs to have create procedure, 
        create mining model, create any directory privilege.
    2 - grant create procedure, create mining model, 
        create any directory to <user>;
    """

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
        """Load an ONNX model to Oracle Database.

        Supply either ``dir`` and ``onnx_file``, or ``url``. Directory loading
        uses DBMS_VECTOR.LOAD_ONNX_MODEL and replaces an existing model. URL
        loading uses DBMS_VECTOR_DATABASE.LOAD_MODEL and requires a new model
        name. The database downloads the URL; this method does not upload files.
        Database errors, including an unavailable LOAD_MODEL API or insufficient
        privileges, propagate without a version check or fallback.
        REST SDK clients support URL loading through their public ``load_model``
        method and remain caller-owned. Directory loading requires a database
        connection. This helper returns None for both transports.

        Args:
            conn: Oracle Connection or REST-configured oracle_vecdb.OracleVecDB client,
            dir: Oracle Directory,
            onnx_file: ONNX file name,
            model_name: Name of the model.
            url: Object storage URL of the ONNX model, instead of a directory.
            model_params: URL-only settings, including the database cloud
                ``credential`` name and ONNX ``metadata`` object.
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
                "function": "embedding",
                "embeddingOutput": "embedding",
                "input": {"input": ["DATA"]},
            },
        )

    def _embed_rest_documents(self, texts: List[str]) -> List[List[float]]:
        RestTransport(self.conn)
        if self.params.get("provider", "database") != "database":
            raise ValueError("REST embeddings require provider='database'")
        model = self.params.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("REST embeddings require a nonempty model")
        if set(self.params) - {"provider", "model"}:
            raise ValueError(
                "REST embeddings support only provider and model parameters"
            )
        if self.proxy:
            raise ValueError("proxy is supported only for native database embeddings")

        response = as_dict(self.conn.generate_embedding(model_name=model, inputs=texts))
        rows = response.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ValueError(
                "Malformed REST embedding response: incorrect result count"
            )
        embeddings = []
        dimension = None
        for text, item in zip(texts, rows):
            row = as_dict(item)
            if row.get("text") is not None and row["text"] != text:
                raise ValueError(
                    "Malformed REST embedding response: text order mismatch"
                )
            vector = row.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ValueError("Malformed REST embedding response: missing vector")
            try:
                if any(isinstance(value, bool) for value in vector):
                    raise ValueError("boolean vector value")
                values = [float(value) for value in vector]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Malformed REST embedding response: invalid vector"
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    "Malformed REST embedding response: vector must be finite"
                )
            if dimension is not None and len(values) != dimension:
                raise ValueError(
                    "Malformed REST embedding response: inconsistent dimensions"
                )
            dimension = len(values)
            embeddings.append(values)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using an OracleEmbeddings.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each input text.
        """
        if not texts:
            return []

        if is_sdk_client(self.conn):
            return self._embed_rest_documents(texts)

        embeddings: List[List[float]] = []
        cursor = None
        try:
            # returns strings or bytes instead of a locator
            oracledb.defaults.fetch_lobs = False
            cursor = self.conn.cursor()
            proxy_was_set = False

            if self.proxy:
                cursor.execute(
                    "begin utl_http.set_proxy(:proxy); end;", proxy=self.proxy
                )
                proxy_was_set = True

            try:
                chunks = []
                for i, text in enumerate(texts, start=1):
                    chunk = {"chunk_id": i, "chunk_data": text}
                    chunks.append(json.dumps(chunk))

                vector_array_type = self.conn.gettype("SYS.VECTOR_ARRAY_T")
                # Newer databases define VECTOR_ARRAY_T as a CLOB collection.
                # Keep the string collection path for earlier definitions.
                inputs_data = (
                    [
                        self.conn.createlob(oracledb.DB_TYPE_CLOB, chunk)
                        for chunk in chunks
                    ]
                    if vector_array_type.element_type == oracledb.DB_TYPE_CLOB
                    else chunks
                )
                inputs = vector_array_type.newobject(inputs_data)
                cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)
                cursor.execute(
                    tag_sql(
                        "select t.* "
                        + "from dbms_vector_chain.utl_to_embeddings(:1, "
                        + "json(:2)) t",
                        "embedding",
                    ),
                    [inputs, self.params],
                )

                for row in cursor:
                    if row is None:
                        embeddings.append([])
                    else:
                        rdata = json.loads(row[0])
                        # dereference string as array
                        vec = json.loads(rdata["embed_vector"])
                        embeddings.append(vec)
            except BaseException:
                if proxy_was_set:
                    try:
                        cursor.execute(
                            "begin utl_http.set_proxy(:proxy); end;", proxy=None
                        )
                    except Exception:
                        logger.exception(
                            "Failed to clear Oracle session proxy after "
                            "embed_documents failed"
                        )
                raise
            else:
                if proxy_was_set:
                    try:
                        cursor.execute(
                            "begin utl_http.set_proxy(:proxy); end;", proxy=None
                        )
                    except Exception:
                        logger.warning(
                            "Failed to clear Oracle session proxy after "
                            "embed_documents succeeded",
                            exc_info=True,
                        )

            return embeddings
        except Exception as ex:
            logger.info(f"An exception occurred :: {ex}")
            traceback.print_exc()
            raise
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    logger.exception("Failed to close Oracle embedding cursor")

    def embed_query(self, text: str) -> List[float]:
        """Compute query embedding using an OracleEmbeddings.
        Args:
            text: The text to embed.
        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]


# uncomment the following code block to run the test

"""
# A sample unit test.

import os
import oracledb
# get the Oracle connection
conn = oracledb.connect(
    dsn=os.environ["VECDB_HOST"]
)
print("Oracle connection is established...")

# params 
embedder_params = {"provider": "database", "model": "demo_model"}
proxy = ""

# instance
embedder = OracleEmbeddings(conn=conn, params=embedder_params, proxy=proxy)

docs = ["hello world!", "hi everyone!", "greetings!"]
embeds = embedder.embed_documents(docs)
print(f"Total Embeddings: {len(embeds)}")
print(f"Embedding generated by OracleEmbeddings: {embeds[0]}\n")

embed = embedder.embed_query("Hello World!")
print(f"Embedding generated by OracleEmbeddings: {embed}")

conn.close()
print("Connection is closed.")

"""
