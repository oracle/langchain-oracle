# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Shared ONNX model loading for embedding and reranking models."""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Dict, Optional

import oracledb

logger = logging.getLogger(__name__)


def load_onnx_model(
    conn: Any,
    dir: Optional[str],
    onnx_file: Optional[str],
    model_name: Optional[str],
    *,
    url: Optional[str],
    model_params: Optional[Dict[str, Any]],
    directory_metadata: Dict[str, Any],
    url_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Load once through the selected transport without retries or fallback."""
    cursor = None
    try:
        if conn is None or not model_name:
            raise ValueError("Invalid input: conn and model_name are required")
        if url is not None:
            if not url or dir is not None or onnx_file is not None:
                raise ValueError(
                    "Invalid input: supply either url or dir and onnx_file"
                )
            if model_params is not None and not isinstance(model_params, dict):
                raise ValueError("Invalid input: model_params must be a dictionary")
        elif not dir or not onnx_file or model_params is not None:
            raise ValueError(
                "Invalid input: dir and onnx_file are required without url; "
                "model_params is supported only with url"
            )

        if url is not None and url_metadata is not None:
            model_params = dict(model_params or {})
            model_params.setdefault("metadata", url_metadata)

        from .vectorstores._vecdb import RestTransport, is_sdk_client

        if is_sdk_client(conn):
            if url is None:
                raise ValueError(
                    "REST SDK model loading requires url; "
                    "directory loading requires an oracledb connection"
                )
            RestTransport(conn)
            conn.load_model(model_name=model_name, url=url, model_params=model_params)
            return

        if url is not None:
            params_json = json.dumps(model_params) if model_params is not None else None
            cursor = conn.cursor()
            cursor.execute(
                """
                begin
                    :result := DBMS_VECTOR_DATABASE.LOAD_MODEL(
                        model_name => :model,
                        url => :url,
                        model_params => json(:model_params));
                end;""",
                result=cursor.var(oracledb.DB_TYPE_CLOB),
                model=model_name,
                url=url,
                model_params=params_json,
            )
            cursor.close()
            return

        cursor = conn.cursor()
        cursor.execute(
            """
            begin
                sys.dbms_data_mining.drop_model(
                    model_name => :model, force => true);
                SYS.DBMS_VECTOR.load_onnx_model(:path, :filename, :model,
                    json(:metadata));
            end;""",
            path=dir,
            filename=onnx_file,
            model=model_name,
            metadata=json.dumps(directory_metadata),
        )

        cursor.close()

    except Exception as ex:
        logger.info(f"An exception occurred :: {ex}")
        traceback.print_exc()
        if cursor is not None:
            cursor.close()
        raise
