# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Run the shared hybrid SQL plans through SDK-authenticated ORDS requests.

This is a narrow cursor interface for the hybrid/index plans, not a general
DB-API connection. Each request can use a different database session. No
caller transaction, connection ownership, or session state is assumed.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any, Optional


class ORDSSQLConnection:
    transaction_in_progress = False

    def __init__(self, client: Any):
        self.sdk_client = client

    def cursor(self) -> ORDSSQLCursor:
        return ORDSSQLCursor(self.sdk_client)


class ORDSSQLCursor:
    def __init__(self, client: Any):
        self.client = client
        self._rows: list[tuple] = []
        self._clob_binds: set[str] = set()

    def __enter__(self) -> ORDSSQLCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    description: Any = None

    def setinputsizes(self, **sizes: Any) -> None:
        # Shared hybrid SQL explicitly marks its JSON request as CLOB.
        self._clob_binds = set(sizes)

    def execute(self, statement: str, parameters: Any = None, **kwargs: Any) -> None:
        from .._vecdb_sdk.services.ords.models.sql_request import SqlRequest
        from .._vecdb_sdk.services.ords.sql_api import SqlApi

        values = kwargs if parameters is None else parameters
        binds = []
        if isinstance(values, (list, tuple)):
            binds = [
                {
                    "index": index,
                    "value": value,
                    "data_type": "CLOB"
                    if isinstance(value, str) and len(value) > 4000
                    else "VARCHAR2",
                }
                for index, value in enumerate(values, 1)
            ]
        elif values:
            binds = [
                {
                    "name": name,
                    "value": value,
                    "data_type": "CLOB" if name in self._clob_binds else "VARCHAR2",
                }
                for name, value in values.items()
            ]
        request = SqlRequest.model_validate(
            {"statementText": statement, "binds": binds}
        )
        # REST-enabled SQL lives under the schema's /_/sql route, rather
        # than the database-management API prefix used by VecDB operations.
        base = self.client.config.rest_url.rstrip("/")
        schema_url, marker, _ = base.partition("/_/db-api/")
        if not marker:
            raise ValueError("Cannot resolve the schema ORDS SQL endpoint")
        response = SqlApi(
            self.client.api_client, sql_base_url=schema_url + "/_"
        ).execute_sql(
            request,
            _content_type="application/json",
            _request_timeout=getattr(self.client.config, "timeout", None),
        )
        error = self._find_sql_error(response)
        if error is not None:
            raise RuntimeError(f"ORDS SQL execution failed: {error}")
        if not isinstance(response, dict) or not isinstance(
            response.get("items"), list
        ):
            raise ValueError("Malformed ORDS SQL response: missing statement results")
        self._rows = []
        for item in response["items"]:
            result = item.get("resultSet")
            if result is None:
                continue
            if result.get("hasMore") or item.get("hasMore"):
                raise ValueError(
                    "ORDS SQL result was truncated; reduce the requested result count"
                )
            metadata = result["metadata"]
            columns = [
                (column.get("jsonColumnName") or column["columnName"]).lower()
                for column in metadata
            ]
            self.description = [(column["columnName"],) for column in metadata]
            for row in result["items"]:
                normalized = {key.lower(): value for key, value in row.items()}
                values = []
                for name in columns:
                    value = normalized[name]
                    if name in {"metadata", "content_metadata"} and isinstance(
                        value, str
                    ):
                        value = json.loads(
                            value, parse_int=Decimal, parse_float=Decimal
                        )
                    values.append(value)
                self._rows.append(tuple(values))

    def fetchone(self) -> Optional[tuple]:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[tuple]:
        result, self._rows = self._rows, []
        return result

    # Adapted from SDK cace12d, HybridSearch._find_sql_error.
    @classmethod
    def _find_sql_error(cls, value: Any) -> Optional[Any]:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = str(key).lower()
                if normalized_key in {"resultset", "result_set"}:
                    continue
                if normalized_key in {
                    "error",
                    "errors",
                    "errorcode",
                    "error_code",
                    "errormessage",
                    "error_message",
                }:
                    if item:
                        return {key: item}
                found = cls._find_sql_error(item)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = cls._find_sql_error(item)
                if found is not None:
                    return found
        elif isinstance(value, str):
            if re.search(
                r"\bORA-\d{5}\b|ERROR at line|Error report -|^SP2-",
                value,
                re.IGNORECASE,
            ):
                return value.strip()
        return None
