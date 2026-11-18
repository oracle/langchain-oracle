# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Extracted SDK ORDS SQL transport; response models remain adapter-owned.

Uses the installed SDK ApiClient for authentication, TLS, and HTTP errors.
The SQL base URL can target standard REST-enabled SQL; the success response
type is object, so this works with SDK
versions that predate the generated SQL response models.
"""

# Source commit: cace12d498e98b2e3d2dfd9b14bf0662f46838da
from typing import Any, Dict, List, Optional, Tuple, Union

from oracle_vecdb.services.ords.api_client import ApiClient, RequestSerialized
from pydantic import Field, StrictFloat, StrictInt, StrictStr, validate_call
from typing_extensions import Annotated

from .models.sql_request import SqlRequest


class SqlApi:
    def __init__(self, api_client=None, *, sql_base_url=None) -> None:
        if api_client is None:
            api_client = ApiClient.get_default()
        self.api_client = api_client
        self.sql_base_url = sql_base_url

    @validate_call
    def execute_sql(
        self,
        sql_request: SqlRequest,
        _request_timeout: Union[
            None,
            Annotated[StrictFloat, Field(gt=0)],
            Tuple[
                Annotated[StrictFloat, Field(gt=0)],
                Annotated[StrictFloat, Field(gt=0)],
            ],
        ] = None,
        _request_auth: Optional[Dict[StrictStr, Any]] = None,
        _content_type: Optional[StrictStr] = None,
        _headers: Optional[Dict[StrictStr, Any]] = None,
        _host_index: Annotated[StrictInt, Field(ge=0, le=0)] = 0,
    ) -> Any:
        """Execute SQL

        Executes one or more SQL statements using ORDS REST-Enabled SQL.

        :param sql_request: (required)
        :type sql_request: SqlRequest
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :type _request_timeout: int, tuple(int, int), optional
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the
                              authentication in the spec for a single request.
        :type _request_auth: dict, optional
        :param _content_type: force content-type for the request.
        :type _content_type: str, Optional
        :param _headers: set to override the headers for a single
                         request; this effectively ignores the headers
                         in the spec for a single request.
        :type _headers: dict, optional
        :param _host_index: set to override the host_index for a single
                            request; this effectively ignores the host_index
                            in the spec for a single request.
        :type _host_index: int, optional
        :return: Returns the result object.
        """  # noqa: E501

        _param = self._execute_sql_serialize(
            sql_request=sql_request,
            _request_auth=_request_auth,
            _content_type=_content_type,
            _headers=_headers,
            _host_index=_host_index,
        )

        _response_types_map: Dict[str, Optional[str]] = {
            "200": "object",
            "400": "ORDSErrorResponse",
            "401": "ORDSErrorResponse",
            "403": "ORDSErrorResponse",
            "500": "ORDSErrorResponse",
        }
        response_data = self.api_client.call_api(
            *_param, _request_timeout=_request_timeout
        )
        response_data.read()
        return self.api_client.response_deserialize(
            response_data=response_data,
            response_types_map=_response_types_map,
        ).data

    def _execute_sql_serialize(
        self,
        sql_request,
        _request_auth,
        _content_type,
        _headers,
        _host_index,
    ) -> RequestSerialized:
        _host = self.sql_base_url

        _collection_formats: Dict[str, str] = {}

        _path_params: Dict[str, str] = {}
        _query_params: List[Tuple[str, str]] = []
        _header_params: Dict[str, Optional[str]] = _headers or {}
        _form_params: List[Tuple[str, str]] = []
        _files: Dict[
            str,
            Union[str, bytes, List[str], List[bytes], List[Tuple[str, bytes]]],
        ] = {}
        _body_params: Optional[bytes] = None

        # process the path parameters
        # process the query parameters
        # process the header parameters
        # process the form parameters
        # process the body parameter
        if sql_request is not None:
            _body_params = sql_request

        # set the HTTP header `Accept`
        if "Accept" not in _header_params:
            _header_params["Accept"] = self.api_client.select_header_accept(
                ["application/json"]
            )

        # set the HTTP header `Content-Type`
        if _content_type:
            _header_params["Content-Type"] = _content_type
        else:
            _default_content_type = self.api_client.select_header_content_type(
                ["application/json", "application/sql"]
            )
            if _default_content_type is not None:
                _header_params["Content-Type"] = _default_content_type

        # authentication setting
        _auth_settings: List[str] = [
            "OAuth2",
            "BasicAuth",
            "BearerAuth",
        ]

        return self.api_client.param_serialize(
            method="POST",
            resource_path="/sql",
            path_params=_path_params,
            query_params=_query_params,
            header_params=_header_params,
            body=_body_params,
            post_params=_form_params,
            files=_files,
            auth_settings=_auth_settings,
            collection_formats=_collection_formats,
            _host=_host,
            _request_auth=_request_auth,
        )
