# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
oracleadb_loader.py

Contains OracleAutonomousDatabaseLoader for connecting to 
Oracle Autonomous Database (ADB).
"""

from typing import Any, Dict, List, Optional, Union

import oracledb
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class OracleAutonomousDatabaseLoader(BaseLoader):
    """
    Load from Oracle ADB

    Autonomous Database connection can be made by connection_string
    or tns alias. wallet_location and wallet_password are required
    for TLS connection.
    Each document will represent one row of the query result.
    Columns are written into the `page_content` and 'metadata' in
    constructor is written into 'metadata' of document,
    by default, the 'metadata' is None.
    """

    def __init__(
        self,
        query: str,
        user: str,
        password: str,
        *,
        schema: Optional[str] = None,
        dsn: Optional[str] = None,
        config_dir: Optional[str] = None,
        wallet_location: Optional[str] = None,
        wallet_password: Optional[str] = None,
        metadata: Optional[List[str]] = None,
        parameter: Optional[Union[list, tuple, dict]] = None,
    ):
        """
        init method
        :param query: sql query to execute
        :param user: username
        :param password: user password
        :param schema: schema to run in database
        :param dsn: data source name used to identify the oracle database to connect to
        :param config_dir: directory of config files(tnsname.ora, ewallet.pem)
        :param wallet_location: location of ewallet.pem, not required for TLS connections
        :param wallet_password: password of wallet, not required for TLS connections
        :param metadata: metadata used in document
        :param parameter: bind variable to use in query
        """  # noqa: E501
        # Mandatory required arguments.
        self.query = query
        self.user = user
        self.password = password

        # Schema
        self.schema = schema

        # TNS connection Method
        self.config_dir = config_dir

        # Wallet configuration is required for mTLS connection
        self.wallet_location = wallet_location
        self.wallet_password = wallet_password

        # metadata column
        self.metadata = metadata

        # parameter, e.g bind variable
        self.parameter = parameter

        # dsn
        self.dsn = dsn

    def _run_query(self) -> List[Dict[str, Any]]:
        connect_param = {"user": self.user, "password": self.password, "dsn": self.dsn}
        if self.config_dir:
            connect_param["config_dir"] = self.config_dir
        if self.wallet_location and self.wallet_password:
            connect_param["wallet_location"] = self.wallet_location
            connect_param["wallet_password"] = self.wallet_password

        try:
            connection = oracledb.connect(**connect_param)
            cursor = connection.cursor()
            if self.schema:
                connection.current_schema = self.schema
            if self.parameter:
                cursor.execute(self.query, self.parameter)
            else:
                cursor.execute(self.query)
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            data = [
                {
                    i: (j if not isinstance(j, oracledb.LOB) else j.read())
                    for i, j in zip(columns, row)
                }
                for row in data
            ]
        except oracledb.DatabaseError as e:
            print("Got error while connecting: " + str(e))  # noqa: T201
            data = []
        finally:
            cursor.close()
            connection.close()

        return data

    def load(self) -> List[Document]:
        data = self._run_query()
        documents = []
        metadata_columns = self.metadata if self.metadata else []
        for row in data:
            metadata = {
                key: value for key, value in row.items() if key in metadata_columns
            }
            doc = Document(page_content=str(row), metadata=metadata)
            documents.append(doc)

        return documents
