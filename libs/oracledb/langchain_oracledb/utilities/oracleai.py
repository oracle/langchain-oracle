# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
oracleai.py

Implements OracleSummary with Oracle AI Vector Search support.

Authors:
    - Harichandan Roy (hroy)
    - David Jiang (ddjiang)
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.documents import Document

if TYPE_CHECKING:
    from oracledb import Connection

import oracledb

logger = logging.getLogger(__name__)

"""OracleSummary class"""


def output_type_handler(cursor: Any, metadata: Any) -> Any:
    if metadata.type_code is oracledb.DB_TYPE_CLOB:
        return cursor.var(oracledb.DB_TYPE_LONG, arraysize=cursor.arraysize)
    if metadata.type_code is oracledb.DB_TYPE_BLOB:
        return cursor.var(oracledb.DB_TYPE_LONG_RAW, arraysize=cursor.arraysize)
    if metadata.type_code is oracledb.DB_TYPE_NCLOB:
        return cursor.var(oracledb.DB_TYPE_LONG_NVARCHAR, arraysize=cursor.arraysize)


class OracleSummary:
    """Get Summary
    Args:
        conn: Oracle Connection,
        params: Summary parameters,
        proxy: Proxy
    """

    def __init__(
        self, conn: Connection, params: Dict[str, Any], proxy: Optional[str] = None
    ):
        self.conn = conn
        self.proxy = proxy
        self.summary_params = params

    def get_summary(self, docs: Any) -> List[str]:
        """Get the summary of the input docs.
        Args:
            docs: The documents to generate summary for.
                  Allowed input types: str, Document, List[str], List[Document]
        Returns:
            List of summary text, one for each input doc.
        """

        if docs is None:
            return []
        if isinstance(docs, list) and not docs:
            return []

        results: List[str] = []
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.outputtypehandler = output_type_handler
            proxy_was_set = False

            if self.proxy:
                cursor.execute(
                    "begin utl_http.set_proxy(:proxy); end;", proxy=self.proxy
                )
                proxy_was_set = True

            try:
                if isinstance(docs, str):
                    results = []

                    summary = cursor.var(oracledb.DB_TYPE_CLOB)
                    cursor.execute(
                        """
                        declare
                            input clob;
                        begin
                            input := :data;
                            :summ := dbms_vector_chain.utl_to_summary(
                                input, json(:params));
                        end;""",
                        data=docs,
                        params=json.dumps(self.summary_params),
                        summ=summary,
                    )

                    if summary.getvalue() is None:
                        results.append("")
                    else:
                        results.append(str(summary.getvalue()))

                elif isinstance(docs, Document):
                    results = []

                    summary = cursor.var(oracledb.DB_TYPE_CLOB)
                    cursor.execute(
                        """
                        declare
                            input clob;
                        begin
                            input := :data;
                            :summ := dbms_vector_chain.utl_to_summary(
                                input, json(:params));
                        end;""",
                        data=docs.page_content,
                        params=json.dumps(self.summary_params),
                        summ=summary,
                    )

                    if summary.getvalue() is None:
                        results.append("")
                    else:
                        results.append(str(summary.getvalue()))

                elif isinstance(docs, List):
                    docs_input = []
                    params = json.dumps(self.summary_params)
                    summary = cursor.var(oracledb.DB_TYPE_CLOB, arraysize=len(docs))

                    for doc in docs:
                        if isinstance(doc, str):
                            docs_input.append((doc, params))
                        elif isinstance(doc, Document):
                            docs_input.append((doc.page_content, params))
                        else:
                            raise Exception("Invalid input type")

                    cursor.setinputsizes(None, None, summary)

                    cursor.executemany(
                        """
                        declare
                            input clob;
                            summ clob;
                        begin
                            input := :1;
                            summ := dbms_vector_chain.utl_to_summary(input, 
                                        json(:2));
                            :3 := summ;
                        end;""",
                        docs_input,
                    )

                    results = [
                        "" if summary.getvalue(i) is None else str(summary.getvalue(i))
                        for i in range(summary.actual_elements)
                    ]

                else:
                    raise Exception("Invalid input type")
            except BaseException:
                if proxy_was_set:
                    try:
                        cursor.execute(
                            "begin utl_http.set_proxy(:proxy); end;", proxy=None
                        )
                    except Exception:
                        logger.exception(
                            "Failed to clear Oracle session proxy after "
                            "get_summary failed"
                        )
                raise
            else:
                if proxy_was_set:
                    cursor.execute("begin utl_http.set_proxy(:proxy); end;", proxy=None)

            return results

        except Exception as ex:
            logger.info(f"An exception occurred :: {ex}")
            traceback.print_exc()
            raise
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    logger.exception("Failed to close Oracle summary cursor")


# uncomment the following code block to run the test

"""
# A sample unit test.

import os

''' get the Oracle connection '''
conn = oracledb.connect(dsn=os.environ["ORACLE_DB_DSN"])
print("Oracle connection is established...")

''' params '''
summary_params = {"provider": "database","glevel": "S",
                  "numParagraphs": 1,"language": "english"} 
proxy = ""

''' instance '''
summ = OracleSummary(conn=conn, params=summary_params, proxy=proxy)

summary = summ.get_summary("In the heart of the forest, " + 
    "a lone fox ventured out at dusk, seeking a lost treasure. " + 
    "With each step, memories flooded back, guiding its path. " + 
    "As the moon rose high, illuminating the night, the fox unearthed " + 
    "not gold, but a forgotten friendship, worth more than any riches.")
print(f"Summary generated by OracleSummary: {summary}")

conn.close()
print("Connection is closed.")

"""
