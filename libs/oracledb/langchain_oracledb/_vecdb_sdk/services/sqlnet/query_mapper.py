##
## Copyright (c) 2026 Oracle and/or its affiliates.
## Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
##

"""SQL*Net DBMS_VECTOR_DATABASE query mappings."""

from __future__ import annotations

from typing import Any, Dict


class SQLNetQueryMapper:
    """Return SQL/PLSQL statements for SQL*Net VecDB operations."""

    _QUERIES: Dict[str, str] = {
        "describe_vector_table": """
            SELECT dbms_vector_database.describe_vector_table(:table_name)
            FROM dual
        """,
        "list_vectors": """
            SELECT dbms_vector_database.list_vectors(
                table_name => :table_name,
                ids => CASE
                    WHEN :ids_json IS NOT NULL
                    THEN JSON(:ids_json)
                    ELSE NULL
                END,
                limit => :limit,
                offset => :offset,
                debug_flags => CASE
                    WHEN :debug_flags_json IS NOT NULL
                    THEN JSON(:debug_flags_json)
                    ELSE NULL
                END
            )
            FROM dual
        """,
        "query": """
            SELECT dbms_vector_database.search(
                table_name => :table_name,
                query_by => JSON(:query_by_json),
                filters => CASE
                    WHEN :filters_json IS NOT NULL
                    THEN JSON(:filters_json)
                    ELSE NULL
                END,
                top_k => :top_k,
                include_vectors => :include_vectors,
                output_selector => NULL,
                advanced_options => CASE
                    WHEN :advanced_options_json IS NOT NULL
                    THEN JSON(:advanced_options_json)
                    ELSE NULL
                END,
                debug_flags => CASE
                    WHEN :debug_flags_json IS NOT NULL
                    THEN JSON(:debug_flags_json)
                    ELSE NULL
                END
            )
            FROM dual
        """,
        "query_with_output_selector": """
            SELECT dbms_vector_database.search(
                table_name => :table_name,
                query_by => JSON(:query_by_json),
                filters => CASE
                    WHEN :filters_json IS NOT NULL
                    THEN JSON(:filters_json)
                    ELSE NULL
                END,
                top_k => :top_k,
                include_vectors => :include_vectors,
                output_selector => JSON(:output_selector_json),
                advanced_options => CASE
                    WHEN :advanced_options_json IS NOT NULL
                    THEN JSON(:advanced_options_json)
                    ELSE NULL
                END,
                debug_flags => CASE
                    WHEN :debug_flags_json IS NOT NULL
                    THEN JSON(:debug_flags_json)
                    ELSE NULL
                END
            )
            FROM dual
        """,
        "delete_vectors": """
            SELECT dbms_vector_database.delete_vectors(
                table_name => :table_name,
                ids => JSON(:ids_json),
                debug_flags => CASE
                    WHEN :debug_flags_json IS NOT NULL
                    THEN JSON(:debug_flags_json)
                    ELSE NULL
                END
            )
            FROM dual
        """,
        "create_index": """
            DECLARE
                l_output CLOB;
            BEGIN
                l_output := dbms_vector_database.create_index(
                    table_name => :table_name,
                    index_params => CASE
                        WHEN :index_params_json IS NOT NULL
                        THEN JSON(:index_params_json)
                        ELSE NULL
                    END,
                    debug_flags => CASE
                        WHEN :debug_flags_json IS NOT NULL
                        THEN JSON(:debug_flags_json)
                        ELSE NULL
                    END
                );
            END;
        """,
    }

    @classmethod
    def get_query(cls, name: str, **format_values: Any) -> str:
        try:
            query = cls._QUERIES[name]
        except KeyError as exc:
            raise KeyError(f"No SQLNet query mapping for {name!r}") from exc
        if format_values:
            query = query.format(**format_values)
        return " ".join(query.split())
