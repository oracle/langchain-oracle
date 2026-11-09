# langchain-oracledb

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-oracledb?label=%20&logo=pypi&logoColor=white)](https://pypi.org/project/langchain-oracledb/)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-oracledb)](https://pypistats.org/packages/langchain-oracledb)
[![License](https://img.shields.io/badge/license-UPL--1.0-blue)](https://oss.oracle.com/licenses/upl/)

This package contains the LangChain integrations with [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/).

Tables created by LangChain are annotated with `LANGCHAIN_ORACLE` when the database supports annotations.

> ⭐ If this package is useful to you, consider [starring the repository](https://github.com/oracle/langchain-oracle) — it helps other developers discover these integrations.

## Installation

```bash
python -m pip install -U langchain-oracledb
```

## Documentation

- [Oracle AI Vector Search: Vector Store](https://python.langchain.com/docs/integrations/vectorstores/oracle/)
- [Oracle AI Vector Search: Generate Summary](https://python.langchain.com/docs/integrations/tools/oracleai/)
- [Oracle AI Vector Search: Document Processing](https://python.langchain.com/docs/integrations/document_loaders/oracleai/)
- [Oracle AI Vector Search: Generate Embeddings](https://python.langchain.com/docs/integrations/text_embedding/oracleai/)

## Examples

The following examples showcase basic usage of the components provided by `langchain-oracledb`.

Please refer to our complete demo guide [Oracle AI Vector Search End-to-End Demo Guide](https://github.com/langchain-ai/langchain/blob/v0.3/cookbook/oracleai_demo.ipynb) to build an end to end RAG pipeline with the help of Oracle AI Vector Search.

### Connect to Oracle Database

Some examples below require a connection with Oracle Database through [`python-oracledb`](https://pypi.org/project/oracledb/). The following sample code will show how to connect to Oracle Database. By default, `python-oracledb` runs in a ‘Thin’ mode which connects directly to Oracle Database. This mode does not need Oracle Client libraries. However, some additional functionality is available when python-oracledb uses them. Python-oracledb is said to be in ‘Thick’ mode when Oracle Client libraries are used. Both modes have comprehensive functionality supporting the Python Database API v2.0 Specification. See the following [guide](https://python-oracledb.readthedocs.io/en/latest/user_guide/appendix_a.html#featuresummary) that talks about features supported in each mode. You might want to switch to Thick mode if you are unable to use Thin mode. For python-oracledb installation help, see [Installing python-oracledb](https://python-oracledb.readthedocs.io/en/latest/user_guide/installation.html).

Check your database connectivity:

```python
import os

import oracledb

# Please update with your username, password, hostname, port and service_name
username = os.environ["ORACLE_DB_USERNAME"]
password = os.environ["ORACLE_DB_PASSWORD"]
dsn = os.environ["ORACLE_DB_DSN"]

connection = oracledb.connect(user=username, password=password, dsn=dsn)
print("Connection successful!")
```

### Vector Stores

#### OracleVS backends and VecDB

`OracleVS` defaults to `backend="auto"`: existing SQL tables remain SQL-backed,
while package-registered VecDB tables use VecDB. New tables use VecDB when the
server and requested name support it, otherwise SQL. Set `backend="sql"` or
`backend="vecdb"` explicitly, and inspect `store.backend` for the selected backend.

Native ownership checks use a bound, single-table
`DBMS_VECTOR_DATABASE.DESCRIBE_VECTOR_TABLE` call. Older packages that reject
cross-schema description use a server-filtered registry lookup returning only a
membership flag; Python does not download the full registry. The targeted
`ALL_TAB_COLUMNS` query still checks table existence and layout.
For native VecDB selection, one SELECT returns both `SEARCH` availability and
`COMPATIBLE`, using an inline function around `DBMS_UTILITY.DB_VERSION` without
requiring `V$PARAMETER` access. Existing SQL stores skip that compatibility query.

Native VecDB uses an `oracledb` connection or pool and requires Oracle Database
23.26.1+ with `COMPATIBLE >=23.4` (or `>=23.6` for 23.26.3+). Package operations
require unqualified table names. Names are quoted and retain their case, as on
the SQL backend, including lowercase, mixed-case, and quoted names containing
literal dots. The default `from_texts()` name `langchain` can therefore use VecDB
in automatic mode on a supported server.

REST resource names follow the SDK naming rules; case-sensitive quoted
identifiers require a native connection or pool. Install the optional SDK extra
for REST (Python 3.10–3.15):

```bash
pip install "langchain-oracledb[vecdb]"
```

```python
import os

from oracle_vecdb import Configuration
from langchain_oracledb import OracleVS

# embedding_model implements LangChain's Embeddings interface.
store = OracleVS.from_vecdb_config(
    config=Configuration(
        rest_url=os.environ["ORACLE_VECDB_REST_URL"],
        username=os.environ["ORACLE_VECDB_USERNAME"],
        password=os.environ["ORACLE_VECDB_PASSWORD"],
    ),
    table_name="DOCUMENTS",
    embedding=embedding_model,
    mutate_on_duplicate=True,
)
store.add_texts(["Oracle vector search"], ids=["doc-1"])
documents = store.similarity_search("vector search", k=4)
```

An existing ORDS-configured `OracleVecDB` client can also be supplied as `client`.
REST writes require `mutate_on_duplicate=True`; insert-only writes, metadata
text/hybrid search and indexing, vectorizer preferences, and migration require a
native connection. Native stores also support the corresponding async APIs.

- Vector similarity filters use OracleVS validation and translate nested paths
  and `$instr` to the VecDB QBE grammar. Unsupported predicates raise `ValueError`
  before the search call: array/object `$eq` and `$ne`, scalar `$ne`, field
  `$not`, and `$regex` (including inside logical expressions). These require
  package support or further proven translations to preserve exact equality,
  array/missing-field negation, type coercion, and `like_regex` semantics.
  Use `backend="sql"` when those predicates are required. This restriction
  applies to vector similarity searches, including async and MMR searches.
- Native VecDB keyword retrieval uses `JSON_TEXTCONTAINS` with `SCORE(1)`
  and returns matches in descending relevance order. `return_scores=True`
  exposes those Oracle Text scores. Keyword-only hybrid retrieval reports
  `vector_score=0`. Numeric document metadata is decoded as `Decimal`, matching
  SQL JSON metadata; booleans and nulls retain their types.
- VecDB reserves metadata key `text` for document content. Search scores remain
  lower-is-better distances; `output_selector` and `advanced_options` expose SDK
  query options.
- Native insert-only writes follow regular SQL insert behavior: autocommit and
  pending transactions are accepted. Row errors (including duplicate IDs) are
  logged and excluded from returned IDs; successful rows are committed, along
  with any pending work on that connection. Driver/commit failures propagate
  without an automatic rollback, as on the SQL backend.
- VecDB creates its vector index with the table. To configure a new table, pass
  `vector_index_params={"idx_type": "HNSW", "neighbors": 32, "efConstruction": 200}`
  to `OracleVS`, `acreate()`, `from_texts()`, or `afrom_texts()`. This VecDB-only
  option uses the same LangChain HNSW/IVF keys as `create_index()`; it is applied
  only to new tables, and SQL stores reject it. Reopening preserves existing
  index settings.
- `create_index()` and `acreate_index()` leave an existing VecDB vector index
  unchanged when no parameters (or an empty dictionary) are supplied. Explicit
  parameters request a vector-only rebuild; a missing index is created instead.
  For example, `create_index(store.client, store, {"idx_type": "IVF",
  "neighbor_part": 32, "accuracy": 90})` maps `neighbor_part` to VecDB partitions.
  VecDB owns the physical index name, so `idx_name` is not applied.
- No distribution setting is injected. Supply `distribute_params` explicitly only
  when appropriate for the target database. Explicit HNSW/IVF requests use the
  LangChain construction defaults where the package requires concrete values;
  absent index configuration is omitted from the package request.
- A VecDB creation/rebuild returns the provider response; a skipped operation
  returns `None`. Check `describe_vecdb_index()` for readiness. Serialize index
  maintenance and use an idle connection for native DDL. Explicit text/hybrid
  index operations remain available with a native connection, and their freshness
  follows the synchronization schedule.
- `drop_table_purge()` and `adrop_table_purge()` use package/SDK cleanup for VecDB
  and SQL cleanup for ordinary tables. Package failures never trigger a SQL drop.

#### OracleVS

Use Oracle Vector Database with `OracleVS`. More information can be found in [Oracle AI Vector Search: Vector Store](https://python.langchain.com/docs/integrations/vectorstores/oracle/) documentation.


```python
from langchain_oracledb.vectorstores import OracleVS
from langchain_oracledb.vectorstores.oraclevs import create_index
from langchain_oracledb.document_loaders.oracleai import OracleTextSplitter
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_oracledb.vectorstores import DistanceStrategy

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-mpnet-base-v2"
)
vector_store = OracleVS(conn, embedding_model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE)

# add texts to the vector database
texts = ["A tablespace can be online (accessible) or offline (not accessible) whenever the database is open.\nA tablespace is usually online so that its data is available to users. The SYSTEM tablespace and temporary tablespaces cannot be taken offline.", "The database stores LOBs differently from other data types. Creating a LOB column implicitly creates a LOB segment and a LOB index. "]
metadata = [
    {"id": "100", "link": "Document Example Test 1"},
    {"id": "101", "link": "Document Example Test 2"},
]

vector_store.add_texts(texts, metadata)

# for large documents, chunk before ingesting so each chunk is stored in its own row
splitter = OracleTextSplitter(conn=conn, params={"split": "sentence", "max": 20})
documents = [
    Document(
        page_content=(
            "A tablespace can be online (accessible) or offline (not accessible). "
            "The SYSTEM tablespace cannot be taken offline."
        ),
        metadata={"id": "200", "link": "Large Document Example"},
    )
]
vector_store.add_documents(
    documents,
    text_splitter=splitter,
    ids=["200"],  # becomes 200#chunk-0, 200#chunk-1, ...
)

create_index(
    conn, vector_store, params={"idx_name": "hnsw_oravs", "idx_type": "HNSW"}
)

# perform siliarity search
vector_store.similarity_search("How does a database stores LOBs?", 1)

```

### Document Loaders

#### OracleDocLoader

Load your documents using `OracleDocLoader`. More information can be found in [Oracle AI Vector Search: Document Processing](https://python.langchain.com/docs/integrations/document_loaders/oracleai/) documentation.

```python
from langchain_oracledb.document_loaders.oracleai import OracleDocLoader

"""
# loading a local file
loader_params = {}
loader_params["file"] = "<file>"

# loading from a local directory
loader_params = {}
loader_params["dir"] = "<directory>"
"""

# loading from Oracle Database table
loader_params = {
    "owner": "<owner>",
    "tablename": "demo_tab",
    "colname": "data",
}

# load the docs
loader = OracleDocLoader(conn=conn, params=loader_params)
docs = loader.load()

# verify
print(f"Number of docs loaded: {len(docs)}")
```

#### OracleTextSplitter

Chunk your documents using `OracleTextSplitter`. More information can be found in [Oracle AI Vector Search: Document Processing](https://python.langchain.com/docs/integrations/document_loaders/oracleai/) documentation.

```python
from langchain_oracledb.document_loaders.oracleai import OracleTextSplitter
from langchain_oracledb.document_loaders.oracleai import OracleDocLoader

# loading from Oracle Database table
loader_params = {
    "owner": "<owner>",
    "tablename": "demo_tab",
    "colname": "data",
}

# load the docs
loader = OracleDocLoader(conn=conn, params=loader_params)
docs = loader.load()

"""
# some examples
# split by chars, max 500 chars
splitter_params = {"split": "chars", "max": 500, "normalize": "all"}

# split by words, max 100 words
splitter_params = {"split": "words", "max": 100, "normalize": "all"}

# split by sentence, max 20 sentences
splitter_params = {"split": "sentence", "max": 20, "normalize": "all"}
"""

# split by default parameters
splitter_params = {"normalize": "all"}

# get the splitter instance
splitter = OracleTextSplitter(conn=conn, params=splitter_params)

list_chunks = []
for doc in docs:
    chunks = splitter.split_text(doc.page_content)
    list_chunks.extend(chunks)

# verify
print(f"Number of Chunks: {len(list_chunks)}")
# print(f"Chunk-0: {list_chunks[0]}") # content
```

#### OracleAutonomousDatabaseLoader

Load documents from Oracle Autonomous Database using `OracleAutonomousDatabaseLoader`. More information can be found in [Oracle Autonomous Database](https://python.langchain.com/docs/integrations/document_loaders/oracleadb_loader/) documentation.

```python
from langchain_oracledb.document_loaders import OracleAutonomousDatabaseLoader
from settings import s

SQL_QUERY = "select channel_id, channel_desc from sh.channels where channel_desc = :1 fetch first 5 rows only"

doc_loader = OracleAutonomousDatabaseLoader(
    query=SQL_QUERY,
    user=s.USERNAME,
    password=s.PASSWORD,
    schema=s.SCHEMA,
    dsn=s.DSN,
    parameters=["Direct Sales"],
)
doc = doc_loader.load()
```

With mutual TLS authentication (mTLS), wallet_location and wallet_password are required to create the connection, user can create connection by providing either connection string or tns configuration details. With TLS authentication, wallet_location and wallet_password are not required. Bind variable option is provided by argument "parameters".

### Embeddings

#### OracleEmbeddings

Generate embeddings for your documents using `OracleEmbeddings`. More information can be found in [Oracle AI Vector Search: Generate Embeddings](https://python.langchain.com/docs/integrations/text_embedding/oracleai/) documentation.

```python
from langchain_oracledb.embeddings.oracleai import OracleEmbeddings

"""
# using ocigenai
embedder_params = {
    "provider": "ocigenai",
    "credential_name": "OCI_CRED",
    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
    "model": "cohere.embed-english-light-v3.0",
}

# using huggingface
embedder_params = {
    "provider": "huggingface",
    "credential_name": "HF_CRED",
    "url": "https://api-inference.huggingface.co/pipeline/feature-extraction/",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "wait_for_model": "true"
}
"""

# using ONNX model loaded to Oracle Database
embedder_params = {"provider": "database", "model": "demo_model"}

# if a proxy is not required for your environment, you can omit the 'proxy' parameter below
embedder = OracleEmbeddings(conn=conn, params=embedder_params, proxy=proxy)
embed = embedder.embed_query("Hello World!")

# verify
print(f"Embedding generated by OracleEmbeddings: {embed}")
```

### Utilities

#### OracleSummary

Generate summary for your documents using `OracleSummary`. More information can be found in [Oracle AI Vector Search: Generate Summary](https://python.langchain.com/docs/integrations/tools/oracleai/) documentation.

```python
from langchain_oracledb.utilities.oracleai import OracleSummary
from langchain_core.documents import Document

"""
# using 'ocigenai' provider
summary_params = {
    "provider": "ocigenai",
    "credential_name": "OCI_CRED",
    "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/summarizeText",
    "model": "cohere.command",
}

# using 'huggingface' provider
summary_params = {
    "provider": "huggingface",
    "credential_name": "HF_CRED",
    "url": "https://api-inference.huggingface.co/models/",
    "model": "facebook/bart-large-cnn",
    "wait_for_model": "true"
}
"""

# using 'database' provider
summary_params = {
    "provider": "database",
    "glevel": "S",
    "numParagraphs": 1,
    "language": "english",
}

# get the summary instance
# remove proxy if not required
summ = OracleSummary(conn=conn, params=summary_params, proxy=proxy)
summary = summ.get_summary(
    "In the heart of the forest, "
    + "a lone fox ventured out at dusk, seeking a lost treasure. "
    + "With each step, memories flooded back, guiding its path. "
    + "As the moon rose high, illuminating the night, the fox unearthed "
    + "not gold, but a forgotten friendship, worth more than any riches."
)

print(f"Summary generated by OracleSummary: {summary}")
```
