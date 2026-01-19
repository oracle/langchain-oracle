# langchain-oracledb

This package contains the LangChain integrations with [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/).

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
import oracledb

# Please update with your username, password, hostname, port and service_name
username = "<username>"
password = "<password>"
dsn = "<hostname>:<port>/<service_name>"

connection = oracledb.connect(user=username, password=password, dsn=dsn)
print("Connection successful!")
```

### Vector Stores

#### OracleVS

Use Oracle Vector Database with `OracleVS`. More information can be found in [Oracle AI Vector Search: Vector Store](https://python.langchain.com/docs/integrations/vectorstores/oracle/) documentation.


```python
from langchain_oracledb.vectorstores import OracleVS
from langchain_oracledb.vectorstores.oraclevs import create_index

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

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

create_index(
    conn, vector_store, params={"idx_name": "hnsw_oravs", "idx_type": "HNSW"}
)

# perform siliarity search
vs.similarity_search("How does a database stores LOBs?", 1)

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
