# Deep Research Samples

This folder contains the datastore-backed Deep Research samples for
`langchain-oci`.

Included files:

1. `adb_multi_store_huggingface_tutorial.md`
2. `adb_multi_store_huggingface_example.py`
3. `opensearch_multi_index_huggingface_tutorial.md`
4. `opensearch_multi_index_huggingface_example.py`

These samples show the full workflow:

1. download a small dataset slice from Hugging Face
2. convert the rows into LangChain `Document` objects
3. ingest those documents into two vector-backed stores
4. run `create_deep_research_agent(...)` across both stores
5. write a final markdown memo

## Step By Step

Follow this order:

1. Install the `libs/oci` dependencies.
2. Export OCI credentials and model settings.
3. Export backend-specific settings for either ADB or OpenSearch.
4. Run one of the sample scripts from the repository root with `PYTHONPATH=libs/oci`.
5. Check the generated markdown report.

### 1. Install Dependencies

```bash
cd libs/oci
poetry install --with lint,typing,test,test_integration
poetry run pip install datasets deepagents
```

For ADB also install:

```bash
poetry run pip install oracledb langchain-oracledb
```

For OpenSearch also install:

```bash
poetry run pip install opensearch-py
```

### 2. Export OCI Settings

```bash
export OCI_COMPARTMENT_ID="ocid1.compartment..."
export OCI_REGION="us-ashburn-1"
export OCI_AUTH_TYPE="API_KEY"
export OCI_AUTH_PROFILE="API_KEY_AUTH"
export OCI_EMBEDDING_MODEL="cohere.embed-v4.0"
export OCI_DEEP_RESEARCH_MODEL="google.gemini-2.5-flash"
```

Optional for a larger output budget:

```bash
export OCI_DEEP_RESEARCH_MAX_TOKENS="65536"
```

### 3. Choose a Backend

For ADB export:

```bash
export ADB_DSN="deepresearch_low"
export ADB_USER="ADMIN"
export ADB_PASSWORD="..."
export ADB_WALLET_LOCATION="$HOME/.oci/wallets/deepresearch"
export ADB_WALLET_PASSWORD="..."
```

For OpenSearch export:

```bash
export OPENSEARCH_ENDPOINT="https://opensearch.example.com:9200"
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="..."
export OPENSEARCH_VECTOR_FIELD="vector_field"
export OPENSEARCH_SEARCH_FIELDS="title,content"
export OPENSEARCH_USE_SSL="true"
export OPENSEARCH_VERIFY_CERTS="false"
```

If you already have the Eolas local config, you can take the OpenSearch values
from `~/Projects/observai/eolas/config/config.local.yaml`.

### 4. Run a Sample

ADB:

```bash
PYTHONPATH=libs/oci python samples/11-deepresearch/adb_multi_store_huggingface_example.py \
  --limit 10 \
  --run-id tutorial01 \
  --output samples/11-deepresearch/adb_multi_store_huggingface_output.md
```

OpenSearch:

```bash
PYTHONPATH=libs/oci python samples/11-deepresearch/opensearch_multi_index_huggingface_example.py \
  --limit 10 \
  --run-id tutorial01 \
  --output samples/11-deepresearch/opensearch_multi_index_huggingface_output.md
```

If you only want to verify ingestion first, add:

```bash
--skip-research
```

### 5. Read the Backend-Specific Tutorial

Use these for the detailed explanation of each path:

1. `samples/11-deepresearch/adb_multi_store_huggingface_tutorial.md`
2. `samples/11-deepresearch/opensearch_multi_index_huggingface_tutorial.md`
