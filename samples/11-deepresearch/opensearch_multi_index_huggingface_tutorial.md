# OpenSearch Multi-Index Deep Research Tutorial

This tutorial is the didactic companion to:

- `samples/11-deepresearch/opensearch_multi_index_huggingface_example.py`

It shows the exact workflow:

1. download two small datasets from Hugging Face
2. convert the rows into `Document` objects
3. ingest them into two OpenSearch indices
4. run `create_deep_research_agent(...)` across both indices

The two stores in this sample are:

1. `medical_research`
2. `news_research`

## 1. Install Dependencies

```bash
cd libs/oci
poetry install --with lint,typing,test,test_integration
poetry run pip install datasets deepagents opensearch-py
```

## 2. Export OCI and OpenSearch Settings

```bash
export OCI_COMPARTMENT_ID="ocid1.compartment..."
export OCI_REGION="us-ashburn-1"
export OCI_AUTH_TYPE="API_KEY"
export OCI_AUTH_PROFILE="API_KEY_AUTH"

export OCI_EMBEDDING_MODEL="cohere.embed-v4.0"
export OCI_DEEP_RESEARCH_MODEL="google.gemini-2.5-flash"

export OPENSEARCH_ENDPOINT="https://opensearch.example.com:9200"
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="..."
export OPENSEARCH_VECTOR_FIELD="vector_field"
export OPENSEARCH_SEARCH_FIELDS="title,content"
export OPENSEARCH_USE_SSL="true"
export OPENSEARCH_VERIFY_CERTS="false"
```

If you are using the local Eolas config from
`~/Projects/observai/eolas/config/config.local.yaml`, the OpenSearch values come
from that file directly:

1. `vector_store.endpoint` -> `OPENSEARCH_ENDPOINT`
2. `vector_store.username` -> `OPENSEARCH_USERNAME`
3. `vector_store.password` -> `OPENSEARCH_PASSWORD`
4. `vector_store.use_ssl` -> `OPENSEARCH_USE_SSL`
5. `vector_store.verify_certs` -> `OPENSEARCH_VERIFY_CERTS`

The sample creates fresh tutorial indices on that same cluster, so you do not
need a pre-existing Eolas KB index name.

Validation:

```bash
python - <<'PY'
import os
required = [
    "OCI_COMPARTMENT_ID",
    "OPENSEARCH_ENDPOINT",
    "OPENSEARCH_USERNAME",
    "OPENSEARCH_PASSWORD",
]
missing = [name for name in required if not os.environ.get(name)]
print("MISSING:", missing)
PY
```

You want `MISSING: []`.

## 3. Run the Sample

From the repository root:

```bash
PYTHONPATH=libs/oci python samples/11-deepresearch/opensearch_multi_index_huggingface_example.py \
  --limit 10 \
  --run-id tutorial01 \
  --output samples/11-deepresearch/opensearch_multi_index_huggingface_output.md
```

What the script does:

1. downloads `pubmed_qa` and `ag_news`
2. converts them into LangChain documents
3. creates two OpenSearch indices named from `--run-id`
4. creates an Eolas-compatible `knn_vector` mapping for each index
5. ingests both corpora using OCI embeddings
6. runs one Deep Research prompt across both stores

The generated index names look like:

1. `hf_pubmed_research_tutorial01`
2. `hf_ag_news_research_tutorial01`

## 4. Expected Output

During ingestion, the script prints a small JSON summary such as:

```json
{
  "ingested": {
    "medical_research": 10,
    "news_research": 10
  },
  "run_id": "tutorial01"
}
```

Then it writes the final report to:

- `samples/11-deepresearch/opensearch_multi_index_huggingface_output.md`

## 5. Stop After Ingest Only

If you want to validate loading first and skip the Gemini run:

```bash
PYTHONPATH=libs/oci python samples/11-deepresearch/opensearch_multi_index_huggingface_example.py \
  --limit 10 \
  --run-id tutorial01 \
  --skip-research
```

## 6. What This Sample Proves

This sample exercises the real shipped path:

1. Hugging Face download
2. OpenSearch datastore construction
3. OpenSearch ingest through the adapter API
4. `create_deep_research_agent(...)`
5. cross-index Deep Research over two vector-backed stores

## Troubleshooting

Problem: `ImportError: No module named datasets`

Fix:

```bash
cd libs/oci
poetry run pip install datasets
```

Problem: OpenSearch connection error

Check:

1. `OPENSEARCH_ENDPOINT`
2. `OPENSEARCH_USERNAME`
3. `OPENSEARCH_PASSWORD`
4. `OPENSEARCH_VECTOR_FIELD`
5. TLS settings in `OPENSEARCH_USE_SSL` and `OPENSEARCH_VERIFY_CERTS`

Problem: Semantic search does not return useful hits

Check:

1. the OpenSearch mapping uses the same vector field named in `OPENSEARCH_VECTOR_FIELD`
2. the same OCI embedding model is used for ingest and retrieval
3. the cluster is reachable from the machine running the sample
