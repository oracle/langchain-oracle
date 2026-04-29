# ADB Multi-Store Deep Research Tutorial

This tutorial is the didactic companion to:

- `samples/11-deepresearch/adb_multi_store_huggingface_example.py`

It shows the exact workflow:

1. download two small datasets from Hugging Face
2. convert the rows into `Document` objects
3. ingest them into two ADB vector tables
4. run `create_deep_research_agent(...)` across both tables

The two stores in this sample are:

1. `medical_research`
2. `news_research`

## 1. Install Dependencies

```bash
cd libs/oci
poetry install --with lint,typing,test,test_integration
poetry run pip install datasets deepagents langchain-oracledb
```

## 2. Export OCI and ADB Settings

```bash
export OCI_COMPARTMENT_ID="ocid1.compartment..."
export OCI_REGION="us-ashburn-1"
export OCI_AUTH_TYPE="API_KEY"
export OCI_AUTH_PROFILE="API_KEY_AUTH"

export OCI_EMBEDDING_MODEL="cohere.embed-v4.0"
export OCI_DEEP_RESEARCH_MODEL="google.gemini-2.5-flash"

export ADB_DSN="deepresearch_low"
export ADB_USER="ADMIN"
export ADB_PASSWORD="..."
export ADB_WALLET_LOCATION="$HOME/.oci/wallets/deepresearch"
export ADB_WALLET_PASSWORD="..."
```

Validation:

```bash
python - <<'PY'
import os
required = [
    "OCI_COMPARTMENT_ID",
    "ADB_DSN",
    "ADB_USER",
    "ADB_PASSWORD",
    "ADB_WALLET_LOCATION",
]
missing = [name for name in required if not os.environ.get(name)]
print("MISSING:", missing)
PY
```

You want `MISSING: []`.

## 3. Run the Sample

From the repository root:

```bash
PYTHONPATH=libs/oci python samples/11-deepresearch/adb_multi_store_huggingface_example.py \
  --limit 10 \
  --run-id tutorial01 \
  --output samples/11-deepresearch/adb_multi_store_huggingface_output.md
```

What the script does:

1. downloads `pubmed_qa` and `ag_news`
2. converts them into LangChain documents
3. creates two ADB tables named from `--run-id`
4. ingests both corpora
5. runs one Deep Research prompt comparing how the two corpora present information

The generated table names look like:

1. `VECTOR_DOCUMENTS_MEDICAL_tutorial01`
2. `VECTOR_DOCUMENTS_NEWS_tutorial01`

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

- `samples/11-deepresearch/adb_multi_store_huggingface_output.md`

## 5. Stop After Ingest Only

If you want to validate loading first and skip the Gemini run:

```bash
PYTHONPATH=libs/oci python samples/11-deepresearch/adb_multi_store_huggingface_example.py \
  --limit 10 \
  --run-id tutorial01 \
  --skip-research
```

## 6. What This Sample Proves

This sample exercises the real shipped path:

1. Hugging Face download
2. ADB datastore construction
3. ADB ingest through the adapter API
4. `create_deep_research_agent(...)`
5. cross-store Deep Research over two vector-backed stores

## Troubleshooting

Problem: `ImportError: No module named datasets`

Fix:

```bash
cd libs/oci
poetry run pip install datasets
```

Problem: ADB connect error

Check:

1. `ADB_DSN`
2. `ADB_USER`
3. `ADB_PASSWORD`
4. `ADB_WALLET_LOCATION`
5. `ADB_WALLET_PASSWORD`

Problem: Deep Research returns nothing

Check:

1. `OCI_COMPARTMENT_ID`
2. `OCI_REGION`
3. `OCI_DEEP_RESEARCH_MODEL`
4. your OCI auth profile and region access
