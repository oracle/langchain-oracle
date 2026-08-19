# langchain-oracledb (js) samples

## RAG Pipeline Application
This sample application uses Oracle AI Database 26ai's Vector Search capabilities along with the OCI Generative AI service to provide relevant answers from the context provided.

The sample data source (corpus) is available in the `example_data` directory.

Here are the environment variables to be set:
| Environment variable | Description | Default value |
|---|---|---|
| `ORACLEDB_USER` | Oracle Database username used to create the database connection - `testuser` in this case | Not provided; required |
| `ORACLEDB_PASSWORD` | Oracle Database password for `ORACLEDB_USER` | Not provided; required |
| `ORACLEDB_CONNECTION_STRING` | Oracle Database connection string  | Not provided; required |
| `EMBEDDING_ONNX_MODEL` | Name of the embedding model available in Oracle Database and used by `OracleEmbeddings` | Not provided; required |
| `DOCUMENTS_FOLDER` | Folder containing the source documents loaded by `OracleDocLoader` | Not provided; required |
| `OCI_COMPARTMENT_ID` | OCI compartment OCID used for the OCI Generative AI inference request | Not provided; required |
| `OCI_MODEL_ID` | OCI Generative AI model ID used for on-demand inference | `xai.grok-4.3` |
| `OCI_ENDPOINT` | OCI Generative AI inference endpoint | `https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com` |
| `OCI_CONFIG_FILE` | Path to the OCI configuration file used by `ConfigFileAuthenticationDetailsProvider` | `~/.oci/config` |
| `OCI_CONFIG_PROFILE` | OCI configuration profile used for authentication | `DEFAULT` |

Run the application as

```
npm install
npx tsx oracle-rag-full-pipeline.ts
```

For all the other relevant setup information, check the [RAG Pipeline Blog](https://medium.com/@sharad-chandran/transparent-rag-pipelines-with-oracle-ai-database-langchain-js-d985704f1fec)
