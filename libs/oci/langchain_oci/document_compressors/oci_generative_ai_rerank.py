# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Document compressor backed by OCI Generative AI rerank models."""

from copy import deepcopy
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import pre_init
from pydantic import ConfigDict, Field

from langchain_oci.common.async_support import OCIAsyncClient
from langchain_oci.common.auth import create_oci_client_kwargs
from langchain_oci.common.utils import CUSTOM_ENDPOINT_PREFIX


class OCIGenAIRerank(BaseDocumentCompressor):
    """Document compressor using OCI Generative AI rerank models.

    Reranks documents against a query with an OCI-hosted rerank model
    (e.g. ``cohere.rerank-v3.5``, ``cohere.rerank-v4.0-fast``,
    ``cohere.rerank-v4.0-pro``) and keeps the ``top_n`` most relevant.

    To authenticate, the OCI client uses the methods described in
    https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm

    Example:
        .. code-block:: python

            from langchain.retrievers import ContextualCompressionRetriever
            from langchain_oci import OCIGenAIRerank

            reranker = OCIGenAIRerank(
                model_id="cohere.rerank-v3.5",
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                compartment_id="MY_OCID",
                top_n=3,
            )
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=vector_store.as_retriever(search_kwargs={"k": 20}),
            )
    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)  #: :meta private:

    model_id: str = "cohere.rerank-v3.5"
    """Id of the OCI GenAI rerank model to use."""

    service_endpoint: Optional[str] = None
    """OCI GenAI service endpoint URL."""

    compartment_id: Optional[str] = None
    """OCID of the compartment."""

    auth_type: Optional[str] = "API_KEY"
    """Authentication type: API_KEY (default), SECURITY_TOKEN,
    INSTANCE_PRINCIPAL or RESOURCE_PRINCIPAL."""

    auth_profile: Optional[str] = "DEFAULT"
    """The name of the profile in ~/.oci/config."""

    auth_file_location: Optional[str] = "~/.oci/config"
    """Path to the OCI config file."""

    top_n: Optional[int] = 3
    """Number of documents to return. None returns all documents reranked."""

    score_threshold: Optional[float] = None
    """If set, drop documents whose relevance score is below this value."""

    max_chunks_per_document: Optional[int] = None
    """Passed through to the OCI rerank API when set."""

    max_tokens_per_document: Optional[int] = None
    """Passed through to the OCI rerank API when set."""

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, protected_namespaces=()
    )

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the OCI client can be constructed."""
        if values.get("client") is not None:
            return values

        try:
            import oci

            client_kwargs = create_oci_client_kwargs(
                auth_type=values["auth_type"],
                service_endpoint=values["service_endpoint"],
                auth_file_location=values["auth_file_location"],
                auth_profile=values["auth_profile"],
            )
            values["client"] = oci.generative_ai_inference.GenerativeAiInferenceClient(
                **client_kwargs
            )
        except ImportError as ex:
            raise ImportError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex
        except Exception as e:
            raise ValueError(
                "Could not authenticate with OCI client. "
                "If INSTANCE_PRINCIPAL or RESOURCE_PRINCIPAL is used, "
                "please check the specified auth_profile, "
                "auth_file_location and auth_type are valid.",
                e,
            ) from e

        return values

    @cached_property
    def _async_client(self) -> OCIAsyncClient:
        """Async client sharing the sync client's signer/config."""
        base_client = self.client.base_client
        return OCIAsyncClient(
            service_endpoint=self.service_endpoint,  # type: ignore[arg-type]
            signer=base_client.signer,
            config=getattr(base_client, "config", {}),
        )

    async def aclose(self) -> None:
        """Close the async HTTP client and release resources."""
        if "_async_client" in self.__dict__:
            await self._async_client.close()
            del self.__dict__["_async_client"]

    def _build_rerank_details(
        self,
        documents: Sequence[str],
        query: str,
        top_n: Optional[int],
    ) -> Any:
        from oci.generative_ai_inference import models

        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode: Any = models.DedicatedServingMode(endpoint_id=self.model_id)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)

        kwargs: Dict[str, Any] = {
            "input": query,
            "documents": list(documents),
            "compartment_id": self.compartment_id,
            "serving_mode": serving_mode,
            "top_n": top_n if top_n is not None else self.top_n,
        }
        if self.max_chunks_per_document is not None:
            kwargs["max_chunks_per_document"] = self.max_chunks_per_document
        if self.max_tokens_per_document is not None:
            kwargs["max_tokens_per_document"] = self.max_tokens_per_document
        return models.RerankTextDetails(**kwargs)

    @staticmethod
    def _doc_text(doc: Union[str, Document, Dict[str, Any]]) -> str:
        if isinstance(doc, Document):
            return doc.page_content
        if isinstance(doc, dict):
            return str(doc.get("text", doc))
        return str(doc)

    def rerank(
        self,
        documents: Sequence[Union[str, Document, Dict[str, Any]]],
        query: str,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents against a query.

        Returns:
            A list of ``{"index": int, "relevance_score": float}`` dicts,
            most relevant first, filtered by ``score_threshold`` when set.
        """
        if not documents:
            return []
        texts = [self._doc_text(d) for d in documents]
        details = self._build_rerank_details(texts, query, top_n)
        response = self.client.rerank_text(details)
        return self._ranks_to_results(response.data.document_ranks)

    async def arerank(
        self,
        documents: Sequence[Union[str, Document, Dict[str, Any]]],
        query: str,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of :meth:`rerank` (native async I/O)."""
        if not documents:
            return []
        texts = [self._doc_text(d) for d in documents]
        details = self._build_rerank_details(texts, query, top_n)
        body = self.client.base_client.sanitize_for_serialization(details)
        data = await self._async_client.rerank_text_async(body)
        ranks = [
            {"index": r.get("index"), "relevance_score": r.get("relevanceScore")}
            for r in data.get("documentRanks", [])
        ]
        return self._filter_and_sort(ranks)

    def _ranks_to_results(self, document_ranks: Any) -> List[Dict[str, Any]]:
        ranks = [
            {"index": r.index, "relevance_score": r.relevance_score}
            for r in document_ranks
        ]
        return self._filter_and_sort(ranks)

    def _filter_and_sort(self, ranks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.score_threshold is not None:
            ranks = [r for r in ranks if r["relevance_score"] >= self.score_threshold]
        return sorted(ranks, key=lambda r: r["relevance_score"], reverse=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Rerank *documents* against *query*, keeping the top_n most relevant.

        Each returned Document is a copy with ``relevance_score`` added to
        its metadata.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(
                page_content=doc.page_content, metadata=deepcopy(doc.metadata)
            )
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Async version of :meth:`compress_documents` (native async I/O)."""
        compressed = []
        for res in await self.arerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(
                page_content=doc.page_content, metadata=deepcopy(doc.metadata)
            )
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
