# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from langchain_oci.agents.oci_agent import (
    AgentEvent,
    AgentHooks,
    AgentResult,
    AgentState,
    BaseCheckpointer,
    Checkpoint,
    CompressionConfig,
    CompressionStrategy,
    ConfidenceSignal,
    FileCheckpointer,
    IterationContext,
    MemoryCheckpointer,
    OCIGenAIAgent,
    ReasoningStep,
    ReflectEvent,
    SignalType,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolExecution,
    ToolHookContext,
    ToolResultContext,
    ToolStartEvent,
    create_logging_hooks,
    create_metrics_hooks,
)
from langchain_oci.agents.react import create_oci_agent
from langchain_oci.chat_models.oci_data_science import (
    ChatOCIModelDeployment,
    ChatOCIModelDeploymentTGI,
    ChatOCIModelDeploymentVLLM,
)
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI, ChatOCIOpenAI
from langchain_oci.common.auth import OCIAuthType
from langchain_oci.embeddings.oci_data_science_model_deployment_endpoint import (
    OCIModelDeploymentEndpointEmbeddings,
)
from langchain_oci.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain_oci.llms.oci_data_science_model_deployment_endpoint import (
    BaseOCIModelDeployment,
    OCIModelDeploymentLLM,
    OCIModelDeploymentTGI,
    OCIModelDeploymentVLLM,
)
from langchain_oci.llms.oci_generative_ai import OCIGenAI, OCIGenAIBase
from langchain_oci.utils.vision import (
    VISION_MODELS,
    encode_image,
    is_vision_model,
    load_image,
)

__all__ = [
    "ChatOCIGenAI",
    "ChatOCIModelDeployment",
    "ChatOCIModelDeploymentTGI",
    "ChatOCIModelDeploymentVLLM",
    "OCIAuthType",
    "OCIGenAIEmbeddings",
    "OCIModelDeploymentEndpointEmbeddings",
    "OCIGenAIBase",
    "OCIGenAI",
    "BaseOCIModelDeployment",
    "OCIModelDeploymentLLM",
    "OCIModelDeploymentTGI",
    "OCIModelDeploymentVLLM",
    "ChatOCIOpenAI",
    # Agents
    "create_oci_agent",
    "OCIGenAIAgent",
    "AgentState",
    "AgentResult",
    "ReasoningStep",
    "ToolExecution",
    "AgentEvent",
    "ThinkEvent",
    "ToolStartEvent",
    "ToolCompleteEvent",
    "ReflectEvent",
    "TerminateEvent",
    # Hooks
    "AgentHooks",
    "ToolHookContext",
    "ToolResultContext",
    "IterationContext",
    "create_logging_hooks",
    "create_metrics_hooks",
    # Compression
    "CompressionConfig",
    "CompressionStrategy",
    # Confidence Signals
    "ConfidenceSignal",
    "SignalType",
    # Checkpointing
    "BaseCheckpointer",
    "Checkpoint",
    "MemoryCheckpointer",
    "FileCheckpointer",
    # Vision / image utilities
    "load_image",
    "encode_image",
    "is_vision_model",
    "VISION_MODELS",
]
