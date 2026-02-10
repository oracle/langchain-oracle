# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Code Assistant Agent Example.

Demonstrates OCIGenAIAgent as a coding assistant that can:
- Search documentation
- Explain code patterns
- Suggest best practices
- Run code validation

Shows streaming events for real-time feedback.
"""
# ruff: noqa: T201

import os

from langchain_core.tools import tool

from langchain_oci import (
    OCIGenAIAgent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)

# Simulated documentation database
DOCS = {
    "async": """
Async/Await in Python:
- Use 'async def' to define coroutines
- Use 'await' to wait for async operations
- Use 'asyncio.run()' to execute async code
- Common patterns: asyncio.gather() for concurrent tasks
""",
    "dataclass": """
Python Dataclasses:
- Import: from dataclasses import dataclass
- Use @dataclass decorator on classes
- Automatic __init__, __repr__, __eq__ generation
- Use field() for default factories
- frozen=True for immutable dataclasses
""",
    "pydantic": """
Pydantic Models:
- Import: from pydantic import BaseModel
- Define fields with type annotations
- Automatic validation on instantiation
- Use Field() for constraints and metadata
- ConfigDict for model configuration
""",
    "typing": """
Python Type Hints:
- Basic: str, int, float, bool
- Collections: List[T], Dict[K, V], Set[T]
- Optional: Optional[T] = Union[T, None]
- Callable: Callable[[Args], Return]
- Generic: TypeVar, Generic[T]
""",
}

CODE_PATTERNS = {
    "singleton": """
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
""",
    "factory": """
class AnimalFactory:
    @staticmethod
    def create(animal_type: str) -> Animal:
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown: {animal_type}")
""",
    "context_manager": """
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)
""",
}


@tool
def search_docs(topic: str) -> str:
    """Search documentation for a programming topic.

    Args:
        topic: The topic to search for (e.g., async, pydantic, typing).

    Returns:
        Documentation for the topic.
    """
    topic_lower = topic.lower()
    for key, doc in DOCS.items():
        if key in topic_lower or topic_lower in key:
            return doc.strip()

    available = ", ".join(DOCS.keys())
    return f"No docs found for '{topic}'. Available topics: {available}"


@tool
def get_code_pattern(pattern_name: str) -> str:
    """Get a code example for a design pattern.

    Args:
        pattern_name: Name of the pattern (e.g., singleton, factory).

    Returns:
        Code example for the pattern.
    """
    pattern_lower = pattern_name.lower().replace(" ", "_")
    for key, code in CODE_PATTERNS.items():
        if key in pattern_lower or pattern_lower in key:
            return f"```python{code}```"

    available = ", ".join(CODE_PATTERNS.keys())
    return f"Pattern '{pattern_name}' not found. Available: {available}"


@tool
def validate_python_syntax(code: str) -> str:
    """Validate Python code syntax.

    Args:
        code: Python code to validate.

    Returns:
        Validation result.
    """
    try:
        compile(code, "<string>", "exec")
        return "✓ Syntax is valid Python code."
    except SyntaxError as e:
        return f"✗ Syntax error at line {e.lineno}: {e.msg}"


@tool
def suggest_improvement(code_description: str) -> str:
    """Suggest improvements for described code.

    Args:
        code_description: Description of the code to improve.

    Returns:
        Improvement suggestions.
    """
    suggestions = []
    desc_lower = code_description.lower()

    if "loop" in desc_lower:
        suggestions.append("Consider list comprehensions for simple loops")
    if "dict" in desc_lower or "dictionary" in desc_lower:
        suggestions.append("Use dict.get() with defaults to avoid KeyError")
    if "file" in desc_lower:
        suggestions.append("Use context managers (with statement) for file handling")
    if "class" in desc_lower:
        suggestions.append("Consider dataclasses for data-holding classes")
    if "error" in desc_lower or "exception" in desc_lower:
        suggestions.append("Catch specific exceptions, not bare except")

    if not suggestions:
        suggestions.append("Follow PEP 8 style guidelines")
        suggestions.append("Add type hints for better code clarity")

    return "Suggestions:\n" + "\n".join(f"• {s}" for s in suggestions)


def main():
    print("=" * 70)
    print("Code Assistant Agent Demo")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[
            search_docs,
            get_code_pattern,
            validate_python_syntax,
            suggest_improvement,
        ],
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
        system_prompt="""You are a helpful coding assistant specializing in Python.
Help developers with documentation, code patterns, and best practices.
Always provide practical, working examples when possible.""",
        enable_reflexion=True,
        max_iterations=6,
    )

    queries = [
        "How do I use async/await in Python?",
        "Show me the singleton pattern",
        "What are the best practices for error handling in Python?",
    ]

    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"Developer: {query}")
        print("─" * 70)

        for event in agent.stream(query):
            if isinstance(event, ThinkEvent):
                if event.tool_calls_planned > 0:
                    print(f"🤔 Planning {event.tool_calls_planned} lookup(s)...")
            elif isinstance(event, ToolStartEvent):
                print(
                    f"🔧 {event.tool_name}({list(event.arguments.values())[0] if event.arguments else ''})"
                )
            elif isinstance(event, ToolCompleteEvent):
                # Show first 100 chars of result
                preview = event.result[:100].replace("\n", " ")
                print(f"   → {preview}...")
            elif isinstance(event, ReflectEvent):
                print(f"📊 Confidence: {event.confidence:.0%}")
            elif isinstance(event, TerminateEvent):
                print(f"\n💡 Assistant:\n{event.final_answer}")


if __name__ == "__main__":
    main()
