# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration test: Real LangChain agent with Gemini that executes code.

This test demonstrates a real agentic workflow where Gemini:
1. Writes Python code to disk
2. Runs linting with ruff
3. Executes pytest tests

All tool executions are REAL - files are written, commands are run.

Prerequisites:
    - OCI authentication configured
    - ruff installed (pip install ruff)
    - pytest installed

Running:
    ```bash
    export OCI_COMPARTMENT_ID="your-compartment"
    export OCI_CONFIG_PROFILE="your-profile"
    export OCI_AUTH_TYPE="SECURITY_TOKEN"
    pytest tests/integration_tests/chat_models/test_gemini_real_agent.py -v -s
    ```
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_oci.chat_models import ChatOCIGenAI

# Test configuration
COMPARTMENT_ID = os.getenv(
    "OCI_COMPARTMENT_ID",
    "ocid1.compartment.oc1..aaaaaaaahpwdqajkowoqh4d2q66l4umblh32vbjkh3qnpfdmzcrb7am6jyuq",
)
SERVICE_ENDPOINT = os.getenv(
    "OCI_GENAI_ENDPOINT",
    "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
)
AUTH_TYPE = os.getenv("OCI_AUTH_TYPE", "SECURITY_TOKEN")
AUTH_PROFILE = os.getenv("OCI_CONFIG_PROFILE", "BOAT-OC1")

# Find Python and ruff executables
PYTHON_EXE = sys.executable
RUFF_EXE = shutil.which("ruff")


# ============================================================================
# REAL TOOLS - These actually execute!
# ============================================================================


@tool
def write_python_file(filepath: str, content: str) -> str:
    """Write Python code to a file.

    Args:
        filepath: Path where to write the file
        content: Python code content to write
    """
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def read_file(filepath: str) -> str:
    """Read contents of a file.

    Args:
        filepath: Path to the file to read
    """
    try:
        with open(filepath, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def run_python_code(filepath: str) -> str:
    """Execute a Python file and return the output.

    Args:
        filepath: Path to the Python file to execute
    """
    try:
        result = subprocess.run(
            [PYTHON_EXE, filepath], capture_output=True, text=True, timeout=30
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        output += f"Exit code: {result.returncode}"
        return output if output.strip() else "No output (exit code 0)"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out after 30 seconds"
    except Exception as e:
        return f"Error executing: {e}"


@tool
def run_ruff_check(filepath: str) -> str:
    """Run ruff linter on a Python file to check for issues.

    Args:
        filepath: Path to the Python file to lint
    """
    if not RUFF_EXE:
        return "ruff not found in PATH"
    try:
        result = subprocess.run(
            [RUFF_EXE, "check", filepath], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return "No linting issues found!"
        return f"Linting issues:\n{result.stdout}{result.stderr}"
    except Exception as e:
        return f"Error running ruff: {e}"


@tool
def run_ruff_format(filepath: str) -> str:
    """Format a Python file using ruff formatter.

    Args:
        filepath: Path to the Python file to format
    """
    if not RUFF_EXE:
        return "ruff not found in PATH"
    try:
        result = subprocess.run(
            [RUFF_EXE, "format", filepath], capture_output=True, text=True, timeout=30
        )
        return f"Formatted {filepath}\n{result.stdout}{result.stderr}"
    except Exception as e:
        return f"Error formatting: {e}"


@tool
def run_python_tests(filepath: str) -> str:
    """Run pytest on a Python test file.

    Args:
        filepath: Path to the test file to run
    """
    try:
        result = subprocess.run(
            [PYTHON_EXE, "-m", "pytest", filepath, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return f"{result.stdout}\n{result.stderr}\nExit code: {result.returncode}"
    except Exception as e:
        return f"Error running tests: {e}"


# ============================================================================
# AGENT IMPLEMENTATION
# ============================================================================


def run_agent(task: str, llm: ChatOCIGenAI, max_iterations: int = 10) -> "list[str]":
    """Run an agent loop with real tool execution.

    Returns list of tool names that were executed.
    """
    tools = [
        write_python_file,
        read_file,
        run_python_code,
        run_ruff_check,
        run_ruff_format,
        run_python_tests,
    ]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [HumanMessage(content=task)]
    executed_tools = []

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for idx, tool_call in enumerate(response.tool_calls):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call.get("id") or f"call_{i}_{idx}"

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                executed_tools.append(tool_name)
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
            )

    return executed_tools


# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.requires("oci")
class TestGeminiRealAgent:
    """Test real agent workflows with Gemini."""

    @pytest.fixture
    def llm(self) -> ChatOCIGenAI:
        """Create LLM instance."""
        return ChatOCIGenAI(
            model_id="google.gemini-2.5-flash",
            compartment_id=COMPARTMENT_ID,
            service_endpoint=SERVICE_ENDPOINT,
            auth_type=AUTH_TYPE,
            auth_profile=AUTH_PROFILE,
            model_kwargs={"temperature": 0.0, "max_tokens": 2048},
        )

    @pytest.fixture
    def work_dir(self) -> str:
        """Create temporary working directory."""
        dir_path = tempfile.mkdtemp(prefix="gemini_agent_test_")
        yield dir_path
        # Cleanup after test
        shutil.rmtree(dir_path, ignore_errors=True)

    def test_agent_writes_and_tests_calculator(
        self, llm: ChatOCIGenAI, work_dir: str
    ) -> None:
        """Test agent can write code, lint it, and run tests.

        This is a REAL test - files are actually written and commands executed.
        """
        task = f"""
        You are a coding assistant. Please do the following:

        1. Write a Python file at {work_dir}/calculator.py with a Calculator class:
           - add(a, b) method
           - subtract(a, b) method
           - multiply(a, b) method
           - divide(a, b) method (handle division by zero with ValueError)

        2. Write a test file at {work_dir}/test_calculator.py with pytest tests

        3. Run ruff to check code quality

        4. Run the tests

        Use the tools provided.
        """

        executed_tools = run_agent(task, llm)

        # Verify tools were actually called
        assert "write_python_file" in executed_tools, "Should have written files"

        # Verify files were actually created
        calc_file = os.path.join(work_dir, "calculator.py")
        test_file = os.path.join(work_dir, "test_calculator.py")

        assert os.path.exists(calc_file), f"calculator.py should exist at {calc_file}"
        assert os.path.exists(test_file), (
            f"test_calculator.py should exist at {test_file}"
        )

        # Verify calculator.py has expected content
        with open(calc_file) as f:
            calc_content = f.read()
        assert "class Calculator" in calc_content, "Should have Calculator class"
        assert "def add" in calc_content, "Should have add method"
        assert "def divide" in calc_content, "Should have divide method"

        # Verify test file has expected content
        with open(test_file) as f:
            test_content = f.read()
        assert "import pytest" in test_content or "def test_" in test_content, (
            "Should have pytest tests"
        )

    def test_agent_generates_fibonacci(self, llm: ChatOCIGenAI, work_dir: str) -> None:
        """Test agent can write and execute a fibonacci function."""
        task = f"""
        Write a Python file at {work_dir}/fib.py with:
        1. A fibonacci(n) function that returns the nth fibonacci number
        2. A main block that prints fibonacci(10)

        Then run the file and tell me the output.
        Use the tools provided.
        """

        executed_tools = run_agent(task, llm)

        assert "write_python_file" in executed_tools
        assert "run_python_code" in executed_tools

        fib_file = os.path.join(work_dir, "fib.py")
        assert os.path.exists(fib_file), "fib.py should exist"

        with open(fib_file) as f:
            content = f.read()
        assert "def fibonacci" in content or "def fib" in content


@pytest.mark.requires("oci")
class TestGeminiRealAgentCrossModel:
    """Test real agent works across different models."""

    @pytest.fixture
    def work_dir(self) -> str:
        """Create temporary working directory."""
        dir_path = tempfile.mkdtemp(prefix="gemini_agent_test_")
        yield dir_path
        shutil.rmtree(dir_path, ignore_errors=True)

    @pytest.mark.parametrize(
        "model_id",
        [
            pytest.param("google.gemini-2.5-flash", id="gemini-2.5-flash"),
            pytest.param("google.gemini-2.5-pro", id="gemini-2.5-pro"),
        ],
    )
    def test_simple_file_write(self, model_id: str, work_dir: str) -> None:
        """Test basic file writing works across Gemini models."""
        llm = ChatOCIGenAI(
            model_id=model_id,
            compartment_id=COMPARTMENT_ID,
            service_endpoint=SERVICE_ENDPOINT,
            auth_type=AUTH_TYPE,
            auth_profile=AUTH_PROFILE,
            model_kwargs={"temperature": 0.0, "max_tokens": 1024},
        )

        task = f"""
        Write a Python file at {work_dir}/hello.py that prints "Hello, World!"
        Use the write_python_file tool.
        """

        executed_tools = run_agent(task, llm, max_iterations=3)

        assert "write_python_file" in executed_tools
        hello_file = os.path.join(work_dir, "hello.py")
        assert os.path.exists(hello_file)

        with open(hello_file) as f:
            content = f.read()
        assert "Hello" in content or "print" in content
