# Contributing to this repository

We welcome your contributions! There are multiple ways to contribute.

## Table of Contents

- [Opening Issues](#opening-issues)
- [Contributing Code](#contributing-code)
- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

## Opening issues

For bugs or enhancement requests, please file a GitHub issue unless it's
security related. When filing a bug remember that the better written the bug is,
the more likely it is to be fixed. If you think you've found a security
vulnerability, do not raise a GitHub issue and follow the instructions in our
[security policy](./SECURITY.md).

## Contributing code

We welcome your code contributions. Before submitting code via a pull request,
you will need to have signed the [Oracle Contributor Agreement][OCA] (OCA) and
your commits need to include the following line using the name and e-mail
address you used to sign the OCA:

```text
Signed-off-by: Your Name <you@example.org>
```

This can be automatically added to pull requests by committing with `--sign-off`
or `-s`, e.g.

```text
git commit --signoff
```

Only pull requests from committers that can be verified as having signed the OCA
can be accepted.

## Development Setup

### Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/) for dependency management
- OCI CLI configured (`~/.oci/config`)
- Access to OCI Generative AI service (for integration tests)

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/oracle/langchain-oracle.git
cd langchain-oracle

# Install langchain-oci with development dependencies
cd libs/oci
poetry install --with dev,test

# Or install langchain-oracledb
cd libs/oracledb
poetry install --with dev,test
```

### Running Tests

```bash
# Unit tests only (no OCI credentials needed)
poetry run pytest tests/unit

# Integration tests (requires OCI credentials)
export COMPARTMENT_ID="ocid1.compartment.oc1..xxx"
export SERVICE_ENDPOINT="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
poetry run pytest tests/integration

# All tests
poetry run pytest

# With coverage
poetry run pytest --cov=langchain_oci --cov-report=html
```

### Code Quality

Use the Makefile targets for consistent code quality checks:

```bash
# Format code
make format

# Lint (runs ruff check, ruff format --diff, mypy)
make lint

# Run all tests
make test
```

---

## Architecture Overview

### langchain-oci Structure

```
libs/oci/langchain_oci/
├── __init__.py              # Public exports
├── chat_models/
│   ├── oci_generative_ai.py # ChatOCIGenAI, ChatOCIOpenAI
│   ├── oci_data_science.py  # ChatOCIModelDeployment variants
│   └── providers/
│       ├── base.py          # Provider base class
│       ├── cohere.py        # CohereProvider
│       └── generic.py       # GenericProvider, MetaProvider, GeminiProvider
├── embeddings/
│   ├── oci_generative_ai.py # OCIGenAIEmbeddings
│   └── image.py             # Image embedding utilities
├── agents/
│   └── react.py             # create_oci_agent()
├── llms/
│   └── oci_generative_ai.py # Legacy OCIGenAI
├── utils/
│   └── vision.py            # Vision utilities
└── common/
    ├── auth.py              # Authentication
    └── utils.py             # Shared utilities
```

### Provider Pattern

Providers abstract model-specific behaviors:

```
ChatOCIGenAI
    │
    ├── model_id="meta.llama-*" → MetaProvider
    ├── model_id="cohere.*"     → CohereProvider
    ├── model_id="google.*"     → GeminiProvider
    └── model_id="xai.*"        → GenericProvider
```

Each provider handles:
- Message format conversion
- Tool calling format
- Response parsing
- Streaming events

---

---

## Testing

### Test Categories

| Type | Location | Requires OCI |
|------|----------|--------------|
| Unit | `tests/unit/` | No |
| Integration | `tests/integration/` | Yes |

### Writing Tests

```python
# Unit test example
def test_vision_model_detection():
    from langchain_oci import is_vision_model
    assert is_vision_model("meta.llama-3.2-90b-vision-instruct")
    assert not is_vision_model("meta.llama-3.3-70b-instruct")

# Integration test example (requires OCI)
@pytest.mark.integration
def test_chat_invoke():
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=os.environ["SERVICE_ENDPOINT"],
        compartment_id=os.environ["COMPARTMENT_ID"],
    )
    response = llm.invoke("Hello")
    assert response.content
```

---

## Pull request process

1. Ensure there is an issue created to track and discuss the fix or enhancement
   you intend to submit.
1. Fork this repository.
1. Create a branch in your fork to implement the changes. We recommend using
   the issue number as part of your branch name, e.g. `1234-fixes`.
1. Ensure that any documentation is updated with the changes that are required
   by your change.
1. Ensure that any samples are updated if the base image has been changed.
1. **Run tests and linting** before submitting.
1. Submit the pull request. *Do not leave the pull request blank*. Explain exactly
   what your changes are meant to do and provide simple steps on how to validate.
   your changes. Ensure that you reference the issue you created as well.
1. We will assign the pull request to 2-3 people for review before it is merged.

## Code of conduct

Follow the [Golden Rule](https://en.wikipedia.org/wiki/Golden_Rule). If you'd
like more specific guidelines, see the [Contributor Covenant Code of Conduct][COC].

[OCA]: https://oca.opensource.oracle.com
[COC]: https://www.contributor-covenant.org/version/1/4/code-of-conduct/
