# Deep Agents on OCI — Capabilities

Part of the [Deep Agents documentation](../../langchain_oci/agents/deepagents/README.md).
See also: [Core guide](guide.md) · [Datastores & retrieval](datastores.md) · [Persistence](persistence.md) · [Operations](operations.md)

Everything here is forwarded verbatim to `deepagents.create_deep_agent`;
semantics below were verified against `deepagents` 0.7.5 (installed range:
`>=0.6.1`). Setting any of these forces the deep path (see
[Core guide](guide.md#the-two-build-paths)).

## Built-in tools

On the deep path the agent always gets: `ls`, `read_file`, `write_file`,
`edit_file`, `glob`, `grep` (virtual filesystem, storage per `backend`),
`task` (subagent delegation, present when any sync subagent — including the
auto-added general-purpose one — exists), and `execute` (returns an error
unless the backend implements the sandbox protocol; no shell execution happens
with the default backend). Your `tools=` and the datastore tools are additive
— built-ins are never removed by adding tools.

## Subagents

```python
research_subagent = {
    "name": "literature-scanner",
    "description": "Scans the research datastore and produces raw findings",
    "system_prompt": "You are a focused literature scanner. Only search and list findings.",
    "tools": [...],          # optional; may also override model, middleware, skills…
}
agent = create_deepagents_agent(..., subagents=[research_subagent])
```

Three spec forms: declarative `SubAgent` dicts (compiled for you, invoked via
`task`), `CompiledSubAgent` (pre-built runnable), and `AsyncSubAgent`
(remote/background, identified by `graph_id`). A default `general-purpose`
subagent is added automatically when you don't provide one. Declarative
subagents inherit the parent's `interrupt_on` and `permissions` unless they
define their own.

## Skills

```python
agent = create_deepagents_agent(..., skills=["/skills/project/"])
```

Skill paths are POSIX-style relative to the backend root. With the default
state backend, supply skill files via `invoke(files={...})`; with a
filesystem/store backend they load from that backend. Same-name skills: last
source wins.

## Memory

```python
agent = create_deepagents_agent(..., memory=["/memory/AGENTS.md"])
```

`AGENTS.md`-style files loaded at startup and injected into the system prompt
— persistent instructions, not conversation history (that's the checkpointer's
job — see [Persistence](persistence.md)).

## Human-in-the-loop (`interrupt_on`)

```python
agent = create_deepagents_agent(
    ...,
    checkpointer=checkpointer,   # required to pause/resume
    interrupt_on={"write_file": True, "edit_file": True},
)
result = agent.invoke(inputs, cfg)           # pauses before write_file
# inspect result["__interrupt__"], then approve:
from langgraph.types import Command
result = agent.invoke(Command(resume=[{"type": "accept"}]), cfg)
```

Values can be `True` or an `InterruptOnConfig` (allowed decision types,
description). Applies to the main agent; declarative subagents inherit unless
they override.

## Filesystem permissions

```python
agent = create_deepagents_agent(
    ...,
    permissions=[
        {"path": "/reports/**", "mode": "allow"},
        {"path": "/**", "mode": "deny"},
    ],
)
```

Rules are evaluated in order, **first match wins**, unmatched calls are
allowed. `mode="interrupt"` pauses for approval (auto-installs the HITL
middleware and merges with `interrupt_on`; user-supplied entries win per tool
name). Enforced at the filesystem-tool level, inherited by subagents unless
they define their own. See the upstream `deepagents` docs for the
`FilesystemPermission` type in your installed version.

## Structured output, context, middleware

- `response_format=` — a Pydantic model / TypedDict / schema dict; the final
  state carries the structured response alongside messages.
- `context_schema=` — typed, immutable run-scoped context (passed at
  `invoke(..., context=...)`).
- `middleware=` — your `AgentMiddleware` instances are inserted **after** the
  deepagents base stack (skills → filesystem → subagents → summarization →
  tool-call patching) and **before** the tail stack (prompt caching, memory,
  HITL). Note the base stack includes `SummarizationMiddleware` — long
  conversations are auto-summarized on the deep path.

## Not exposed (documented limitation)

Upstream `create_deep_agent` also accepts `state_schema` (custom
`DeepAgentState` subclass). The `langchain-oci` factory does **not** expose it
today — if you need custom state channels, prefer middleware-scoped state, or
call `deepagents.create_deep_agent` directly with a `ChatOCIGenAI` you build
yourself.
