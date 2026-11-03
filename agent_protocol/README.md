# Agent wire contract

Canonical source: openai-agent-demo/src/agent_protocol/agent.proto. This host copy
must remain byte-identical to that source. InteractionCore contains no wire protocol.

Regenerate from repository root with the pinned orchestrator dependencies:

```bash
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. agent_protocol/agent.proto
```

INSERT and YIELD reuse the active ANSWER ci/epoch; each new ANSWER advances epoch.
END clears Agent context while the RPC stays open. Transport failure is terminal:
the next new ANSWER may establish a fresh RPC/context in the host session.
Never replay actions from the failed RPC.
