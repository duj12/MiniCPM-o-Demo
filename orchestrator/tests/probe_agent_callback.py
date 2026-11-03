"""Retired HTTP/remote-IC probe; use the current Agent gRPC integration suite."""
import sys


def main() -> int:
    print("This probe used retired InteractionCore gRPC / HTTP apply_agent callbacks.")
    print("Use the local-IC and independent AgentService/Interact tests:")
    print("  python -m pytest -q orchestrator/tests/test_agent_integration.py")
    print("  python -m pytest -q orchestrator/tests/test_ic_inprocess_isolation.py")
    print("For the real Agent loopback, follow orchestrator/README.md and supply its source/interpreter.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
