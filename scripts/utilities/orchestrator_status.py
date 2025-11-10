#!/usr/bin/env python3
"""Emit multiprovider + orchestration status for MCP clients."""

import json
import os
from pathlib import Path

from ..core.agent_orchestrator import NISAgentOrchestrator
from ..utils.env_config import EnvConfig

ROOT = Path(__file__).resolve().parents[2]


def load_provider_registry() -> dict:
    registry_path = os.getenv(
        "NIS_PROVIDER_REGISTRY",
        ROOT / "configs" / "provider_registry.yaml",
    )
    registry_path = Path(registry_path)
    if registry_path.exists():
        import yaml

        with registry_path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def snapshot_orchestrator() -> dict:
    orchestrator = NISAgentOrchestrator()
    orchestrator.load_agents()
    return {
        "agents": {
            agent_id: {
                "name": definition.name,
                "type": definition.agent_type.value,
                "priority": definition.priority,
                "description": definition.description,
            }
            for agent_id, definition in orchestrator.agents.items()
        }
    }


def snapshot_config() -> dict:
    env = EnvConfig()
    llm_config = env.get_llm_config()
    return {
        "default_provider": llm_config.get("agent_llm_config", {}).get("default_provider"),
        "fallback_to_mock": llm_config.get("agent_llm_config", {}).get("fallback_to_mock"),
        "providers": list(llm_config.get("providers", {}).keys()),
    }


def main() -> None:
    status = {
        "orchestrator": snapshot_orchestrator(),
        "providers": load_provider_registry(),
        "llm_config": snapshot_config(),
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
