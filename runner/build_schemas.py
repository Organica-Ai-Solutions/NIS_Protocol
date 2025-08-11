"""
BuildSpec validation and hashing for Babylon visualization builds.

No external deps. Conservative defaults compatible with NIS runner.
"""

from typing import Any, Dict, List
import json


def _is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def validate_build_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("build_spec must be an object")
    template = str(spec.get("template", "babylon-inline@1.0.0"))
    routes = spec.get("routes", ["/"])
    if not isinstance(routes, list) or not all(isinstance(r, str) for r in routes):
        routes = ["/"]
    features = spec.get("features", {})
    if not isinstance(features, dict):
        features = {}
    # Only allow safe known flags
    neural_background = bool(features.get("neuralBackground", False))
    hud_audit = bool(features.get("hudAudit", True))
    physics = str(features.get("physics", "cannon-es"))
    if physics not in ("cannon-es",):
        physics = "cannon-es"
    out = {
        "template": template,
        "routes": routes,
        "features": {
            "neuralBackground": neural_background,
            "hudAudit": hud_audit,
            "physics": physics,
        },
    }
    return out


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


