"""
Visualization intent schemas and lightweight validators for the NIS runner.

No external dependencies. These functions validate and clamp payloads to
safe ranges compatible with the frontend Babylon.js integration.
"""

from typing import Any, Dict, List, Tuple, Union

Vec3 = Tuple[float, float, float]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf"))


def _clamp(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def _clamp_vec3(vec: List[float], min_value: float, max_value: float) -> List[float]:
    if not isinstance(vec, (list, tuple)) or len(vec) != 3:
        raise ValueError("vec3 must be a list/tuple of three numbers")
    x, y, z = vec
    if not (_is_number(x) and _is_number(y) and _is_number(z)):
        raise ValueError("vec3 components must be numbers")
    return [
        _clamp(float(x), min_value, max_value),
        _clamp(float(y), min_value, max_value),
        _clamp(float(z), min_value, max_value),
    ]


def validate_scene_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clamp a scene_spec payload. Returns the sanitized payload.
    Raises ValueError on invalid shape.
    """
    if not isinstance(payload, dict):
        raise ValueError("scene_spec payload must be an object")

    version = str(payload.get("version", "1")).strip()
    if version not in ("1", "1.0"):
        version = "1"
    payload["version"] = version

    payload["engine"] = "babylon"

    physics = payload.get("physics")
    if physics is not None:
        if not isinstance(physics, dict):
            raise ValueError("physics must be an object")
        # Engine forward-compat: allow cannon-es (default) or havok-v2
        engine = str(physics.get("engine", "cannon-es")).lower()
        if engine not in ("cannon-es", "havok-v2"):
            engine = "cannon-es"
        physics["engine"] = engine
        # Optional API selector for frontend (v1: CannonJSPlugin, v2: Havok V2)
        physics_api = physics.get("physics_api")
        if physics_api is not None:
            pa = str(physics_api).lower()
            if pa not in ("v1", "v2"):
                pa = "v1"
            physics["physics_api"] = pa
        gravity = physics.get("gravity")
        if gravity is None:
            physics["gravity"] = [0.0, -9.81, 0.0]
        else:
            physics["gravity"] = _clamp_vec3(list(gravity), -1000.0, 1000.0)
        payload["physics"] = physics

    camera = payload.get("camera")
    if not isinstance(camera, dict) or "id" not in camera or "type" not in camera:
        raise ValueError("camera with id and type is required")
    cam_type = str(camera.get("type")).lower()
    if cam_type in ("arc", "arcrotate"):
        camera["type"] = "arc"
        # Normalize target and radius if present
        if "target" in camera:
            camera["target"] = _clamp_vec3(list(camera["target"]), -5000.0, 5000.0)
        if "radius" in camera and _is_number(camera["radius"]):
            camera["radius"] = _clamp(float(camera["radius"]), 0.1, 5000.0)
        if "alpha" in camera and _is_number(camera["alpha"]):
            camera["alpha"] = float(camera["alpha"])
        if "beta" in camera and _is_number(camera["beta"]):
            camera["beta"] = float(camera["beta"])
    elif cam_type in ("universal", "free"):
        camera["type"] = "universal"
        if "position" not in camera:
            raise ValueError("universal camera requires position")
        camera["position"] = _clamp_vec3(list(camera["position"]), -5000.0, 5000.0)
        if "target" in camera:
            camera["target"] = _clamp_vec3(list(camera["target"]), -5000.0, 5000.0)
    else:
        raise ValueError("camera.type must be arc/arcRotate or universal/free")
    payload["camera"] = camera

    lights = payload.get("lights", [])
    if not isinstance(lights, list):
        raise ValueError("lights must be an array")
    sanitized_lights: List[Dict[str, Any]] = []
    for light in lights:
        if not isinstance(light, dict) or "id" not in light or "type" not in light:
            continue
        ltype = str(light["type"]).lower()
        if ltype == "hemispheric":
            if "direction" in light:
                light["direction"] = _clamp_vec3(list(light["direction"]), -1e6, 1e6)
            if "intensity" in light and _is_number(light["intensity"]):
                light["intensity"] = _clamp(float(light["intensity"]), 0.0, 10.0)
            sanitized_lights.append(light)
        elif ltype == "directional":
            if "direction" not in light:
                continue
            light["direction"] = _clamp_vec3(list(light["direction"]), -1e6, 1e6)
            if "intensity" in light and _is_number(light["intensity"]):
                light["intensity"] = _clamp(float(light["intensity"]), 0.0, 10.0)
            sanitized_lights.append(light)
        elif ltype == "point":
            if "position" not in light:
                continue
            light["position"] = _clamp_vec3(list(light["position"]), -5000.0, 5000.0)
            if "intensity" in light and _is_number(light["intensity"]):
                light["intensity"] = _clamp(float(light["intensity"]), 0.0, 10.0)
            sanitized_lights.append(light)
    payload["lights"] = sanitized_lights

    materials = payload.get("materials")
    if materials is not None and not isinstance(materials, list):
        raise ValueError("materials must be an array if provided")
    # Materials and optional presets
    payload["materials"] = materials or []
    material_presets = payload.get("material_presets") or payload.get("materialPresets")
    if material_presets is not None:
        if not isinstance(material_presets, (list, dict)):
            raise ValueError("material_presets must be an array or object if provided")
        # support either list of {name, friction, restitution, density} or dict name->props
        def _sanitize_props(props: Dict[str, Any]) -> Dict[str, Any]:
            out = {}
            if not isinstance(props, dict):
                return out
            if "friction" in props and _is_number(props["friction"]):
                out["friction"] = _clamp(float(props["friction"]), 0.0, 1.0)
            if "restitution" in props and _is_number(props["restitution"]):
                out["restitution"] = _clamp(float(props["restitution"]), 0.0, 1.0)
            if "density" in props and _is_number(props["density"]):
                out["density"] = _clamp(float(props["density"]), 0.0, 50000.0)
            return out
        presets_out: Dict[str, Any] = {}
        if isinstance(material_presets, dict):
            for name, props in material_presets.items():
                presets_out[str(name)] = _sanitize_props(props)
        else:
            for item in material_presets:
                if isinstance(item, dict) and "name" in item:
                    presets_out[str(item["name"])] = _sanitize_props(item)
        payload["material_presets"] = presets_out

    meshes = payload.get("meshes", [])
    if not isinstance(meshes, list):
        raise ValueError("meshes must be an array")
    if len(meshes) > 200:
        raise ValueError("meshes exceed maximum of 200")
    sanitized_meshes: List[Dict[str, Any]] = []
    for mesh in meshes:
        if not isinstance(mesh, dict) or "id" not in mesh or "kind" not in mesh:
            continue
        kind = str(mesh["kind"]).lower()
        if kind not in ("box", "sphere", "plane", "ground"):
            continue
        # Clamp transforms
        if "position" in mesh:
            mesh["position"] = _clamp_vec3(list(mesh["position"]), -5000.0, 5000.0)
        if "rotation" in mesh:
            mesh["rotation"] = [float(x) for x in mesh["rotation"]]
        if "scaling" in mesh:
            mesh["scaling"] = _clamp_vec3(list(mesh["scaling"]), 0.01, 1000.0)
        # Clamp sizes
        for size_key in ("size", "diameter", "width", "height"):
            if size_key in mesh and _is_number(mesh[size_key]):
                mesh[size_key] = _clamp(float(mesh[size_key]), 0.01, 1000.0)
        if "segments" in mesh and _is_number(mesh["segments"]):
            mesh["segments"] = int(_clamp(float(mesh["segments"]), 3, 256))
        if "subdivisions" in mesh and _is_number(mesh["subdivisions"]):
            mesh["subdivisions"] = int(_clamp(float(mesh["subdivisions"]), 1, 256))
        # Physics
        physics = mesh.get("physics")
        if physics is not None:
            if not isinstance(physics, dict) or "impostor" not in physics:
                physics = None
            else:
                impostor = str(physics.get("impostor"))
                if impostor not in (
                    "BoxImpostor",
                    "SphereImpostor",
                    "PlaneImpostor",
                    "MeshImpostor",
                    "NoImpostor",
                ):
                    impostor = "NoImpostor"
                physics["impostor"] = impostor
                if "mass" in physics and _is_number(physics["mass"]):
                    physics["mass"] = _clamp(float(physics["mass"]), 0.0, 1000.0)
                if "friction" in physics and _is_number(physics["friction"]):
                    physics["friction"] = _clamp(float(physics["friction"]), 0.0, 1.0)
                if "restitution" in physics and _is_number(physics["restitution"]):
                    physics["restitution"] = _clamp(float(physics["restitution"]), 0.0, 1.0)
        mesh["physics"] = physics
        sanitized_meshes.append(mesh)
    payload["meshes"] = sanitized_meshes

    # Optional constraints (joints)
    constraints = payload.get("constraints")
    if constraints is not None:
        if not isinstance(constraints, list):
            raise ValueError("constraints must be an array if provided")
        sanitized_constraints: List[Dict[str, Any]] = []
        for c in constraints:
            if not isinstance(c, dict):
                continue
            ctype = str(c.get("type", "")).lower()
            if ctype not in ("hinge", "slider", "ball"):
                continue
            body_a = c.get("bodyA") or c.get("body_a")
            body_b = c.get("bodyB") or c.get("body_b")
            if not (isinstance(body_a, str) and isinstance(body_b, str)):
                continue
            axis = c.get("axis")
            if axis is not None:
                try:
                    axis = _clamp_vec3(list(axis), -1.0, 1.0)
                except Exception:
                    axis = None
            limits = c.get("limits")
            if limits is not None and isinstance(limits, dict):
                out_limits: Dict[str, float] = {}
                for k in ("min", "max", "lower", "upper"):
                    if k in limits and _is_number(limits[k]):
                        out_limits[k] = float(limits[k])
                limits = out_limits
            sanitized_constraints.append({
                "type": ctype,
                "bodyA": body_a,
                "bodyB": body_b,
                **({"axis": axis} if axis is not None else {}),
                **({"limits": limits} if limits else {}),
            })
        payload["constraints"] = sanitized_constraints

    # Environment optional
    env = payload.get("environment")
    if env is not None and not isinstance(env, dict):
        raise ValueError("environment must be an object if provided")
    payload["environment"] = env

    # Optional units
    units = payload.get("units")
    if units is not None:
        if not isinstance(units, dict):
            raise ValueError("units must be an object if provided")
        length = str(units.get("length", "m"))
        mass = str(units.get("mass", "kg"))
        time_u = str(units.get("time", "s"))
        # Accept common SI; pass-through otherwise but default to SI
        def _allow(v: str, allowed: List[str], default: str) -> str:
            return v if v in allowed else default
        units_out = {
            "length": _allow(length, ["m", "cm", "mm"], "m"),
            "mass": _allow(mass, ["kg", "g"], "kg"),
            "time": _allow(time_u, ["s", "ms"], "s"),
        }
        payload["units"] = units_out

    return payload


def validate_step(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("step payload must be an object")
    version = str(payload.get("version", "1")).strip()
    if version not in ("1", "1.0"):
        version = "1"
    payload["version"] = version

    dt = payload.get("dt")
    if dt is not None and _is_number(dt):
        payload["dt"] = _clamp(float(dt), 0.0, 0.033)

    ops = payload.get("ops", [])
    if not isinstance(ops, list):
        raise ValueError("ops must be an array")
    sanitized_ops: List[Dict[str, Any]] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        target = op.get("target", {})
        action = op.get("action", {})
        if not isinstance(target, dict) or not isinstance(action, dict):
            continue
        ttype = target.get("type")
        tid = target.get("id")
        if ttype not in ("mesh", "camera", "light") or not isinstance(tid, str):
            continue
        atype = action.get("type")
        if atype == "translate" or atype == "rotate" or atype == "setPosition":
            val = action.get("value")
            if val is None:
                continue
            action["value"] = _clamp_vec3(list(val), -5000.0, 5000.0)
        elif atype == "applyImpulse":
            impulse = action.get("impulse")
            if impulse is None:
                continue
            vec = _clamp_vec3(list(impulse), -1e4, 1e4)
            # Cap vector magnitude to 1e4
            # Simple component-wise cap suffices for safety
            action["impulse"] = vec
            if "contactPoint" in action:
                action["contactPoint"] = _clamp_vec3(list(action["contactPoint"]), -5000.0, 5000.0)
        elif atype == "setTarget":
            val = action.get("value")
            if val is None:
                continue
            action["value"] = _clamp_vec3(list(val), -5000.0, 5000.0)
        elif atype == "arcDelta":
            for key in ("dAlpha", "dBeta"):
                if key in action and _is_number(action[key]):
                    action[key] = float(action[key])
            if "dRadius" in action and _is_number(action["dRadius"]):
                action["dRadius"] = _clamp(float(action["dRadius"]), -1000.0, 1000.0)
        elif atype == "setIntensity":
            if _is_number(action.get("value", None)):
                action["value"] = _clamp(float(action["value"]), 0.0, 10.0)
            else:
                continue
        else:
            continue
        sanitized_ops.append({"target": {"type": ttype, "id": tid}, "action": action})

    payload["ops"] = sanitized_ops
    return payload


def validate_chart(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("chart payload must be an object")
    # Accept simple or multi-series
    if "series" not in payload:
        raise ValueError("chart requires series")
    series = payload["series"]
    if not isinstance(series, list):
        raise ValueError("chart.series must be an array")
    # If points are of shape {t, y}
    if series and isinstance(series[0], dict) and "points" not in series[0] and "t" in series[0]:
        sanitized = []
        for p in series:
            if not isinstance(p, dict):
                continue
            t = float(p.get("t", 0.0))
            y = float(p.get("y", 0.0))
            sanitized.append({"t": t, "y": y})
        payload["series"] = sanitized
        return payload
    # Multi-series
    sanitized_series = []
    for s in series:
        if not isinstance(s, dict) or "points" not in s:
            continue
        name = str(s.get("name", "series"))
        points = []
        for p in s.get("points", []):
            if not isinstance(p, dict):
                continue
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
            points.append({"x": x, "y": y})
        sanitized_series.append({"name": name, "points": points, "color": s.get("color")})
    payload["series"] = sanitized_series
    return payload


def validate_overlay(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("overlay payload must be an object")
    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        raise ValueError("overlay.elements must be an array")
    sanitized_elements: List[Dict[str, Any]] = []
    for el in elements:
        if not isinstance(el, dict) or "type" not in el or "text" not in el:
            continue
        etype = str(el.get("type"))
        if etype not in ("text", "metric"):
            continue
        pos = el.get("position")
        if pos is not None and isinstance(pos, dict):
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            el["position"] = {"x": x, "y": y}
        sanitized_elements.append({"type": etype, "text": str(el["text"]), "position": el.get("position")})
    payload["elements"] = sanitized_elements
    return payload


def wrap_response(intent_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if intent_type not in ("scene_spec", "step", "chart", "overlay"):
        raise ValueError("invalid intent type")
    return {"type": intent_type, "payload": payload}


