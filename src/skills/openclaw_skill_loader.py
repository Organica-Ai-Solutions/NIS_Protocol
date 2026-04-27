"""
OpenClaw-compatible Skill Loader for NIS Protocol

Parses SKILL.md files with YAML frontmatter (name, description + full OpenClaw metadata)
and exposes skill metadata for agent routing with progressive disclosure.

Compatible with OpenClaw skill format for cross-platform skill sharing.

SKILL.md frontmatter schema (subset of OpenClaw):
  name: <string>              (required)
  description: <string>       (required — this is the primary trigger)
  metadata:
    openclaw:
      emoji: <string>
      always: <bool>          — always include in context (even without trigger)
      skillKey: <string>      — stable identifier across renames
      primaryEnv: <string>    — primary env var required
      homepage: <string>
      os: [linux, darwin, win32]
      requires:
        bins: [binary1, binary2]
        anyBins: [binary1, binary2]   — need at least one
        env: [ENV_VAR1]
        config: [config_path]
      install:
        - kind: brew|node|go|uv|download
          package: <string>
          ...

Copyright 2025 Organica AI Solutions
"""

import os
import platform
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ──────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────

@dataclass
class OpenClawSkillMeta:
    """Parsed openclaw metadata block from frontmatter."""
    emoji: Optional[str] = None
    always: bool = False
    skill_key: Optional[str] = None
    primary_env: Optional[str] = None
    homepage: Optional[str] = None
    os_filter: Optional[List[str]] = None        # None = any OS
    requires_bins: List[str] = field(default_factory=list)
    requires_any_bins: List[str] = field(default_factory=list)
    requires_env: List[str] = field(default_factory=list)
    requires_config: List[str] = field(default_factory=list)
    install: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillMetadata:
    """Parsed skill metadata from SKILL.md frontmatter."""
    name: str
    description: str
    openclaw: Optional[OpenClawSkillMeta] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillEntry:
    """Full skill entry with metadata and resource paths."""
    skill: SkillMetadata
    skill_dir: str
    skill_md_path: str
    scripts: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    assets: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────
#  Frontmatter parsing
# ──────────────────────────────────────────────────────────

def _parse_raw_frontmatter(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract YAML frontmatter and body from markdown.
    Returns (frontmatter_dict | None, body_str).
    """
    match = re.match(r'^---[ \t]*\n(.*?)\n---[ \t]*\n(.*)$', content, re.DOTALL)
    if not match:
        # Try without trailing newline after closing ---
        match = re.match(r'^---[ \t]*\n(.*?)\n---[ \t]*$', content, re.DOTALL)
        if not match:
            return None, content
        fm_raw, body = match.group(1), ""
    else:
        fm_raw, body = match.group(1), match.group(2)

    if YAML_AVAILABLE:
        try:
            data = yaml.safe_load(fm_raw)
            return data if isinstance(data, dict) else {}, body
        except Exception:
            pass

    # Fallback: minimal key: value regex parsing (top-level only)
    data: Dict[str, Any] = {}
    for line in fm_raw.splitlines():
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$', line)
        if m:
            val = m.group(2).strip().strip('"\'')
            data[m.group(1)] = val
    return data, body


def _normalise_str_list(val: Any) -> List[str]:
    """Normalise YAML value to a list of strings."""
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, (list, tuple)):
        return [str(v).strip() for v in val if str(v).strip()]
    return []


def _parse_openclaw_meta(raw_fm: Dict[str, Any]) -> Optional[OpenClawSkillMeta]:
    """
    Parse the metadata.openclaw block from raw frontmatter.
    Returns None if no openclaw block found.
    """
    meta_block = raw_fm.get("metadata")
    if not isinstance(meta_block, dict):
        return None
    oc_block = meta_block.get("openclaw")
    if not isinstance(oc_block, dict):
        return None

    requires_block = oc_block.get("requires") or {}
    requires = requires_block if isinstance(requires_block, dict) else {}

    install_raw = oc_block.get("install") or []
    install: List[Dict[str, Any]] = []
    if isinstance(install_raw, list):
        for spec in install_raw:
            if isinstance(spec, dict) and spec.get("kind"):
                install.append(spec)

    return OpenClawSkillMeta(
        emoji=oc_block.get("emoji") if isinstance(oc_block.get("emoji"), str) else None,
        always=bool(oc_block.get("always", False)),
        skill_key=oc_block.get("skillKey") if isinstance(oc_block.get("skillKey"), str) else None,
        primary_env=oc_block.get("primaryEnv") if isinstance(oc_block.get("primaryEnv"), str) else None,
        homepage=oc_block.get("homepage") if isinstance(oc_block.get("homepage"), str) else None,
        os_filter=_normalise_str_list(oc_block.get("os")) or None,
        requires_bins=_normalise_str_list(requires.get("bins")),
        requires_any_bins=_normalise_str_list(requires.get("anyBins")),
        requires_env=_normalise_str_list(requires.get("env")),
        requires_config=_normalise_str_list(requires.get("config")),
        install=install,
        raw=oc_block,
    )


# ──────────────────────────────────────────────────────────
#  Eligibility / filtering (mirrors OpenClaw shouldIncludeSkill)
# ──────────────────────────────────────────────────────────

def _current_platform() -> str:
    """Return normalized platform string matching OpenClaw: linux | darwin | win32."""
    p = platform.system().lower()
    if p == "windows":
        return "win32"
    if p == "darwin":
        return "darwin"
    return "linux"


def _check_eligibility(entry: SkillEntry) -> bool:
    """
    Check whether a skill is eligible to run on this host.
    Mirrors OpenClaw shouldIncludeSkill logic:
      - OS filter
      - required binaries (all must exist)
      - anyBins (at least one must exist)
      - required env vars (must be set and non-empty)
    """
    oc = entry.skill.openclaw
    if oc is None:
        return True  # No constraints — always eligible

    # OS filter
    if oc.os_filter:
        cur = _current_platform()
        if cur not in oc.os_filter:
            return False

    # Required binaries (all must be present)
    for bin_name in oc.requires_bins:
        if not shutil.which(bin_name):
            return False

    # anyBins — at least one must be present
    if oc.requires_any_bins:
        if not any(shutil.which(b) for b in oc.requires_any_bins):
            return False

    # Required env vars
    for env_var in oc.requires_env:
        if not os.environ.get(env_var, "").strip():
            return False

    # Required config paths
    for cfg_path in oc.requires_config:
        p = Path(cfg_path).expanduser()
        if not p.exists():
            return False

    return True


# ──────────────────────────────────────────────────────────
#  Skill directory resolution
# ──────────────────────────────────────────────────────────

def _resolve_skill_dirs() -> List[Path]:
    """Resolve ordered list of directories to search for skills."""
    dirs: List[Path] = []

    # 1. NIS Protocol skills/ folder (repo root)
    nis_root = Path(__file__).resolve().parent.parent.parent
    nis_skills = nis_root / "skills"
    if nis_skills.is_dir():
        dirs.append(nis_skills)

    # 2. OpenClaw workspace skills (~/.openclaw/workspace/skills)
    openclaw_ws = Path.home() / ".openclaw" / "workspace" / "skills"
    if openclaw_ws.is_dir():
        dirs.append(openclaw_ws)

    # 3. Environment override (NIS_SKILLS_DIR)
    env_dir = os.environ.get("NIS_SKILLS_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if p.is_dir() and p not in dirs:
            dirs.append(p)

    return dirs


def _list_skill_dirs(root: Path) -> List[str]:
    """Return names of immediate child dirs that contain a SKILL.md file."""
    result: List[str] = []
    try:
        for entry in root.iterdir():
            if entry.is_dir() and not entry.name.startswith(".") and entry.name != "node_modules":
                if (entry / "SKILL.md").is_file():
                    result.append(entry.name)
    except OSError:
        pass
    return sorted(result)


# ──────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────

def load_skill(skill_dir: Path) -> Optional[SkillEntry]:
    """
    Load and parse a single skill from a directory.
    Returns None if the directory has no valid SKILL.md.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return None
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    raw_fm, _body = _parse_raw_frontmatter(content)
    if not raw_fm:
        return None

    name = str(raw_fm.get("name") or raw_fm.get("title") or skill_dir.name).strip()
    desc = str(raw_fm.get("description") or "").strip()

    oc_meta = _parse_openclaw_meta(raw_fm)
    skill_meta = SkillMetadata(name=name, description=desc, openclaw=oc_meta, raw=raw_fm)

    # Discover resource files
    def _collect(subdir: str) -> List[str]:
        d = skill_dir / subdir
        if not d.is_dir():
            return []
        return sorted(str(f.relative_to(skill_dir)) for f in d.rglob("*") if f.is_file())

    return SkillEntry(
        skill=skill_meta,
        skill_dir=str(skill_dir),
        skill_md_path=str(skill_md),
        scripts=_collect("scripts"),
        references=_collect("references"),
        assets=_collect("assets"),
    )


def list_skills(
    skill_filter: Optional[List[str]] = None,
    check_eligibility: bool = True,
) -> List[SkillEntry]:
    """
    List all available skills from configured directories.

    Args:
        skill_filter:       Optional list of skill names to include.
        check_eligibility:  If True, skip skills whose OS/binary/env requirements
                            are not met on this host.
    Returns:
        Deduplicated list of SkillEntry (first occurrence wins when names collide).
    """
    seen: Dict[str, SkillEntry] = {}
    for root in _resolve_skill_dirs():
        for dir_name in _list_skill_dirs(root):
            # Name filter
            if skill_filter and dir_name not in skill_filter:
                continue
            skill_path = root / dir_name
            entry = load_skill(skill_path)
            if not entry:
                continue
            # Also check by skill.name (may differ from dir_name)
            if skill_filter and entry.skill.name not in skill_filter and dir_name not in skill_filter:
                continue
            if entry.skill.name in seen:
                continue  # First directory wins
            # Eligibility check
            if check_eligibility and not _check_eligibility(entry):
                continue
            seen[entry.skill.name] = entry
    return list(seen.values())


def get_skill_content(
    name: str,
    include_body: bool = True,
    check_eligibility: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Get full skill content by skill name or directory name.

    Returns dict with:
      name, description, emoji, always, os, requires, install,
      scripts, references, assets, body (if include_body)
    """
    for root in _resolve_skill_dirs():
        skill_path = root / name
        if not skill_path.is_dir():
            continue
        entry = load_skill(skill_path)
        if not entry:
            continue
        if check_eligibility and not _check_eligibility(entry):
            return None

        oc = entry.skill.openclaw

        result: Dict[str, Any] = {
            "name": entry.skill.name,
            "description": entry.skill.description,
            # OpenClaw metadata
            "emoji": oc.emoji if oc else None,
            "always": (oc.always if oc else False),
            "skillKey": (oc.skill_key if oc else None),
            "primaryEnv": (oc.primary_env if oc else None),
            "homepage": (oc.homepage if oc else None),
            "os": (oc.os_filter if oc else None),
            "requires": (
                {
                    "bins": oc.requires_bins,
                    "anyBins": oc.requires_any_bins,
                    "env": oc.requires_env,
                    "config": oc.requires_config,
                }
                if oc else None
            ),
            "install": (oc.install if oc else None),
            # Resources
            "scripts": entry.scripts,
            "references": entry.references,
            "assets": entry.assets,
        }

        if include_body:
            try:
                _fm, body = _parse_raw_frontmatter(
                    Path(entry.skill_md_path).read_text(encoding="utf-8", errors="replace")
                )
                result["body"] = body.strip() if body else ""
            except OSError:
                result["body"] = ""

        return result
    return None
