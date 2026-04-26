"""
NIS Protocol v4.0 - OpenClaw-compatible Skills Routes

Exposes skills in OpenClaw SKILL.md format for agent routing and
progressive disclosure. Skills can be loaded from:
- NIS_Protocol/skills/
- ~/.openclaw/workspace/skills/
- NIS_SKILLS_DIR env var

Endpoints:
  GET /v4/skills                   — list all skills (metadata only)
  GET /v4/skills/prompt/snapshot   — build agent context prompt
  GET /v4/skills/{name}            — full skill content (metadata + body)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("nis.routes.skills")

router = APIRouter(prefix="/v4/skills", tags=["Skills"])

try:
    from src.skills.openclaw_skill_loader import list_skills, get_skill_content, SkillEntry
    SKILLS_AVAILABLE = True
except ImportError as e:
    SKILLS_AVAILABLE = False
    logger.warning("Skills module not available: %s", e)


def _entry_to_dict(entry: "SkillEntry") -> Dict[str, Any]:
    """Convert SkillEntry to JSON-serializable dict with full OpenClaw metadata."""
    raw = entry.skill.raw or {}
    # Resolve openclaw metadata block
    oc_meta: Dict[str, Any] = {}
    meta_block = raw.get("metadata")
    if isinstance(meta_block, dict):
        oc_block = meta_block.get("openclaw")
        if isinstance(oc_block, dict):
            oc_meta = oc_block

    return {
        "name": entry.skill.name,
        "description": entry.skill.description,
        "emoji": oc_meta.get("emoji"),
        "always": oc_meta.get("always", False),
        "primaryEnv": oc_meta.get("primaryEnv"),
        "homepage": oc_meta.get("homepage"),
        "skillKey": oc_meta.get("skillKey"),
        "os": oc_meta.get("os"),
        "requires": oc_meta.get("requires"),
        "install": oc_meta.get("install"),
        "scripts": entry.scripts,
        "references": entry.references,
        "assets": entry.assets,
    }


def _parse_names_filter(names: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated skill names query param into a list."""
    if not names:
        return None
    return [s.strip() for s in names.split(",") if s.strip()]


@router.get("", summary="List available skills")
async def list_available_skills(
    names: Optional[str] = Query(
        None,
        alias="filter",
        description="Comma-separated skill names to include (omit for all)",
    ),
    always_only: bool = Query(
        False,
        description="Only return skills marked always:true in metadata",
    ),
) -> Dict[str, Any]:
    """
    List all available OpenClaw-compatible skills with full metadata.
    Skills are loaded from NIS_Protocol/skills, ~/.openclaw/workspace/skills, or NIS_SKILLS_DIR.
    """
    if not SKILLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Skills module not available")

    skill_filter = _parse_names_filter(names)
    entries = list_skills(skill_filter=skill_filter)

    if always_only:
        entries = [
            e for e in entries
            if isinstance(e.skill.raw.get("metadata"), dict)
            and isinstance(e.skill.raw["metadata"].get("openclaw"), dict)
            and e.skill.raw["metadata"]["openclaw"].get("always") is True
        ]

    return {
        "skills": [_entry_to_dict(e) for e in entries],
        "count": len(entries),
    }


@router.get("/prompt/snapshot", summary="Build skills prompt for agent context")
async def get_skills_prompt(
    names: Optional[str] = Query(
        None,
        alias="filter",
        description="Comma-separated skill names",
    ),
    max_chars: int = Query(30000, ge=1000, le=100000, description="Max prompt chars"),
) -> Dict[str, Any]:
    """
    Build a prompt string from skill metadata (name + description) for agent context.
    Matches OpenClaw's buildWorkspaceSkillsPrompt behavior — only metadata is always in context.
    """
    if not SKILLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Skills module not available")

    skill_filter = _parse_names_filter(names)
    entries = list_skills(skill_filter=skill_filter)

    lines: List[str] = []
    for e in entries:
        raw = e.skill.raw or {}
        oc_meta: Dict[str, Any] = {}
        if isinstance(raw.get("metadata"), dict) and isinstance(
            raw["metadata"].get("openclaw"), dict
        ):
            oc_meta = raw["metadata"]["openclaw"]
        emoji = oc_meta.get("emoji", "")
        prefix = f"{emoji} " if emoji else ""
        lines.append(f"## {prefix}{e.skill.name}\n{e.skill.description}")

    prompt = "\n\n".join(lines)
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n\n[... truncated]"

    return {
        "prompt": prompt,
        "skills": [e.skill.name for e in entries],
        "char_count": len(prompt),
    }


@router.get("/{name}", summary="Get full skill content")
async def get_skill(
    name: str,
    include_body: bool = Query(True, description="Include SKILL.md body text"),
) -> Dict[str, Any]:
    """
    Get full skill content by name: metadata, optional body, and resource paths.
    Progressive disclosure: call with include_body=false for metadata-only (always in context),
    then again with include_body=true when the skill is triggered.
    """
    if not SKILLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Skills module not available")

    content = get_skill_content(name, include_body=include_body)
    if not content:
        raise HTTPException(status_code=404, detail=f"Skill not found: {name}")
    return content
