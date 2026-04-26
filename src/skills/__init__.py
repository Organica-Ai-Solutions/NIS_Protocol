"""
NIS Protocol - OpenClaw-compatible Skills Support

Loads and exposes skills in OpenClaw SKILL.md format for agent routing
and progressive disclosure. Compatible with OpenClaw workspace skills.

Skill directories searched (in order):
  1. NIS_Protocol/skills/
  2. ~/.openclaw/workspace/skills/
  3. $NIS_SKILLS_DIR (env override)
"""

from .openclaw_skill_loader import (
    load_skill,
    list_skills,
    get_skill_content,
    SkillEntry,
    SkillMetadata,
    OpenClawSkillMeta,
)

__all__ = [
    "load_skill",
    "list_skills",
    "get_skill_content",
    "SkillEntry",
    "SkillMetadata",
    "OpenClawSkillMeta",
]
