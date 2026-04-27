"""
NIS Protocol — SkillLoader
===========================
Learns from OpenFang's SKILL.md pattern but goes deeper:
  - Loads domain expertise files (SKILL.md / .toml / .json) at runtime
  - Injects them into agent context as the KNOWLEDGE layer of DIKW
  - Hot-reloads when skills change on disk
  - Indexes skills by semantic tags for context-aware injection
  - Reports which skills contributed to each agent decision (attribution)

DIKW mapping:
  Data        → raw SKILL.md files on disk
  Information → parsed, structured skill dicts with tags
  Knowledge   → skill context injected into LLM system prompts
  Wisdom      → skill attribution in audit trail (what was used, why)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("nis.skill_loader")


@dataclass
class Skill:
    """A loaded, parsed skill."""
    name: str
    path: str
    content: str                     # raw SKILL.md / JSON text
    tags: List[str]                  # extracted tags for context-routing
    summary: str                     # first paragraph / description
    sections: Dict[str, str]         # heading → body for structured access
    file_hash: str                   # SHA256 of file content (change detection)
    loaded_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    use_count: int = 0


class SkillLoader:
    """
    Hot-reloading skill injection engine for the NIS NeuroKernel.

    Usage:
        loader = SkillLoader(skill_dirs=["agents", "skills"])
        loader.load_all()

        # Inject into a system prompt
        extra_context = loader.build_context_for(query="pick up the red block")
        system_prompt = base_prompt + extra_context
    """

    def __init__(self, skill_dirs: Optional[List[str]] = None):
        base = Path(__file__).resolve().parent.parent.parent  # NIS_Protocol root
        self.skill_dirs: List[Path] = []
        for d in (skill_dirs or ["agents", "skills", "src/agents"]):
            p = base / d
            if p.exists():
                self.skill_dirs.append(p)

        self._skills: Dict[str, Skill] = {}   # name → Skill
        self._tag_index: Dict[str, List[str]] = {}  # tag → [skill_name]

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_all(self) -> int:
        """Scan all skill dirs, load/refresh any SKILL.md or agent.toml."""
        loaded = 0
        for skill_dir in self.skill_dirs:
            for path in skill_dir.rglob("SKILL.md"):
                if self._load_file(path):
                    loaded += 1
            for path in skill_dir.rglob("agent.toml"):
                if self._load_toml_skill(path):
                    loaded += 1
        logger.info(f"SkillLoader: {loaded} skills loaded from {len(self.skill_dirs)} dirs")
        return loaded

    def _file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def _load_file(self, path: Path) -> bool:
        """Load or hot-reload a SKILL.md file."""
        try:
            h = self._file_hash(path)
            name = path.parent.name or path.stem
            existing = self._skills.get(name)
            if existing and existing.file_hash == h:
                return False  # unchanged

            content = path.read_text(encoding="utf-8")
            skill = Skill(
                name=name,
                path=str(path),
                content=content,
                tags=self._extract_tags(content, name),
                summary=self._extract_summary(content),
                sections=self._parse_sections(content),
                file_hash=h,
            )
            self._skills[name] = skill
            self._rebuild_tag_index()
            logger.debug(f"Loaded skill '{name}' ({len(content)} chars, tags={skill.tags})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load skill {path}: {e}")
            return False

    def _load_toml_skill(self, path: Path) -> bool:
        """Extract system_prompt from an agent.toml as a skill."""
        try:
            content = path.read_text(encoding="utf-8")
            # Extract name
            m = re.search(r'^name\s*=\s*"([^"]+)"', content, re.MULTILINE)
            name = (m.group(1) if m else path.parent.name) + ".toml"
            # Extract description
            m2 = re.search(r'^description\s*=\s*"([^"]+)"', content, re.MULTILINE)
            desc = m2.group(1) if m2 else ""
            # Extract system_prompt block
            m3 = re.search(r'system_prompt\s*=\s*"""(.*?)"""', content, re.DOTALL)
            prompt_text = m3.group(1).strip() if m3 else desc

            h = self._file_hash(path)
            existing = self._skills.get(name)
            if existing and existing.file_hash == h:
                return False

            skill_content = f"# {name}\n{desc}\n\n{prompt_text}"
            skill = Skill(
                name=name,
                path=str(path),
                content=skill_content,
                tags=self._extract_tags(skill_content, name),
                summary=desc[:200],
                sections=self._parse_sections(skill_content),
                file_hash=h,
            )
            self._skills[name] = skill
            self._rebuild_tag_index()
            return True
        except Exception as e:
            logger.warning(f"Failed to load toml skill {path}: {e}")
            return False

    # ── Tag extraction ────────────────────────────────────────────────────────

    _ROBOTICS_TAGS = {"arm", "servo", "xarm", "robot", "pick", "place", "gripper", "calibrat"}
    _COSMOS_TAGS   = {"cosmos", "vision", "visual", "spatial", "depth", "camera", "reason"}
    _CODE_TAGS     = {"code", "python", "function", "debug", "error", "test", "build"}
    _SECURITY_TAGS = {"security", "auth", "injection", "threat", "audit", "safe"}

    def _extract_tags(self, content: str, name: str) -> List[str]:
        text = (content + " " + name).lower()
        tags = set()
        for t in self._ROBOTICS_TAGS:
            if t in text:
                tags.add("robotics")
                break
        for t in self._COSMOS_TAGS:
            if t in text:
                tags.add("cosmos")
                break
        for t in self._CODE_TAGS:
            if t in text:
                tags.add("coding")
                break
        for t in self._SECURITY_TAGS:
            if t in text:
                tags.add("security")
                break
        # Extract explicit tags from TOML `tags = ["..."]` — only when file is TOML
        if name.endswith(".toml"):
            m = re.findall(r'"([a-z][a-z0-9\-]+)"', content)
            tags.update(m[:8])
        # Tag from filename/parent
        tags.add(name.split(".")[0].lower().replace("-", "_"))
        return sorted(tags)

    def _extract_summary(self, content: str) -> str:
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        for line in lines:
            if not line.startswith("#"):
                return line[:300]
        return lines[0][:300] if lines else ""

    def _parse_sections(self, content: str) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        current_heading = "_root"
        current_body: List[str] = []
        for line in content.splitlines():
            if line.startswith("#"):
                if current_body:
                    sections[current_heading] = "\n".join(current_body).strip()
                current_heading = line.lstrip("#").strip()
                current_body = []
            else:
                current_body.append(line)
        if current_body:
            sections[current_heading] = "\n".join(current_body).strip()
        return sections

    def _rebuild_tag_index(self):
        self._tag_index = {}
        for name, skill in self._skills.items():
            for tag in skill.tags:
                self._tag_index.setdefault(tag, []).append(name)

    # ── Context injection ─────────────────────────────────────────────────────

    def get_skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def skills_for_query(self, query: str, max_skills: int = 3) -> List[Skill]:
        """Return the most relevant skills for a given query string."""
        query_lower = query.lower()
        scored: List[tuple[float, Skill]] = []
        for skill in self._skills.values():
            score = 0.0
            for tag in skill.tags:
                if tag in query_lower:
                    score += 2.0
            for word in query_lower.split():
                if word in skill.content.lower():
                    score += 0.1
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:max_skills]]

    def build_context_for(self, query: str, max_skills: int = 2) -> str:
        """
        Build a context block to inject into an LLM system prompt.
        Returns empty string if no relevant skills found.
        """
        skills = self.skills_for_query(query, max_skills)
        if not skills:
            return ""
        parts = ["\n\n--- INJECTED DOMAIN SKILLS ---"]
        for skill in skills:
            parts.append(f"\n[SKILL: {skill.name}]\n{skill.summary}")
            # Include the most relevant section
            best_section = self._best_section(skill, query)
            if best_section:
                parts.append(best_section[:600])
            skill.use_count += 1
            skill.last_used = time.time()
        parts.append("--- END SKILLS ---")
        return "\n".join(parts)

    def _best_section(self, skill: Skill, query: str) -> str:
        query_lower = query.lower()
        best, best_score = "", 0
        for heading, body in skill.sections.items():
            score = sum(1 for w in query_lower.split() if w in (heading + body).lower())
            if score > best_score:
                best, best_score = body, score
        return best

    def inject_into_prompt(self, system_prompt: str, query: str) -> str:
        """Append relevant skill context to a system prompt."""
        ctx = self.build_context_for(query)
        return system_prompt + ctx if ctx else system_prompt

    # ── Hot-reload ────────────────────────────────────────────────────────────

    def refresh(self) -> int:
        """Re-check all files for changes, reload if modified. Returns count of refreshed skills."""
        refreshed = 0
        for skill_dir in self.skill_dirs:
            for path in skill_dir.rglob("SKILL.md"):
                if self._load_file(path):
                    refreshed += 1
            for path in skill_dir.rglob("agent.toml"):
                if self._load_toml_skill(path):
                    refreshed += 1
        return refreshed

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_skills(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": s.name, "tags": s.tags, "summary": s.summary[:100],
                "use_count": s.use_count, "loaded_at": s.loaded_at,
            }
            for s in self._skills.values()
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_skills": len(self._skills),
            "tag_index": {t: len(names) for t, names in self._tag_index.items()},
            "skill_dirs": [str(d) for d in self.skill_dirs],
            "most_used": sorted(
                [(s.name, s.use_count) for s in self._skills.values()],
                key=lambda x: x[1], reverse=True
            )[:5],
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_skill_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    global _skill_loader
    if _skill_loader is None:
        _skill_loader = SkillLoader()
        _skill_loader.load_all()
    return _skill_loader
