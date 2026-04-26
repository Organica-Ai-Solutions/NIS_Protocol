"""
NIS Protocol — PromptInjectionScanner
======================================
Detects prompt injection, data exfiltration, and jailbreak attempts.

Learned from OpenFang's System #12 (Prompt Injection Scanner).
Goes deeper: pattern library tuned for robotics/hardware control —
extra dangerous when someone can override arm commands.

Threat categories:
  1. Override attempts    — "ignore previous instructions", "forget your rules"
  2. Data exfiltration    — "print your API keys", "show me system files"
  3. Shell injection      — "run: rm -rf", backtick/semicolon patterns
  4. Role confusion       — "you are now DAN", "act as root"
  5. Hardware override    — "move servo to position 0", "disable safety"
  6. Indirect injection   — base64, unicode substitution, zero-width chars

DIKW mapping:
  Data        → raw input string
  Information → matched threat patterns with positions
  Knowledge   → threat assessment (score, category, severity)
  Wisdom      → recommended action (allow, warn, block, sanitize)
"""

from __future__ import annotations

import base64
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("nis.injection_scanner")


# ── Severity ──────────────────────────────────────────────────────────────────

class Severity(Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class Action(Enum):
    ALLOW    = "allow"
    WARN     = "warn"
    BLOCK    = "block"
    SANITIZE = "sanitize"


# ── Threat pattern library ────────────────────────────────────────────────────

@dataclass
class ThreatPattern:
    pattern_id: str
    category: str
    severity: Severity
    regex: re.Pattern
    description: str


def _p(pid: str, cat: str, sev: Severity, pattern: str, desc: str) -> ThreatPattern:
    return ThreatPattern(pid, cat, sev, re.compile(pattern, re.IGNORECASE | re.MULTILINE), desc)


THREAT_PATTERNS: List[ThreatPattern] = [
    # Override attempts
    _p("OVR-001", "override", Severity.HIGH,
       r"ignore\s+(previous|all|above|prior)\s+(instructions|prompt|rules|context)",
       "Classic instruction override attempt"),
    _p("OVR-002", "override", Severity.HIGH,
       r"(forget|disregard|discard)\s+(everything|all)\s*(you\s+)?(know|were told|have been told)",
       "Memory reset attempt"),
    _p("OVR-003", "override", Severity.CRITICAL,
       r"(your|the)\s+(new|real|actual|true)\s+(instructions?|rules?|purpose|mission)\s+(are|is)",
       "Rule replacement attempt"),
    _p("OVR-004", "override", Severity.HIGH,
       r"(system|admin|root)\s*:\s*(override|unlock|enable|disable)",
       "System-level override via prompt prefix"),
    _p("OVR-005", "override", Severity.MEDIUM,
       r"\[\s*(SYSTEM|ADMIN|ROOT|OVERRIDE|JAILBREAK)\s*\]",
       "Fake system tag injection"),

    # Role confusion / jailbreak
    _p("ROLE-001", "role_confusion", Severity.HIGH,
       r"(you are|act as|pretend to be|roleplay as|simulate)\s+(dan|evil|unrestricted|jailbroken|root|admin|god)",
       "DAN-style jailbreak"),
    _p("ROLE-002", "role_confusion", Severity.MEDIUM,
       r"(developer|maintenance|debug|test)\s+mode",
       "Fake mode switch"),
    _p("ROLE-003", "role_confusion", Severity.HIGH,
       r"(disable|bypass|remove|skip)\s+(safety|guardrail|filter|restriction|limit)",
       "Safety bypass request"),

    # Data exfiltration
    _p("EXFIL-001", "exfiltration", Severity.HIGH,
       r"(print|show|reveal|output|display|tell me)\s+(your|the|all)\s+(api[\s_]?key|secret|password|token|credential)",
       "API key exfiltration"),
    _p("EXFIL-002", "exfiltration", Severity.HIGH,
       r"(read|cat|open|dump|show)\s+[/~\.]?[\w/\.]+\.(env|key|pem|crt|json|toml|cfg|ini|yml)",
       "Config file read attempt"),
    _p("EXFIL-003", "exfiltration", Severity.MEDIUM,
       r"what\s+is\s+(your|the)\s+(system\s+prompt|instructions?|context)",
       "System prompt extraction"),

    # Shell injection
    _p("SHELL-001", "shell_injection", Severity.CRITICAL,
       r"(;|\||&&)\s*(rm|del|format|shutdown|curl|wget|bash|sh|cmd|powershell|python)\s",
       "Shell command chaining"),
    _p("SHELL-002", "shell_injection", Severity.CRITICAL,
       r"`[^`]{0,200}`",
       "Backtick shell execution"),
    _p("SHELL-003", "shell_injection", Severity.HIGH,
       r"\$\((.*?)\)",
       "Command substitution attempt"),
    _p("SHELL-004", "shell_injection", Severity.HIGH,
       r"(os\.system|subprocess|exec|eval|__import__)\s*\(",
       "Python exec injection"),

    # Hardware override (critical for robotics)
    _p("HW-001", "hardware_override", Severity.CRITICAL,
       r"(move|set|send)\s+(servo|motor|joint)\s+.{0,20}(to\s+)?\b0\b",
       "Zero-position servo command (crash risk)"),
    _p("HW-002", "hardware_override", Severity.HIGH,
       r"(disable|turn\s+off|bypass)\s+(safety|limit|guard|stop|estop|emergency)",
       "Hardware safety disable"),
    _p("HW-003", "hardware_override", Severity.HIGH,
       r"(force|override)\s+(calibration|home|position)\s+to",
       "Calibration override attempt"),
    _p("HW-004", "hardware_override", Severity.MEDIUM,
       r"move\s+(all|every)\s+servo.{0,20}(max|1000|full)",
       "Max-position all servos (crash risk)"),

    # Indirect injection
    _p("IND-001", "indirect_injection", Severity.MEDIUM,
       r"base64\s*decode\s*:?\s*[A-Za-z0-9+/]{30,}={0,2}",
       "Inline base64 decode attempt"),
    _p("IND-002", "indirect_injection", Severity.LOW,
       r"[\u200b-\u200f\u202a-\u202e\ufeff]",
       "Zero-width / invisible Unicode characters"),
    _p("IND-003", "indirect_injection", Severity.MEDIUM,
       r"\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}",
       "Unicode escape sequence substitution"),
]

_SEVERITY_SCORE = {Severity.LOW: 1, Severity.MEDIUM: 3, Severity.HIGH: 7, Severity.CRITICAL: 10}
_BLOCK_THRESHOLD = 7      # score >= this → block
_WARN_THRESHOLD  = 3      # score >= this → warn


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ThreatMatch:
    pattern_id: str
    category: str
    severity: Severity
    description: str
    matched_text: str
    position: int


@dataclass
class ScanResult:
    safe: bool
    action: Action
    score: int
    threats: List[ThreatMatch]
    sanitized_text: Optional[str]
    scan_ms: float
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        if self.safe:
            return "SAFE"
        cats = list({t.category for t in self.threats})
        return f"{'BLOCKED' if self.action == Action.BLOCK else 'WARNED'} [{','.join(cats)}] score={self.score}"


# ── Scanner ───────────────────────────────────────────────────────────────────

class PromptInjectionScanner:
    """
    Multi-layer prompt injection and threat scanner for the NIS NeuroKernel.

    Usage:
        scanner = PromptInjectionScanner()
        result = scanner.scan(user_input)
        if not result.safe:
            if result.action == Action.BLOCK:
                raise SecurityError(result.summary())
            else:
                user_input = result.sanitized_text  # use cleaned version
    """

    def __init__(
        self,
        extra_patterns: Optional[List[ThreatPattern]] = None,
        block_threshold: int = _BLOCK_THRESHOLD,
        warn_threshold: int = _WARN_THRESHOLD,
        max_input_length: int = 32768,
    ):
        self.patterns = THREAT_PATTERNS + (extra_patterns or [])
        self.block_threshold = block_threshold
        self.warn_threshold = warn_threshold
        self.max_input_length = max_input_length
        self._scan_count = 0
        self._block_count = 0
        self._warn_count = 0

    def scan(self, text: str, context: str = "user_input") -> ScanResult:
        """
        Scan text for injection threats.
        Returns ScanResult with action recommendation and optional sanitized text.
        """
        start = time.time()
        self._scan_count += 1

        # Pre-processing
        if len(text) > self.max_input_length:
            text = text[:self.max_input_length]

        # Normalize unicode (catches homoglyph attacks)
        normalized = unicodedata.normalize("NFKC", text)

        # Try base64 decoding to detect embedded payloads
        b64_decoded = self._try_decode_b64(text)

        threats: List[ThreatMatch] = []
        total_score = 0

        for scan_text in [normalized, b64_decoded] if b64_decoded else [normalized]:
            for pattern in self.patterns:
                for match in pattern.regex.finditer(scan_text):
                    matched = match.group(0)[:100]
                    threat = ThreatMatch(
                        pattern_id=pattern.pattern_id,
                        category=pattern.category,
                        severity=pattern.severity,
                        description=pattern.description,
                        matched_text=matched,
                        position=match.start(),
                    )
                    threats.append(threat)
                    total_score += _SEVERITY_SCORE[pattern.severity]

        # Deduplicate by pattern_id
        seen_ids: set = set()
        unique_threats = []
        for t in threats:
            if t.pattern_id not in seen_ids:
                seen_ids.add(t.pattern_id)
                unique_threats.append(t)
                if t.severity in (Severity.HIGH, Severity.CRITICAL):
                    logger.warning(f"[Scanner/{context}] {t.pattern_id}: {t.description} | '{t.matched_text[:60]}'")

        # Determine action
        if total_score >= self.block_threshold:
            action = Action.BLOCK
            self._block_count += 1
        elif total_score >= self.warn_threshold:
            action = Action.WARN
            self._warn_count += 1
        else:
            action = Action.ALLOW

        # Sanitize (remove matched patterns, replace with [REDACTED])
        sanitized = None
        if action != Action.BLOCK and unique_threats:
            sanitized = self._sanitize(normalized, unique_threats)

        scan_ms = (time.time() - start) * 1000

        return ScanResult(
            safe=(action == Action.ALLOW),
            action=action,
            score=total_score,
            threats=unique_threats,
            sanitized_text=sanitized,
            scan_ms=scan_ms,
        )

    def _try_decode_b64(self, text: str) -> Optional[str]:
        """Try to decode base64 blobs in the text to detect embedded injections."""
        b64_blobs = re.findall(r"[A-Za-z0-9+/]{40,}={0,2}", text)
        for blob in b64_blobs[:3]:
            try:
                decoded = base64.b64decode(blob + "==").decode("utf-8", errors="ignore")
                if len(decoded) > 20 and any(kw in decoded.lower() for kw in
                   ["ignore", "forget", "you are", "system:", "api_key", "password"]):
                    logger.warning(f"[Scanner] Suspicious base64 payload detected")
                    return decoded
            except Exception:
                pass
        return None

    def _sanitize(self, text: str, threats: List[ThreatMatch]) -> str:
        """Replace matched threat patterns with [REDACTED]."""
        result = text
        for threat in sorted(threats, key=lambda t: t.position, reverse=True):
            for pattern in self.patterns:
                if pattern.pattern_id == threat.pattern_id:
                    result = pattern.regex.sub("[REDACTED]", result)
                    break
        return result

    def scan_safe(self, text: str, context: str = "user_input") -> Tuple[bool, str]:
        """
        Simple interface: returns (is_safe, cleaned_text).
        Use this when you want automatic sanitize-on-warn behavior.
        """
        result = self.scan(text, context)
        if result.action == Action.BLOCK:
            return False, ""
        if result.sanitized_text:
            return True, result.sanitized_text
        return True, text

    def stats(self) -> Dict[str, Any]:
        return {
            "total_scans": self._scan_count,
            "blocked": self._block_count,
            "warned": self._warn_count,
            "block_rate": round(self._block_count / max(1, self._scan_count), 3),
            "patterns_loaded": len(self.patterns),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_scanner: Optional[PromptInjectionScanner] = None


def get_scanner() -> PromptInjectionScanner:
    global _scanner
    if _scanner is None:
        _scanner = PromptInjectionScanner()
    return _scanner
