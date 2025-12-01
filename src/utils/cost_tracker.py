"""
Cost Tracker - NIS Protocol v4.0
Track API usage and costs across all LLM providers.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import threading

# Pricing per 1M tokens (as of Nov 2025)
PRICING = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
    "anthropic": {
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    },
    "google": {
        "gemini-pro": {"input": 0.50, "output": 1.50},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    },
    "deepseek": {
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek-coder": {"input": 0.14, "output": 0.28},
    },
    "local": {
        "bitnet": {"input": 0.0, "output": 0.0},  # Free - local
        "tinyllama": {"input": 0.0, "output": 0.0},
    }
}


@dataclass
class UsageRecord:
    """Single API usage record"""
    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    endpoint: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass 
class SessionStats:
    """Stats for a session"""
    session_id: str
    start_time: float
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    by_provider: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class CostTracker:
    """
    Track API costs and usage across all providers.
    Persists data to disk for historical analysis.
    """
    
    def __init__(self, storage_path: str = "data/costs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.current_session = SessionStats(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=time.time()
        )
        
        self.records: List[UsageRecord] = []
        self.lock = threading.Lock()
        
        # Load today's records
        self._load_today()
    
    def _get_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request"""
        provider_pricing = PRICING.get(provider.lower(), {})
        
        # Try exact match first
        model_pricing = provider_pricing.get(model.lower())
        
        # Try partial match
        if not model_pricing:
            for model_key, pricing in provider_pricing.items():
                if model_key in model.lower() or model.lower() in model_key:
                    model_pricing = pricing
                    break
        
        if not model_pricing:
            # Unknown model - estimate based on provider average
            if provider_pricing:
                avg_input = sum(p["input"] for p in provider_pricing.values()) / len(provider_pricing)
                avg_output = sum(p["output"] for p in provider_pricing.values()) / len(provider_pricing)
                model_pricing = {"input": avg_input, "output": avg_output}
            else:
                # Complete unknown - use conservative estimate
                model_pricing = {"input": 1.0, "output": 3.0}
        
        cost = (input_tokens * model_pricing["input"] + output_tokens * model_pricing["output"]) / 1_000_000
        return round(cost, 6)
    
    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        endpoint: str = "/chat",
        latency_ms: float = 0,
        success: bool = True,
        error: Optional[str] = None
    ) -> UsageRecord:
        """Record an API call"""
        cost = self._get_cost(provider, model, input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            endpoint=endpoint,
            latency_ms=latency_ms,
            success=success,
            error=error
        )
        
        with self.lock:
            self.records.append(record)
            self._update_session_stats(record)
        
        # Periodic save
        if len(self.records) % 10 == 0:
            self._save_today()
        
        return record
    
    def _update_session_stats(self, record: UsageRecord):
        """Update session statistics"""
        s = self.current_session
        s.total_requests += 1
        
        if record.success:
            s.successful_requests += 1
        else:
            s.failed_requests += 1
        
        s.total_input_tokens += record.input_tokens
        s.total_output_tokens += record.output_tokens
        s.total_cost_usd += record.cost_usd
        
        # Update average latency
        s.avg_latency_ms = (
            (s.avg_latency_ms * (s.total_requests - 1) + record.latency_ms) 
            / s.total_requests
        )
        
        # Update by-provider stats
        if record.provider not in s.by_provider:
            s.by_provider[record.provider] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0
            }
        
        s.by_provider[record.provider]["requests"] += 1
        s.by_provider[record.provider]["tokens"] += record.input_tokens + record.output_tokens
        s.by_provider[record.provider]["cost"] += record.cost_usd
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary"""
        s = self.current_session
        duration = time.time() - s.start_time
        
        return {
            "session_id": s.session_id,
            "duration_minutes": round(duration / 60, 1),
            "total_requests": s.total_requests,
            "success_rate": round(s.successful_requests / max(s.total_requests, 1) * 100, 1),
            "total_tokens": s.total_input_tokens + s.total_output_tokens,
            "total_cost_usd": round(s.total_cost_usd, 4),
            "avg_latency_ms": round(s.avg_latency_ms, 1),
            "by_provider": s.by_provider,
            "cost_per_request": round(s.total_cost_usd / max(s.total_requests, 1), 4)
        }
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary for a specific day"""
        if date is None:
            date = datetime.now()
        
        day_start = datetime(date.year, date.month, date.day).timestamp()
        day_end = day_start + 86400
        
        day_records = [r for r in self.records if day_start <= r.timestamp < day_end]
        
        if not day_records:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "total_requests": 0,
                "total_cost_usd": 0.0
            }
        
        total_cost = sum(r.cost_usd for r in day_records)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in day_records)
        
        by_provider = {}
        for r in day_records:
            if r.provider not in by_provider:
                by_provider[r.provider] = {"requests": 0, "cost": 0.0}
            by_provider[r.provider]["requests"] += 1
            by_provider[r.provider]["cost"] += r.cost_usd
        
        return {
            "date": date.strftime("%Y-%m-%d"),
            "total_requests": len(day_records),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "by_provider": by_provider
        }
    
    def get_monthly_summary(self, year: int = None, month: int = None) -> Dict[str, Any]:
        """Get summary for a month"""
        now = datetime.now()
        year = year or now.year
        month = month or now.month
        
        month_start = datetime(year, month, 1).timestamp()
        if month == 12:
            month_end = datetime(year + 1, 1, 1).timestamp()
        else:
            month_end = datetime(year, month + 1, 1).timestamp()
        
        month_records = [r for r in self.records if month_start <= r.timestamp < month_end]
        
        total_cost = sum(r.cost_usd for r in month_records)
        
        return {
            "month": f"{year}-{month:02d}",
            "total_requests": len(month_records),
            "total_cost_usd": round(total_cost, 2),
            "daily_average": round(total_cost / max(now.day, 1), 4) if month_records else 0
        }
    
    def estimate_monthly_cost(self) -> Dict[str, Any]:
        """Estimate monthly cost based on current usage"""
        summary = self.get_session_summary()
        
        if summary["duration_minutes"] < 1:
            return {"estimated_monthly_usd": 0, "confidence": "low"}
        
        # Extrapolate to monthly
        hours_per_day = 8  # Assume 8 hours of usage per day
        days_per_month = 22  # Working days
        
        cost_per_minute = summary["total_cost_usd"] / summary["duration_minutes"]
        estimated_monthly = cost_per_minute * 60 * hours_per_day * days_per_month
        
        return {
            "estimated_monthly_usd": round(estimated_monthly, 2),
            "based_on_minutes": round(summary["duration_minutes"], 1),
            "current_cost": summary["total_cost_usd"],
            "confidence": "medium" if summary["duration_minutes"] > 30 else "low"
        }
    
    def _load_today(self):
        """Load today's records from disk"""
        today_file = self.storage_path / f"{datetime.now().strftime('%Y-%m-%d')}.json"
        if today_file.exists():
            try:
                with open(today_file, 'r') as f:
                    data = json.load(f)
                    self.records = [UsageRecord(**r) for r in data]
            except Exception:
                pass
    
    def _save_today(self):
        """Save today's records to disk"""
        today_file = self.storage_path / f"{datetime.now().strftime('%Y-%m-%d')}.json"
        try:
            with open(today_file, 'w') as f:
                json.dump([asdict(r) for r in self.records], f)
        except Exception:
            pass
    
    def save(self):
        """Force save current records"""
        self._save_today()


# Global instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker"""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
