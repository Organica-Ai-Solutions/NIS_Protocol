# NIS Protocol Refactoring Plan

## Objectives

1. **Remove all "Cursor" references** - Replace with "Action Validation Pattern"
2. **Replace simulation modes with real implementations**
3. **Harden runner security**

## Progress Tracker

### 1. Remove Cursor References ✅ (In Progress)

**Files Updated**:
- ✅ `src/core/agent_orchestrator.py` - Renamed to "Action Validation Pattern"

**Files Remaining**:
- `src/mcp/ui_resources.py` - CSS cursor properties (keep these - they're valid CSS)
- `docs/archived/CURSOR_PATTERN_IMPLEMENTATION_PLAN.md` - Already archived, no action needed

**Status**: Complete - All code references updated

---

### 2. Replace Simulation Modes with Real Implementations

#### Components Currently in Simulation Mode

**High Priority - Robotics Protocols**:

1. **CAN Protocol** (`routes/robotics.py`, `src/protocols/can_protocol.py`)
   - Current: `simulation_mode=True` flag
   - Issue: Returns mock data instead of real CAN bus communication
   - Fix: Implement actual SocketCAN interface for Linux
   - Hardware Required: CAN adapter (USB-CAN or built-in)

2. **OBD-II Protocol** (`routes/robotics.py`, `src/protocols/obd_protocol.py`)
   - Current: Returns default vehicle data
   - Issue: No real OBD-II device connection
   - Fix: Implement ELM327 protocol over serial/Bluetooth
   - Hardware Required: OBD-II adapter

3. **Robotics Agent** (`src/agents/robotics/unified_robotics_agent.py`)
   - Current: SITL (Software In The Loop) simulation
   - Issue: No real hardware control
   - Fix: Conditional - use simulation when hardware unavailable, real when present
   - Hardware Required: Drone/robot with MAVLink support

**Medium Priority - Training & Optimization**:

4. **BitNet Online Trainer** (`src/agents/training/bitnet_online_trainer.py`)
   - Current: Falls back to simulation if model weights missing
   - Issue: Training doesn't persist or improve model
   - Fix: Implement real model fine-tuning with checkpointing
   - Resources Required: GPU, model storage

**Low Priority - Testing Components**:

5. **NVIDIA Isaac Bridge** (`src/agents/isaac/isaac_bridge_agent.py`)
   - Current: Simulation mode for testing
   - Issue: No real Isaac Sim connection
   - Fix: Keep simulation mode, add real connection when Isaac Sim available
   - Resources Required: NVIDIA Isaac Sim license

#### Implementation Strategy

**Phase 1: Auto-Detection (Recommended)**
```python
# Pattern to use throughout codebase
class ProtocolHandler:
    def __init__(self):
        self.hardware_available = self._detect_hardware()
        self.mode = "real" if self.hardware_available else "simulation"
    
    def _detect_hardware(self) -> bool:
        # Try to connect to real hardware
        # Return True if successful, False if not available
        pass
```

**Phase 2: Graceful Fallback**
- Always try real hardware first
- Fall back to simulation with clear logging
- Return metadata indicating which mode was used

**Phase 3: Configuration Override**
- Environment variable: `FORCE_SIMULATION=true/false`
- Allows testing simulation even when hardware present

---

### 3. Runner Security Hardening

#### Current Security (`runner/runner_app.py`)

**Existing Protections**:
- ✅ RestrictedPython for code sandboxing
- ✅ Blocked imports (os, sys, subprocess, requests, etc.)
- ✅ Safe builtins only
- ✅ Allowed scientific libraries (numpy, pandas, scipy)
- ✅ File extension whitelist
- ✅ Execution timeout (30s default)
- ✅ Memory limit (512MB default)

**Security Gaps Identified**:

1. **Network Access**
   - Issue: No network isolation
   - Fix: Add network namespace isolation (Linux) or firewall rules

2. **File System Access**
   - Issue: Can read/write to `/app/workspace` and `/app/temp`
   - Fix: Implement chroot jail or container-level restrictions

3. **Resource Exhaustion**
   - Issue: No CPU throttling
   - Fix: Add cgroup limits for CPU usage

4. **Process Limits**
   - Issue: MAX_PROCESSES=5 but not enforced
   - Fix: Implement actual process counting and killing

5. **Logging & Audit**
   - Issue: Limited execution tracking
   - Fix: Add comprehensive audit logging

#### Hardening Implementation

**Priority 1: Network Isolation**
```python
# Add to runner_app.py
import socket

class NetworkBlocker:
    """Block all network access during code execution"""
    
    @staticmethod
    def block_network():
        # Override socket creation
        socket.socket = lambda *args, **kwargs: None
```

**Priority 2: Enhanced Resource Limits**
```python
# Add cgroup-based CPU limits
import resource

def set_resource_limits():
    # CPU time limit
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    # Memory limit
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    # File size limit
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
```

**Priority 3: Audit Logging**
```python
# Add execution audit trail
class ExecutionAuditor:
    def log_execution(self, exec_id: str, code: str, result: dict):
        audit_entry = {
            "execution_id": exec_id,
            "timestamp": time.time(),
            "code_hash": hashlib.sha256(code.encode()).hexdigest(),
            "success": result.get("success"),
            "violations": result.get("security_violations", []),
            "resource_usage": {
                "cpu_time": result.get("execution_time_seconds"),
                "memory_mb": result.get("memory_used_mb")
            }
        }
        # Store in database or log file
```

**Priority 4: Input Validation**
```python
# Add code analysis before execution
class CodeValidator:
    DANGEROUS_PATTERNS = [
        r'__import__',
        r'eval\(',
        r'exec\(',
        r'compile\(',
        r'open\(',
        r'file\(',
    ]
    
    def validate(self, code: str) -> tuple[bool, list[str]]:
        violations = []
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                violations.append(f"Dangerous pattern detected: {pattern}")
        return len(violations) == 0, violations
```

---

## Implementation Order

### Week 1: Cursor Removal & Documentation
- [x] Update `agent_orchestrator.py`
- [ ] Update any remaining code comments
- [ ] Update API documentation
- [ ] Update README if needed

### Week 2: Simulation Mode Refactoring
- [ ] Implement auto-detection pattern
- [ ] Update CAN protocol with hardware detection
- [ ] Update OBD protocol with hardware detection
- [ ] Update robotics agent with SITL/real mode switching
- [ ] Add configuration overrides

### Week 3: Runner Security Hardening
- [ ] Implement network isolation
- [ ] Add enhanced resource limits
- [ ] Implement audit logging
- [ ] Add input validation
- [ ] Add process monitoring and limits
- [ ] Security testing and penetration testing

### Week 4: Testing & Documentation
- [ ] Integration tests for all changes
- [ ] Security audit
- [ ] Performance benchmarks
- [ ] Update deployment documentation
- [ ] Update security documentation

---

## Testing Strategy

### Simulation Mode Testing
```bash
# Test with hardware unavailable
FORCE_SIMULATION=true python -m pytest tests/test_protocols.py

# Test with hardware detection
python -m pytest tests/test_protocols.py

# Test graceful fallback
# (Disconnect hardware mid-test)
```

### Security Testing
```bash
# Test network blocking
python tests/security/test_network_isolation.py

# Test resource limits
python tests/security/test_resource_limits.py

# Test malicious code detection
python tests/security/test_code_validation.py
```

---

## Success Criteria

### Cursor Removal
- ✅ No references to "Cursor" in active code (excluding CSS)
- ✅ All documentation updated
- ✅ API responses don't mention "Cursor"

### Simulation Mode
- ✅ Hardware auto-detection works
- ✅ Graceful fallback to simulation
- ✅ Clear logging of which mode is active
- ✅ Metadata in responses indicates real vs simulation
- ✅ No breaking changes to API

### Runner Security
- ✅ Network access blocked during execution
- ✅ Resource limits enforced (CPU, memory, disk)
- ✅ Comprehensive audit logging
- ✅ Input validation catches dangerous patterns
- ✅ No security vulnerabilities in penetration testing
- ✅ Performance impact < 10%

---

## Notes

- Keep simulation mode available for testing and development
- Don't break existing API contracts
- Maintain backward compatibility
- Document all security changes
- Add metrics for monitoring production usage
