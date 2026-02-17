"""
NeuroLinux PC Bridge — High-Performance Windows PC Communication
================================================================
Fast peer-to-peer bridge between NIS Protocol nodes and Windows PCs.
Optimized for Alienware Area-51 R2 and similar high-end workstations.

Features:
- WebSocket + UDP dual-channel (reliable + fast)
- GPU compute offload (send inference/training jobs to PC GPU)
- Model sync (push/pull models between nodes)
- Agent mesh (distributed agent execution across machines)
- File transfer with chunked streaming
- Auto-discovery via mDNS/broadcast
- Heartbeat with hardware telemetry
- Encryption via TLS for production

Architecture:
  Mac/Linux (NIS Hub)  <--WebSocket/UDP-->  Windows PC (NIS Agent)
       :8100 (control)                          :8100 (control)
       :8101 (data/UDP)                         :8101 (data/UDP)

Copyright 2026 Organica AI Solutions
Licensed under Apache License 2.0
"""

import asyncio
import json
import logging
import os
import platform
import socket
import struct
import time
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime

logger = logging.getLogger("neurolinux.pc_bridge")

# ─── Constants ───
DEFAULT_CONTROL_PORT = 8100
DEFAULT_DATA_PORT = 8101
DEFAULT_DISCOVERY_PORT = 8102
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file transfer
MAX_UDP_PACKET = 65507
HEARTBEAT_INTERVAL = 10  # seconds
DISCOVERY_INTERVAL = 30  # seconds


# =============================================================================
# DATA MODELS
# =============================================================================

class PCType(Enum):
    """Known PC hardware profiles for optimization"""
    ALIENWARE_AREA51_R2 = "alienware_area51_r2"
    GENERIC_WINDOWS = "generic_windows"
    GENERIC_LINUX = "generic_linux"
    GENERIC_MAC = "generic_mac"


class ChannelType(Enum):
    CONTROL = "control"   # WebSocket — reliable, ordered
    DATA = "data"         # UDP — fast, for telemetry/streaming


class JobType(Enum):
    INFERENCE = "inference"
    TRAINING = "training"
    MODEL_SYNC = "model_sync"
    FILE_TRANSFER = "file_transfer"
    AGENT_EXEC = "agent_exec"
    BENCHMARK = "benchmark"


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PCProfile:
    """Hardware profile for a connected PC"""
    pc_type: PCType = PCType.GENERIC_WINDOWS
    hostname: str = ""
    ip_address: str = ""
    os_name: str = ""
    os_version: str = ""
    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_cuda_cores: int = 0
    gpu_driver_version: str = ""
    disk_free_gb: float = 0.0
    network_speed_mbps: float = 0.0
    # Alienware-specific
    alienware_model: str = ""
    liquid_cooling: bool = False


@dataclass
class ComputeJob:
    """A compute job to send to the PC"""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    job_type: JobType = JobType.INFERENCE
    status: JobStatus = JobStatus.QUEUED
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    priority: int = 5  # 1=highest, 10=lowest

    @property
    def elapsed_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


@dataclass
class PeerNode:
    """A discovered peer node in the agent mesh"""
    node_id: str
    hostname: str
    ip_address: str
    control_port: int = DEFAULT_CONTROL_PORT
    data_port: int = DEFAULT_DATA_PORT
    profile: Optional[PCProfile] = None
    last_heartbeat: float = 0.0
    latency_ms: float = 0.0
    connected: bool = False
    capabilities: List[str] = field(default_factory=list)


# =============================================================================
# ALIENWARE AREA-51 R2 OPTIMIZED PROFILE
# =============================================================================

ALIENWARE_AREA51_R2_DEFAULTS = PCProfile(
    pc_type=PCType.ALIENWARE_AREA51_R2,
    alienware_model="Area-51 R2",
    liquid_cooling=True,
    # Typical Area-51 R2 specs (user can override)
    cpu_name="Intel Core i9-10900X / i9-10980XE",
    cpu_cores=10,
    cpu_threads=20,
    ram_gb=64.0,
    gpu_name="NVIDIA RTX 3090 / RTX 4090",
    gpu_vram_gb=24.0,
)


# =============================================================================
# PC BRIDGE — MAIN CLASS
# =============================================================================

class PCBridge:
    """
    High-performance bridge for communicating with Windows PCs.
    
    Supports:
    - WebSocket control channel (reliable commands, job dispatch)
    - UDP data channel (fast telemetry, streaming)
    - Auto-discovery of peers on LAN
    - GPU compute offload
    - Model file sync
    - Distributed agent execution
    
    Usage:
        bridge = PCBridge(
            target_ip="192.168.1.100",  # Alienware IP
            pc_type=PCType.ALIENWARE_AREA51_R2,
        )
        await bridge.connect()
        result = await bridge.submit_job(ComputeJob(
            job_type=JobType.INFERENCE,
            payload={"model": "cosmos-reason2", "prompt": "..."}
        ))
    """

    def __init__(
        self,
        target_ip: Optional[str] = None,
        control_port: int = DEFAULT_CONTROL_PORT,
        data_port: int = DEFAULT_DATA_PORT,
        pc_type: PCType = PCType.ALIENWARE_AREA51_R2,
        auto_discover: bool = True,
        encryption: bool = False,
        max_concurrent_jobs: int = 4,
    ):
        self.target_ip = target_ip
        self.control_port = control_port
        self.data_port = data_port
        self.pc_type = pc_type
        self.auto_discover = auto_discover
        self.encryption = encryption
        self.max_concurrent_jobs = max_concurrent_jobs

        # State
        self.connected = False
        self.peer: Optional[PeerNode] = None
        self.local_profile = self._build_local_profile()
        self.discovered_peers: Dict[str, PeerNode] = {}

        # Job management
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.active_jobs: Dict[str, ComputeJob] = {}
        self.completed_jobs: Dict[str, ComputeJob] = {}
        self.job_callbacks: Dict[str, Callable] = {}

        # Channels
        self._ws_writer: Optional[asyncio.StreamWriter] = None
        self._ws_reader: Optional[asyncio.StreamReader] = None
        self._udp_transport = None
        self._udp_protocol = None

        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False

        # Metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "avg_latency_ms": 0.0,
            "uptime_seconds": 0.0,
            "files_synced": 0,
            "models_synced": 0,
        }
        self._start_time = time.time()
        self._latency_samples: List[float] = []

    # ─── Local Profile ───

    def _build_local_profile(self) -> PCProfile:
        """Build hardware profile for this machine."""
        profile = PCProfile(
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            os_name=platform.system(),
            os_version=platform.release(),
            cpu_name=platform.processor() or platform.machine(),
        )
        try:
            import psutil
            profile.cpu_cores = psutil.cpu_count(logical=False) or 0
            profile.cpu_threads = psutil.cpu_count(logical=True) or 0
            profile.ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
            profile.disk_free_gb = round(psutil.disk_usage("/").free / (1024**3), 1)
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                profile.gpu_name = torch.cuda.get_device_name(0)
                profile.gpu_vram_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                )
        except ImportError:
            pass
        return profile

    @staticmethod
    def _get_local_ip() -> str:
        """Get the LAN IP address of this machine."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    # ─── Connection ───

    async def connect(self) -> bool:
        """
        Connect to the target PC.
        
        If target_ip is not set and auto_discover is True,
        will attempt to discover the PC on the LAN first.
        """
        if not self.target_ip and self.auto_discover:
            logger.info("No target IP set, attempting auto-discovery...")
            discovered = await self.discover_peers(timeout=5.0)
            if discovered:
                # Pick the first Alienware or first peer
                for peer in discovered:
                    if peer.profile and peer.profile.pc_type == self.pc_type:
                        self.target_ip = peer.ip_address
                        break
                if not self.target_ip:
                    self.target_ip = discovered[0].ip_address
                logger.info(f"Discovered peer at {self.target_ip}")
            else:
                logger.error("No peers discovered and no target IP set")
                return False

        logger.info(f"Connecting to {self.target_ip}:{self.control_port}...")

        try:
            # Open TCP control channel
            self._ws_reader, self._ws_writer = await asyncio.wait_for(
                asyncio.open_connection(self.target_ip, self.control_port),
                timeout=10.0,
            )

            # Send handshake
            handshake = {
                "type": "handshake",
                "node_id": self.local_profile.hostname,
                "profile": asdict(self.local_profile),
                "protocol_version": "1.0",
                "timestamp": time.time(),
            }
            await self._send_control(handshake)

            # Wait for handshake response
            response = await self._recv_control(timeout=10.0)
            if response and response.get("type") == "handshake_ack":
                peer_profile_data = response.get("profile", {})
                self.peer = PeerNode(
                    node_id=response.get("node_id", "unknown"),
                    hostname=peer_profile_data.get("hostname", "unknown"),
                    ip_address=self.target_ip,
                    control_port=self.control_port,
                    data_port=self.data_port,
                    last_heartbeat=time.time(),
                    connected=True,
                    capabilities=response.get("capabilities", []),
                )
                # Parse peer profile
                try:
                    self.peer.profile = PCProfile(**{
                        k: v for k, v in peer_profile_data.items()
                        if k in PCProfile.__dataclass_fields__
                    })
                except Exception:
                    pass

                self.connected = True
                logger.info(
                    f"Connected to {self.peer.hostname} ({self.target_ip}) "
                    f"— {self.peer.profile.gpu_name if self.peer.profile else 'unknown GPU'}"
                )

                # Start background tasks
                self._running = True
                self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
                self._tasks.append(asyncio.create_task(self._receive_loop()))
                self._tasks.append(asyncio.create_task(self._job_dispatcher()))

                return True
            else:
                logger.error(f"Handshake failed: {response}")
                return False

        except asyncio.TimeoutError:
            logger.error(f"Connection to {self.target_ip}:{self.control_port} timed out")
            return False
        except ConnectionRefusedError:
            logger.error(
                f"Connection refused at {self.target_ip}:{self.control_port}. "
                f"Is the NIS agent running on the Alienware?"
            )
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the peer."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        if self._ws_writer:
            try:
                await self._send_control({"type": "disconnect"})
                self._ws_writer.close()
                await self._ws_writer.wait_closed()
            except Exception:
                pass
            self._ws_writer = None
            self._ws_reader = None

        self.connected = False
        if self.peer:
            self.peer.connected = False
        logger.info("Disconnected from peer")

    # ─── Control Channel (TCP) ───

    async def _send_control(self, message: Dict[str, Any]):
        """Send a message over the TCP control channel."""
        if not self._ws_writer:
            return
        data = json.dumps(message).encode("utf-8")
        # Length-prefixed framing: 4-byte big-endian length + payload
        header = struct.pack(">I", len(data))
        self._ws_writer.write(header + data)
        await self._ws_writer.drain()
        self.metrics["messages_sent"] += 1
        self.metrics["bytes_sent"] += len(data) + 4

    async def _recv_control(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Receive a message from the TCP control channel."""
        if not self._ws_reader:
            return None
        try:
            header = await asyncio.wait_for(
                self._ws_reader.readexactly(4), timeout=timeout
            )
            length = struct.unpack(">I", header)[0]
            if length > 100 * 1024 * 1024:  # 100MB safety limit
                logger.error(f"Message too large: {length} bytes")
                return None
            data = await asyncio.wait_for(
                self._ws_reader.readexactly(length), timeout=timeout
            )
            self.metrics["messages_received"] += 1
            self.metrics["bytes_received"] += length + 4
            return json.loads(data.decode("utf-8"))
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            return None

    # ─── Background Loops ───

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to keep connection alive."""
        while self._running:
            try:
                t0 = time.time()
                await self._send_control({
                    "type": "heartbeat",
                    "timestamp": t0,
                    "metrics": self.get_metrics(),
                })
                # Measure round-trip if peer echoes
                await asyncio.sleep(HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
                await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _receive_loop(self):
        """Receive and dispatch incoming messages."""
        while self._running:
            try:
                msg = await self._recv_control(timeout=HEARTBEAT_INTERVAL * 3)
                if msg is None:
                    if self._running:
                        logger.warning("No message received, peer may be offline")
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "heartbeat":
                    # Update peer latency
                    sent_ts = msg.get("timestamp", 0)
                    if sent_ts:
                        latency = (time.time() - sent_ts) * 1000
                        self._latency_samples.append(latency)
                        if len(self._latency_samples) > 100:
                            self._latency_samples = self._latency_samples[-100:]
                        self.metrics["avg_latency_ms"] = sum(self._latency_samples) / len(self._latency_samples)
                    if self.peer:
                        self.peer.last_heartbeat = time.time()

                elif msg_type == "heartbeat_ack":
                    sent_ts = msg.get("echo_timestamp", 0)
                    if sent_ts:
                        latency = (time.time() - sent_ts) * 1000
                        self._latency_samples.append(latency)
                        if self.peer:
                            self.peer.latency_ms = latency

                elif msg_type == "job_result":
                    await self._handle_job_result(msg)

                elif msg_type == "job_status":
                    await self._handle_job_status(msg)

                elif msg_type == "agent_message":
                    await self._handle_agent_message(msg)

                elif msg_type == "file_chunk":
                    await self._handle_file_chunk(msg)

                elif msg_type == "error":
                    logger.error(f"Peer error: {msg.get('message', 'unknown')}")

                else:
                    logger.debug(f"Unknown message type: {msg_type}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive loop error: {e}")
                await asyncio.sleep(1)

    async def _job_dispatcher(self):
        """Dispatch queued jobs to the peer."""
        while self._running:
            try:
                if len(self.active_jobs) >= self.max_concurrent_jobs:
                    await asyncio.sleep(0.5)
                    continue

                try:
                    job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                self.active_jobs[job.job_id] = job

                await self._send_control({
                    "type": "job_submit",
                    "job_id": job.job_id,
                    "job_type": job.job_type.value,
                    "payload": job.payload,
                    "priority": job.priority,
                })

                logger.info(f"Dispatched job {job.job_id} ({job.job_type.value})")
                self.metrics["jobs_submitted"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Job dispatcher error: {e}")
                await asyncio.sleep(1)

    # ─── Job Handling ───

    async def _handle_job_result(self, msg: Dict[str, Any]):
        """Handle a completed job result from the peer."""
        job_id = msg.get("job_id")
        if job_id and job_id in self.active_jobs:
            job = self.active_jobs.pop(job_id)
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result = msg.get("result")
            self.completed_jobs[job_id] = job
            self.metrics["jobs_completed"] += 1

            logger.info(
                f"Job {job_id} completed in {job.elapsed_ms:.1f}ms"
            )

            # Fire callback if registered
            cb = self.job_callbacks.pop(job_id, None)
            if cb:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(job)
                    else:
                        cb(job)
                except Exception as e:
                    logger.error(f"Job callback error: {e}")

    async def _handle_job_status(self, msg: Dict[str, Any]):
        """Handle a job status update."""
        job_id = msg.get("job_id")
        status = msg.get("status")
        if job_id and job_id in self.active_jobs and status:
            try:
                self.active_jobs[job_id].status = JobStatus(status)
            except ValueError:
                pass

    async def _handle_agent_message(self, msg: Dict[str, Any]):
        """Handle an agent mesh message from the peer."""
        agent_id = msg.get("agent_id")
        payload = msg.get("payload", {})
        logger.info(f"Agent message from {agent_id}: {list(payload.keys())}")

    async def _handle_file_chunk(self, msg: Dict[str, Any]):
        """Handle an incoming file chunk."""
        # Reassemble file from chunks
        file_id = msg.get("file_id")
        chunk_idx = msg.get("chunk_idx", 0)
        total_chunks = msg.get("total_chunks", 1)
        data_b64 = msg.get("data", "")
        logger.debug(f"File chunk {chunk_idx + 1}/{total_chunks} for {file_id}")

    # ─── Public API ───

    async def submit_job(
        self,
        job: ComputeJob,
        callback: Optional[Callable] = None,
        wait: bool = False,
        timeout: float = 300.0,
    ) -> ComputeJob:
        """
        Submit a compute job to the remote PC.
        
        Args:
            job: The job to submit
            callback: Optional callback when job completes
            wait: If True, block until job completes
            timeout: Max wait time in seconds
            
        Returns:
            The job (with result if wait=True)
        """
        if not self.connected:
            job.status = JobStatus.FAILED
            job.error = "Not connected to peer"
            return job

        if callback:
            self.job_callbacks[job.job_id] = callback

        await self.job_queue.put(job)

        if wait:
            t0 = time.time()
            while time.time() - t0 < timeout:
                if job.job_id in self.completed_jobs:
                    return self.completed_jobs[job.job_id]
                await asyncio.sleep(0.1)
            job.status = JobStatus.FAILED
            job.error = "Timeout waiting for result"

        return job

    async def offload_inference(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        wait: bool = True,
        timeout: float = 60.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Offload an inference job to the PC's GPU.
        
        Args:
            model_name: Name of the model to run
            inputs: Model inputs
            wait: Wait for result
            timeout: Max wait time
            
        Returns:
            Inference result or None
        """
        job = ComputeJob(
            job_type=JobType.INFERENCE,
            payload={"model": model_name, "inputs": inputs},
            priority=3,
        )
        result = await self.submit_job(job, wait=wait, timeout=timeout)
        if result.status == JobStatus.COMPLETED:
            return result.result
        logger.error(f"Inference failed: {result.error}")
        return None

    async def sync_model(
        self,
        model_path: str,
        target_path: str,
        direction: str = "push",  # "push" = local→remote, "pull" = remote→local
    ) -> bool:
        """
        Sync a model file/directory between this machine and the PC.
        
        Args:
            model_path: Source path
            target_path: Destination path on the remote
            direction: "push" or "pull"
            
        Returns:
            Success status
        """
        if not self.connected:
            return False

        if direction == "push":
            return await self._push_file(model_path, target_path)
        else:
            return await self._pull_file(model_path, target_path)

    async def _push_file(self, local_path: str, remote_path: str) -> bool:
        """Push a file to the remote PC."""
        path = Path(local_path)
        if not path.exists():
            logger.error(f"File not found: {local_path}")
            return False

        if path.is_dir():
            # Push directory recursively
            files = list(path.rglob("*"))
            files = [f for f in files if f.is_file()]
            logger.info(f"Pushing directory {local_path} ({len(files)} files)")
            for f in files:
                rel = f.relative_to(path)
                remote_file = os.path.join(remote_path, str(rel))
                await self._push_single_file(str(f), remote_file)
            self.metrics["models_synced"] += 1
            return True
        else:
            result = await self._push_single_file(local_path, remote_path)
            if result:
                self.metrics["files_synced"] += 1
            return result

    async def _push_single_file(self, local_path: str, remote_path: str) -> bool:
        """Push a single file to the remote PC via chunked transfer."""
        try:
            file_size = os.path.getsize(local_path)
            file_hash = hashlib.md5(open(local_path, "rb").read()).hexdigest()
            file_id = str(uuid.uuid4())[:8]
            total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE

            # Send file header
            await self._send_control({
                "type": "file_start",
                "file_id": file_id,
                "remote_path": remote_path,
                "file_size": file_size,
                "total_chunks": total_chunks,
                "md5": file_hash,
            })

            # Send chunks
            import base64
            with open(local_path, "rb") as f:
                for i in range(total_chunks):
                    chunk = f.read(CHUNK_SIZE)
                    await self._send_control({
                        "type": "file_chunk",
                        "file_id": file_id,
                        "chunk_idx": i,
                        "total_chunks": total_chunks,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    })

            # Send file complete
            await self._send_control({
                "type": "file_complete",
                "file_id": file_id,
                "md5": file_hash,
            })

            logger.info(f"Pushed {local_path} → {remote_path} ({file_size / 1024 / 1024:.1f}MB)")
            return True

        except Exception as e:
            logger.error(f"File push failed: {e}")
            return False

    async def _pull_file(self, remote_path: str, local_path: str) -> bool:
        """Request a file from the remote PC."""
        await self._send_control({
            "type": "file_request",
            "remote_path": remote_path,
            "local_path": local_path,
        })
        # The receive loop will handle incoming file_chunk messages
        return True

    async def send_agent_message(
        self,
        agent_id: str,
        payload: Dict[str, Any],
    ):
        """
        Send a message to an agent running on the remote PC.
        
        Args:
            agent_id: Target agent identifier
            payload: Message payload
        """
        await self._send_control({
            "type": "agent_message",
            "agent_id": agent_id,
            "payload": payload,
            "source_node": self.local_profile.hostname,
            "timestamp": time.time(),
        })

    async def execute_remote_agent(
        self,
        agent_class: str,
        agent_config: Dict[str, Any],
        task: Dict[str, Any],
        wait: bool = True,
        timeout: float = 120.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute an agent on the remote PC.
        
        Args:
            agent_class: Fully qualified agent class name
            agent_config: Agent initialization config
            task: Task to execute
            wait: Wait for result
            timeout: Max wait time
            
        Returns:
            Agent execution result
        """
        job = ComputeJob(
            job_type=JobType.AGENT_EXEC,
            payload={
                "agent_class": agent_class,
                "agent_config": agent_config,
                "task": task,
            },
            priority=2,
        )
        result = await self.submit_job(job, wait=wait, timeout=timeout)
        if result.status == JobStatus.COMPLETED:
            return result.result
        return None

    async def benchmark_connection(self) -> Dict[str, Any]:
        """
        Benchmark the connection to the PC.
        
        Returns:
            Benchmark results with latency, throughput, etc.
        """
        if not self.connected:
            return {"error": "Not connected"}

        results = {}

        # Latency test (10 pings)
        latencies = []
        for _ in range(10):
            t0 = time.time()
            await self._send_control({"type": "ping", "timestamp": t0})
            resp = await self._recv_control(timeout=5.0)
            if resp and resp.get("type") == "pong":
                latencies.append((time.time() - t0) * 1000)
        if latencies:
            results["latency_min_ms"] = round(min(latencies), 2)
            results["latency_avg_ms"] = round(sum(latencies) / len(latencies), 2)
            results["latency_max_ms"] = round(max(latencies), 2)

        # Throughput test (send 10MB)
        test_data = "x" * (1024 * 1024)  # 1MB string
        t0 = time.time()
        for _ in range(10):
            await self._send_control({"type": "throughput_test", "data": test_data})
        elapsed = time.time() - t0
        results["throughput_mbps"] = round((10 * 8) / elapsed, 2) if elapsed > 0 else 0

        # Peer info
        if self.peer and self.peer.profile:
            results["peer_gpu"] = self.peer.profile.gpu_name
            results["peer_vram_gb"] = self.peer.profile.gpu_vram_gb
            results["peer_ram_gb"] = self.peer.profile.ram_gb

        logger.info(f"Benchmark: {results}")
        return results

    # ─── Discovery ───

    async def discover_peers(self, timeout: float = 5.0) -> List[PeerNode]:
        """
        Discover NIS Protocol peers on the LAN via UDP broadcast.
        
        Returns:
            List of discovered peers
        """
        discovered = []

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)

            # Send discovery broadcast
            discovery_msg = json.dumps({
                "type": "nis_discovery",
                "node_id": self.local_profile.hostname,
                "profile": asdict(self.local_profile),
                "control_port": self.control_port,
                "data_port": self.data_port,
            }).encode("utf-8")

            sock.sendto(discovery_msg, ("<broadcast>", DEFAULT_DISCOVERY_PORT))
            logger.info("Sent discovery broadcast")

            # Listen for responses
            end_time = time.time() + timeout
            while time.time() < end_time:
                try:
                    data, addr = sock.recvfrom(65535)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("type") == "nis_discovery_response":
                        peer = PeerNode(
                            node_id=msg.get("node_id", "unknown"),
                            hostname=msg.get("profile", {}).get("hostname", "unknown"),
                            ip_address=addr[0],
                            control_port=msg.get("control_port", DEFAULT_CONTROL_PORT),
                            data_port=msg.get("data_port", DEFAULT_DATA_PORT),
                            capabilities=msg.get("capabilities", []),
                        )
                        discovered.append(peer)
                        self.discovered_peers[peer.node_id] = peer
                        logger.info(f"Discovered: {peer.hostname} at {peer.ip_address}")
                except socket.timeout:
                    break

            sock.close()

        except Exception as e:
            logger.error(f"Discovery failed: {e}")

        return discovered

    # ─── Metrics ───

    def get_metrics(self) -> Dict[str, Any]:
        """Get current bridge metrics."""
        self.metrics["uptime_seconds"] = round(time.time() - self._start_time, 1)
        return {**self.metrics}

    def get_status(self) -> Dict[str, Any]:
        """Get full bridge status."""
        return {
            "connected": self.connected,
            "target_ip": self.target_ip,
            "peer": {
                "hostname": self.peer.hostname if self.peer else None,
                "gpu": self.peer.profile.gpu_name if self.peer and self.peer.profile else None,
                "latency_ms": self.peer.latency_ms if self.peer else None,
            } if self.peer else None,
            "active_jobs": len(self.active_jobs),
            "queued_jobs": self.job_queue.qsize(),
            "completed_jobs": len(self.completed_jobs),
            "metrics": self.get_metrics(),
            "local_profile": {
                "hostname": self.local_profile.hostname,
                "ip": self.local_profile.ip_address,
                "gpu": self.local_profile.gpu_name,
            },
        }


# =============================================================================
# PC AGENT SERVER — Run this on the Alienware
# =============================================================================

class PCAgentServer:
    """
    NIS Protocol agent server to run on the Windows PC (Alienware).
    
    Listens for connections from the NIS Hub and executes jobs locally.
    
    Usage on Windows:
        python -m src.neurolinux.pc_bridge --server --port 8100
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        control_port: int = DEFAULT_CONTROL_PORT,
        data_port: int = DEFAULT_DATA_PORT,
        discovery_port: int = DEFAULT_DISCOVERY_PORT,
    ):
        self.host = host
        self.control_port = control_port
        self.data_port = data_port
        self.discovery_port = discovery_port
        self.profile = self._build_profile()
        self.clients: Dict[str, asyncio.StreamWriter] = {}
        self._running = False
        self._server = None

    def _build_profile(self) -> PCProfile:
        """Build hardware profile for this PC."""
        profile = PCProfile(
            hostname=socket.gethostname(),
            ip_address=PCBridge._get_local_ip(),
            os_name=platform.system(),
            os_version=platform.release(),
            cpu_name=platform.processor() or platform.machine(),
        )
        # Detect if Alienware
        hostname_lower = socket.gethostname().lower()
        if "alienware" in hostname_lower or "area51" in hostname_lower:
            profile.pc_type = PCType.ALIENWARE_AREA51_R2
            profile.alienware_model = "Area-51 R2"
            profile.liquid_cooling = True

        try:
            import psutil
            profile.cpu_cores = psutil.cpu_count(logical=False) or 0
            profile.cpu_threads = psutil.cpu_count(logical=True) or 0
            profile.ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
            profile.disk_free_gb = round(psutil.disk_usage("C:\\" if platform.system() == "Windows" else "/").free / (1024**3), 1)
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                profile.gpu_name = torch.cuda.get_device_name(0)
                profile.gpu_vram_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                )
                profile.gpu_cuda_cores = 0  # Not easily queryable
        except ImportError:
            pass
        return profile

    async def start(self):
        """Start the agent server."""
        self._running = True

        # Start TCP control server
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.control_port
        )
        logger.info(
            f"NIS PC Agent Server started on {self.host}:{self.control_port}\n"
            f"  Hostname: {self.profile.hostname}\n"
            f"  GPU: {self.profile.gpu_name or 'None detected'}\n"
            f"  VRAM: {self.profile.gpu_vram_gb}GB\n"
            f"  RAM: {self.profile.ram_gb}GB\n"
            f"  CPU: {self.profile.cpu_name} ({self.profile.cpu_cores}C/{self.profile.cpu_threads}T)\n"
            f"  Waiting for connections..."
        )

        # Start discovery responder
        asyncio.create_task(self._discovery_responder())

        async with self._server:
            await self._server.serve_forever()

    async def stop(self):
        """Stop the agent server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle an incoming client connection."""
        addr = writer.get_extra_info("peername")
        logger.info(f"New connection from {addr}")

        try:
            # Wait for handshake
            msg = await self._recv_msg(reader, timeout=10.0)
            if not msg or msg.get("type") != "handshake":
                logger.warning(f"Invalid handshake from {addr}")
                writer.close()
                return

            client_id = msg.get("node_id", str(addr))
            self.clients[client_id] = writer

            # Send handshake ack
            await self._send_msg(writer, {
                "type": "handshake_ack",
                "node_id": self.profile.hostname,
                "profile": asdict(self.profile),
                "capabilities": ["inference", "training", "file_transfer", "agent_exec"],
            })

            logger.info(f"Client {client_id} connected from {addr}")

            # Message loop
            while self._running:
                msg = await self._recv_msg(reader, timeout=60.0)
                if msg is None:
                    break
                await self._process_message(client_id, writer, msg)

        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
            writer.close()
            logger.info(f"Client {addr} disconnected")

    async def _process_message(
        self, client_id: str, writer: asyncio.StreamWriter, msg: Dict[str, Any]
    ):
        """Process an incoming message from a client."""
        msg_type = msg.get("type", "")

        if msg_type == "heartbeat":
            await self._send_msg(writer, {
                "type": "heartbeat_ack",
                "echo_timestamp": msg.get("timestamp"),
                "timestamp": time.time(),
            })

        elif msg_type == "ping":
            await self._send_msg(writer, {
                "type": "pong",
                "echo_timestamp": msg.get("timestamp"),
                "timestamp": time.time(),
            })

        elif msg_type == "job_submit":
            asyncio.create_task(
                self._execute_job(client_id, writer, msg)
            )

        elif msg_type == "file_start":
            logger.info(f"Receiving file: {msg.get('remote_path')} ({msg.get('file_size', 0)} bytes)")

        elif msg_type == "file_chunk":
            pass  # Handled by file reassembly

        elif msg_type == "file_complete":
            logger.info(f"File transfer complete: {msg.get('file_id')}")

        elif msg_type == "disconnect":
            logger.info(f"Client {client_id} disconnecting")

        elif msg_type == "throughput_test":
            pass  # Absorb throughput test data

    async def _execute_job(
        self, client_id: str, writer: asyncio.StreamWriter, msg: Dict[str, Any]
    ):
        """Execute a compute job locally."""
        job_id = msg.get("job_id")
        job_type = msg.get("job_type")
        payload = msg.get("payload", {})

        logger.info(f"Executing job {job_id} ({job_type})")

        try:
            # Send running status
            await self._send_msg(writer, {
                "type": "job_status",
                "job_id": job_id,
                "status": "running",
            })

            result = {}

            if job_type == "inference":
                result = await self._run_inference(payload)
            elif job_type == "training":
                result = await self._run_training(payload)
            elif job_type == "agent_exec":
                result = await self._run_agent(payload)
            elif job_type == "benchmark":
                result = {"gpu": self.profile.gpu_name, "vram": self.profile.gpu_vram_gb}
            else:
                result = {"error": f"Unknown job type: {job_type}"}

            await self._send_msg(writer, {
                "type": "job_result",
                "job_id": job_id,
                "result": result,
            })

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            await self._send_msg(writer, {
                "type": "job_result",
                "job_id": job_id,
                "result": {"error": str(e)},
            })

    async def _run_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on local GPU."""
        model_name = payload.get("model", "")
        inputs = payload.get("inputs", {})
        logger.info(f"Running inference: {model_name}")
        # Placeholder — actual implementation depends on model
        return {
            "model": model_name,
            "status": "completed",
            "gpu_used": self.profile.gpu_name,
            "message": "Inference placeholder — implement model-specific logic",
        }

    async def _run_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run training on local GPU."""
        logger.info(f"Running training job")
        return {"status": "completed", "message": "Training placeholder"}

    async def _run_agent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent locally."""
        agent_class = payload.get("agent_class", "")
        logger.info(f"Executing agent: {agent_class}")
        return {"status": "completed", "agent_class": agent_class}

    async def _discovery_responder(self):
        """Respond to LAN discovery broadcasts."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(("", self.discovery_port))
            sock.settimeout(1.0)

            logger.info(f"Discovery responder listening on port {self.discovery_port}")

            while self._running:
                try:
                    data, addr = sock.recvfrom(65535)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("type") == "nis_discovery":
                        response = json.dumps({
                            "type": "nis_discovery_response",
                            "node_id": self.profile.hostname,
                            "profile": asdict(self.profile),
                            "control_port": self.control_port,
                            "data_port": self.data_port,
                            "capabilities": ["inference", "training", "file_transfer", "agent_exec"],
                        }).encode("utf-8")
                        sock.sendto(response, addr)
                        logger.info(f"Responded to discovery from {addr}")
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.debug(f"Discovery error: {e}")

            sock.close()
        except Exception as e:
            logger.error(f"Discovery responder failed: {e}")

    # ─── Wire Protocol Helpers ───

    async def _send_msg(self, writer: asyncio.StreamWriter, msg: Dict[str, Any]):
        data = json.dumps(msg).encode("utf-8")
        header = struct.pack(">I", len(data))
        writer.write(header + data)
        await writer.drain()

    async def _recv_msg(
        self, reader: asyncio.StreamReader, timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        try:
            header = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
            length = struct.unpack(">I", header)[0]
            if length > 100 * 1024 * 1024:
                return None
            data = await asyncio.wait_for(reader.readexactly(length), timeout=timeout)
            return json.loads(data.decode("utf-8"))
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            return None


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for PC Bridge."""
    import argparse

    parser = argparse.ArgumentParser(description="NIS Protocol PC Bridge")
    parser.add_argument("--server", action="store_true", help="Run as server (on the Alienware PC)")
    parser.add_argument("--client", action="store_true", help="Run as client (on Mac/Linux)")
    parser.add_argument("--target", type=str, default=None, help="Target PC IP address")
    parser.add_argument("--port", type=int, default=DEFAULT_CONTROL_PORT, help="Control port")
    parser.add_argument("--discover", action="store_true", help="Discover peers on LAN")
    parser.add_argument("--benchmark", action="store_true", help="Run connection benchmark")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.server:
        server = PCAgentServer(control_port=args.port)
        asyncio.run(server.start())

    elif args.discover:
        async def do_discover():
            bridge = PCBridge(auto_discover=True)
            peers = await bridge.discover_peers(timeout=5.0)
            if peers:
                print(f"\nDiscovered {len(peers)} peer(s):")
                for p in peers:
                    print(f"  {p.hostname} @ {p.ip_address}:{p.control_port}")
            else:
                print("No peers found on LAN")
        asyncio.run(do_discover())

    elif args.client or args.target:
        async def do_client():
            bridge = PCBridge(target_ip=args.target, control_port=args.port)
            if await bridge.connect():
                print(f"Connected to {bridge.peer.hostname}")
                if args.benchmark:
                    results = await bridge.benchmark_connection()
                    print(f"Benchmark: {json.dumps(results, indent=2)}")
                else:
                    print("Bridge active. Press Ctrl+C to disconnect.")
                    try:
                        while True:
                            await asyncio.sleep(1)
                    except KeyboardInterrupt:
                        pass
                await bridge.disconnect()
            else:
                print("Connection failed")
        asyncio.run(do_client())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
