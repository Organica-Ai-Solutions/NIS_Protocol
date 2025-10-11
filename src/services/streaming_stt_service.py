#!/usr/bin/env python3
"""
High-Performance Streaming Speech-to-Text Service
Optimized for <500ms latency with partial transcription
"""

import asyncio
import io
import time
import tempfile
import threading
from typing import AsyncGenerator, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)

class StreamingSTTService:
    """
    High-performance streaming STT with partial transcription
    Optimized for real-time voice chat with <500ms latency
    """
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.model: Optional[WhisperModel] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.is_processing = False
        
        # Performance optimization settings
        self.chunk_duration = 0.5  # 500ms chunks for low latency
        self.overlap_duration = 0.1  # 100ms overlap for continuity
        self.min_audio_length = 0.3  # Minimum 300ms for processing
        self.sample_rate = 16000
        
        # Partial transcription settings
        self.partial_threshold = 0.7  # Confidence threshold for partial results
        self.last_partial_text = ""
        self.partial_callback: Optional[Callable] = None
        
    async def initialize(self):
        """Initialize the Whisper model asynchronously"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            start_time = time.time()
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model
            )
            
            load_time = time.time() - start_time
            logger.info(f"Whisper model loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize STT model: {e}")
            return False
    
    def _load_model(self) -> WhisperModel:
        """Load Whisper model in thread"""
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type="int8",  # Faster inference
            cpu_threads=4,  # Optimize for multi-core
            num_workers=1  # Single worker for lower memory
        )
    
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[dict]:
        """
        Process audio chunk with streaming transcription
        Returns partial or final transcription results
        """
        if not self.model:
            await self.initialize()
        
        try:
            # Add to buffer
            with self.buffer_lock:
                self.audio_buffer.append(audio_data)
            
            # Check if we have enough audio to process
            total_duration = len(self.audio_buffer) * 0.2  # Assuming 200ms chunks
            
            if total_duration >= self.min_audio_length:
                return await self._transcribe_buffer()
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {"error": str(e)}
    
    async def _transcribe_buffer(self) -> dict:
        """Transcribe current audio buffer"""
        if self.is_processing:
            return None  # Skip if already processing
        
        self.is_processing = True
        start_time = time.time()
        
        try:
            # Get audio data from buffer
            with self.buffer_lock:
                if not self.audio_buffer:
                    return None
                
                # Combine audio chunks
                combined_audio = b''.join(self.audio_buffer)
                # Keep some overlap for continuity
                overlap_chunks = max(1, int(len(self.audio_buffer) * 0.2))
                self.audio_buffer = self.audio_buffer[-overlap_chunks:]
            
            # Convert to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(combined_audio)
                temp_file.flush()
                
                # Transcribe in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._transcribe_file,
                    temp_file.name
                )
            
            processing_time = time.time() - start_time
            
            # Prepare result
            transcription_result = {
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "is_partial": result.get("is_partial", False),
                "processing_time_ms": int(processing_time * 1000),
                "timestamp": time.time()
            }
            
            # Handle partial transcription callback
            if self.partial_callback and result.get("text"):
                try:
                    self.partial_callback(transcription_result)
                except Exception as e:
                    logger.error(f"Partial callback error: {e}")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": str(e), "processing_time_ms": int((time.time() - start_time) * 1000)}
        
        finally:
            self.is_processing = False
    
    def _transcribe_file(self, file_path: str) -> dict:
        """Transcribe audio file using Whisper"""
        try:
            segments, info = self.model.transcribe(
                file_path,
                beam_size=1,  # Faster inference
                best_of=1,    # Single pass
                temperature=0.0,  # Deterministic
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,  # Faster for streaming
                word_timestamps=False,  # Skip word-level timing
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )
            
            # Combine segments
            text_parts = []
            total_confidence = 0.0  # Accumulator for averaging
            segment_count = 0
            
            for segment in segments:
                text_parts.append(segment.text.strip())
                total_confidence += segment.avg_logprob
                segment_count += 1
            
            final_text = " ".join(text_parts).strip()
            avg_confidence = total_confidence / max(segment_count, 1)

            # Convert log probability to probability score (0-1)
            confidence = float(np.clip(np.exp(avg_confidence), 0.0, 1.0))
            
            return {
                "text": final_text,
                "confidence": confidence,
                "is_partial": confidence < self.partial_threshold,
                "segments": segment_count
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return {"text": "", "confidence": None, "error": str(e)}
    
    def set_partial_callback(self, callback: Callable):
        """Set callback for partial transcription results"""
        self.partial_callback = callback
    
    def clear_buffer(self):
        """Clear audio buffer"""
        with self.buffer_lock:
            self.audio_buffer.clear()
    
    async def finalize_transcription(self) -> Optional[dict]:
        """Process remaining audio in buffer for final transcription"""
        if self.audio_buffer:
            return await self._transcribe_buffer()
        return None
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        with self.buffer_lock:
            buffer_size = len(self.audio_buffer)
            buffer_duration = buffer_size * 0.2  # Assuming 200ms chunks
        
        return {
            "model_size": self.model_size,
            "device": self.device,
            "buffer_size": buffer_size,
            "buffer_duration_ms": int(buffer_duration * 1000),
            "is_processing": self.is_processing,
            "chunk_duration_ms": int(self.chunk_duration * 1000),
            "min_audio_length_ms": int(self.min_audio_length * 1000)
        }

class AudioBuffer:
    """
    Optimized audio buffer for streaming processing
    Handles variable chunk sizes and maintains audio continuity
    """
    
    def __init__(self, max_duration: float = 5.0, sample_rate: int = 16000):
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
    
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to buffer"""
        with self.lock:
            self.buffer = np.concatenate([self.buffer, audio_data])
            
            # Trim buffer if too long
            if len(self.buffer) > self.max_samples:
                excess = len(self.buffer) - self.max_samples
                self.buffer = self.buffer[excess:]
    
    def get_audio(self, duration: float) -> np.ndarray:
        """Get audio data for specified duration"""
        samples_needed = int(duration * self.sample_rate)
        
        with self.lock:
            if len(self.buffer) >= samples_needed:
                audio = self.buffer[:samples_needed].copy()
                self.buffer = self.buffer[samples_needed:]
                return audio
            else:
                # Return all available audio
                audio = self.buffer.copy()
                self.buffer = np.array([], dtype=np.float32)
                return audio
    
    def get_duration(self) -> float:
        """Get current buffer duration in seconds"""
        with self.lock:
            return len(self.buffer) / self.sample_rate
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)

# Global STT service instance
_stt_service: Optional[StreamingSTTService] = None

async def get_stt_service() -> StreamingSTTService:
    """Get or create global STT service instance"""
    global _stt_service
    
    if _stt_service is None:
        _stt_service = StreamingSTTService(model_size="base", device="cpu")
        await _stt_service.initialize()
    
    return _stt_service

async def transcribe_audio_stream(audio_data: bytes) -> Optional[dict]:
    """Convenience function for streaming transcription"""
    stt_service = await get_stt_service()
    return await stt_service.process_audio_chunk(audio_data)
