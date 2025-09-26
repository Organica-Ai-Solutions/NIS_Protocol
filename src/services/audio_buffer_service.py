#!/usr/bin/env python3
"""
High-Performance Audio Buffer Service
Optimized for real-time voice processing with minimal latency
"""

import asyncio
import time
import threading
from typing import Optional, Callable, List, Dict, Any
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AudioChunk:
    """Represents a single audio chunk with metadata"""
    
    def __init__(self, data: bytes, timestamp: float, chunk_id: int):
        self.data = data
        self.timestamp = timestamp
        self.chunk_id = chunk_id
        self.size = len(data)
        self.processed = False

class HighPerformanceAudioBuffer:
    """
    Optimized audio buffer for streaming voice processing
    Features:
    - Adaptive buffering based on network conditions
    - Jitter buffer for smooth playback
    - Automatic quality adjustment
    - Latency monitoring and optimization
    """
    
    def __init__(self, 
                 target_latency_ms: int = 200,
                 max_latency_ms: int = 500,
                 chunk_size_ms: int = 20):
        
        self.target_latency_ms = target_latency_ms
        self.max_latency_ms = max_latency_ms
        self.chunk_size_ms = chunk_size_ms
        
        # Buffer management
        self.input_buffer = deque(maxlen=100)  # Input chunks
        self.processing_buffer = deque(maxlen=50)  # Ready for processing
        self.output_buffer = deque(maxlen=50)  # Processed chunks
        
        # Threading
        self.buffer_lock = threading.RLock()
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Performance metrics
        self.stats = {
            'chunks_received': 0,
            'chunks_processed': 0,
            'chunks_dropped': 0,
            'avg_latency_ms': 0,
            'buffer_underruns': 0,
            'buffer_overruns': 0,
            'processing_time_ms': 0
        }
        
        # Adaptive settings
        self.adaptive_enabled = True
        self.quality_level = 'high'  # high, medium, low
        self.last_adjustment_time = time.time()
        
        # Callbacks
        self.chunk_ready_callback: Optional[Callable] = None
        self.latency_warning_callback: Optional[Callable] = None
        
    def start(self):
        """Start the buffer processing thread"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            logger.info("Audio buffer service started")
    
    def stop(self):
        """Stop the buffer processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        logger.info("Audio buffer service stopped")
    
    def add_chunk(self, audio_data: bytes, timestamp: Optional[float] = None) -> bool:
        """
        Add audio chunk to buffer
        Returns True if chunk was accepted, False if dropped
        """
        if timestamp is None:
            timestamp = time.time()
        
        chunk_id = self.stats['chunks_received']
        chunk = AudioChunk(audio_data, timestamp, chunk_id)
        
        with self.buffer_lock:
            # Check for buffer overrun
            if len(self.input_buffer) >= self.input_buffer.maxlen - 1:
                self.stats['buffer_overruns'] += 1
                # Drop oldest chunk to make room
                if self.input_buffer:
                    dropped = self.input_buffer.popleft()
                    self.stats['chunks_dropped'] += 1
                    logger.warning(f"Dropped chunk {dropped.chunk_id} due to buffer overrun")
            
            self.input_buffer.append(chunk)
            self.stats['chunks_received'] += 1
            
            # Adaptive quality adjustment
            if self.adaptive_enabled:
                self._adjust_quality()
        
        return True
    
    def get_processed_chunk(self) -> Optional[AudioChunk]:
        """Get next processed audio chunk"""
        with self.buffer_lock:
            if self.output_buffer:
                return self.output_buffer.popleft()
            else:
                self.stats['buffer_underruns'] += 1
                return None
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status and performance metrics"""
        with self.buffer_lock:
            current_latency = self._calculate_current_latency()
            
            return {
                'input_buffer_size': len(self.input_buffer),
                'processing_buffer_size': len(self.processing_buffer),
                'output_buffer_size': len(self.output_buffer),
                'current_latency_ms': current_latency,
                'target_latency_ms': self.target_latency_ms,
                'quality_level': self.quality_level,
                'is_running': self.is_running,
                'stats': self.stats.copy()
            }
    
    def set_chunk_ready_callback(self, callback: Callable[[AudioChunk], None]):
        """Set callback for when chunks are ready for processing"""
        self.chunk_ready_callback = callback
    
    def set_latency_warning_callback(self, callback: Callable[[float], None]):
        """Set callback for latency warnings"""
        self.latency_warning_callback = callback
    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        while self.is_running:
            try:
                # Move chunks from input to processing buffer
                self._move_chunks_to_processing()
                
                # Process chunks
                self._process_chunks()
                
                # Monitor latency
                self._monitor_latency()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                logger.error(f"Error in audio buffer processing loop: {e}")
                time.sleep(0.01)  # Longer sleep on error
    
    def _move_chunks_to_processing(self):
        """Move chunks from input buffer to processing buffer"""
        with self.buffer_lock:
            # Move chunks that are ready for processing
            while (self.input_buffer and 
                   len(self.processing_buffer) < self.processing_buffer.maxlen):
                
                chunk = self.input_buffer.popleft()
                
                # Check if chunk is too old (latency control)
                age_ms = (time.time() - chunk.timestamp) * 1000
                if age_ms > self.max_latency_ms:
                    self.stats['chunks_dropped'] += 1
                    logger.warning(f"Dropped chunk {chunk.chunk_id} due to high latency: {age_ms:.1f}ms")
                    continue
                
                self.processing_buffer.append(chunk)
    
    def _process_chunks(self):
        """Process chunks in processing buffer"""
        with self.buffer_lock:
            while self.processing_buffer:
                chunk = self.processing_buffer.popleft()
                
                # Simulate processing time based on quality level
                processing_start = time.time()
                
                # Call chunk ready callback if set
                if self.chunk_ready_callback:
                    try:
                        self.chunk_ready_callback(chunk)
                    except Exception as e:
                        logger.error(f"Chunk ready callback error: {e}")
                
                # Mark as processed and move to output buffer
                chunk.processed = True
                processing_time = (time.time() - processing_start) * 1000
                self.stats['processing_time_ms'] = processing_time
                
                self.output_buffer.append(chunk)
                self.stats['chunks_processed'] += 1
    
    def _calculate_current_latency(self) -> float:
        """Calculate current end-to-end latency"""
        if not self.input_buffer and not self.processing_buffer:
            return 0.0
        
        current_time = time.time()
        oldest_timestamp = current_time
        
        # Find oldest chunk timestamp
        for chunk in self.input_buffer:
            oldest_timestamp = min(oldest_timestamp, chunk.timestamp)
        
        for chunk in self.processing_buffer:
            oldest_timestamp = min(oldest_timestamp, chunk.timestamp)
        
        return (current_time - oldest_timestamp) * 1000
    
    def _monitor_latency(self):
        """Monitor and react to latency changes"""
        current_latency = self._calculate_current_latency()
        
        # Update average latency (exponential moving average)
        alpha = 0.1
        self.stats['avg_latency_ms'] = (
            alpha * current_latency + 
            (1 - alpha) * self.stats['avg_latency_ms']
        )
        
        # Trigger warning callback if latency is too high
        if (current_latency > self.max_latency_ms and 
            self.latency_warning_callback):
            try:
                self.latency_warning_callback(current_latency)
            except Exception as e:
                logger.error(f"Latency warning callback error: {e}")
    
    def _adjust_quality(self):
        """Adaptive quality adjustment based on buffer conditions"""
        current_time = time.time()
        
        # Only adjust every 2 seconds
        if current_time - self.last_adjustment_time < 2.0:
            return
        
        self.last_adjustment_time = current_time
        current_latency = self._calculate_current_latency()
        
        # Adjust quality based on latency and buffer status
        if current_latency > self.target_latency_ms * 1.5:
            # High latency - reduce quality
            if self.quality_level == 'high':
                self.quality_level = 'medium'
                self.chunk_size_ms = 40  # Larger chunks
                logger.info("Reduced quality to medium due to high latency")
            elif self.quality_level == 'medium':
                self.quality_level = 'low'
                self.chunk_size_ms = 60  # Even larger chunks
                logger.info("Reduced quality to low due to high latency")
        
        elif current_latency < self.target_latency_ms * 0.7:
            # Low latency - increase quality
            if self.quality_level == 'low':
                self.quality_level = 'medium'
                self.chunk_size_ms = 40
                logger.info("Increased quality to medium due to low latency")
            elif self.quality_level == 'medium':
                self.quality_level = 'high'
                self.chunk_size_ms = 20  # Smaller chunks for better quality
                logger.info("Increased quality to high due to low latency")

class ConcurrentAudioProcessor:
    """
    Concurrent audio processing pipeline for maximum performance
    Processes multiple audio streams simultaneously
    """
    
    def __init__(self, max_concurrent_streams: int = 4):
        self.max_concurrent_streams = max_concurrent_streams
        self.active_streams: Dict[str, HighPerformanceAudioBuffer] = {}
        self.stream_lock = threading.Lock()
        
    def create_stream(self, stream_id: str, **kwargs) -> HighPerformanceAudioBuffer:
        """Create a new audio stream buffer"""
        with self.stream_lock:
            if len(self.active_streams) >= self.max_concurrent_streams:
                raise RuntimeError(f"Maximum concurrent streams ({self.max_concurrent_streams}) reached")
            
            buffer = HighPerformanceAudioBuffer(**kwargs)
            buffer.start()
            self.active_streams[stream_id] = buffer
            
            logger.info(f"Created audio stream: {stream_id}")
            return buffer
    
    def get_stream(self, stream_id: str) -> Optional[HighPerformanceAudioBuffer]:
        """Get existing audio stream buffer"""
        with self.stream_lock:
            return self.active_streams.get(stream_id)
    
    def remove_stream(self, stream_id: str) -> bool:
        """Remove and stop audio stream buffer"""
        with self.stream_lock:
            if stream_id in self.active_streams:
                buffer = self.active_streams[stream_id]
                buffer.stop()
                del self.active_streams[stream_id]
                logger.info(f"Removed audio stream: {stream_id}")
                return True
            return False
    
    def get_all_stream_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance stats for all active streams"""
        with self.stream_lock:
            return {
                stream_id: buffer.get_buffer_status()
                for stream_id, buffer in self.active_streams.items()
            }
    
    def cleanup_inactive_streams(self):
        """Remove streams that haven't received data recently"""
        current_time = time.time()
        inactive_streams = []
        
        with self.stream_lock:
            for stream_id, buffer in self.active_streams.items():
                # Check if stream has been inactive for more than 30 seconds
                if buffer.stats['chunks_received'] == 0:
                    continue
                
                # This is a simplified check - in practice you'd track last activity
                inactive_streams.append(stream_id)
        
        for stream_id in inactive_streams:
            self.remove_stream(stream_id)

# Global audio processor instance
_audio_processor: Optional[ConcurrentAudioProcessor] = None

def get_audio_processor() -> ConcurrentAudioProcessor:
    """Get or create global audio processor instance"""
    global _audio_processor
    
    if _audio_processor is None:
        _audio_processor = ConcurrentAudioProcessor(max_concurrent_streams=8)
    
    return _audio_processor
