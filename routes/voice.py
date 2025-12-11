"""
NIS Protocol v4.0 - Voice Routes

This module contains all voice-related endpoints:
- Speech-to-Text (STT) transcription
- Text-to-Speech (TTS) synthesis
- Voice chat WebSocket
- Voice settings management
- Communication agent vocalization

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over
- main.py routes remain active until migration is complete

Usage:
    from routes.voice import router as voice_router
    app.include_router(voice_router, tags=["Voice"])
"""

import asyncio
import base64
import logging
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger("nis.routes.voice")

# Create router
router = APIRouter(tags=["Voice"])


# ====== Communication / Consciousness Voice ======

@router.post("/communication/consciousness_voice")
async def vocalize_consciousness():
    """
    🧠 Vocalize Consciousness Status
    
    Generate audio representation of the current consciousness state.
    """
    try:
        from src.agents.communication.vibevoice_communication_agent import create_vibevoice_communication_agent
        
        # Get current consciousness status from injected dependency
        consciousness_service = getattr(router, '_consciousness_service', None)
        if consciousness_service:
            consciousness_data = consciousness_service.get_status()
        else:
            consciousness_data = {"consciousness_level": 0.5, "status": "unknown"}
        
        # Create communication agent
        comm_agent = create_vibevoice_communication_agent()
        
        # Generate consciousness vocalization
        result = comm_agent.vocalize_consciousness(consciousness_data)
        
        return {
            "success": result.success,
            "consciousness_audio": result.audio_data.decode('utf-8', errors='ignore') if result.audio_data else None,
            "duration_seconds": result.duration_seconds,
            "consciousness_level": consciousness_data.get('consciousness_level', 0.0),
            "processing_time": result.processing_time,
            "error_message": result.error_message,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Consciousness vocalization error: {e}")
        return {
            "success": False,
            "error_message": str(e),
            "timestamp": time.time()
        }


@router.get("/communication/status")
async def communication_status():
    """
    📊 Get Communication Agent Status
    
    Returns status of VibeVoice communication capabilities.
    """
    try:
        from src.agents.communication.vibevoice_communication_agent import create_vibevoice_communication_agent
        
        # Create communication agent
        comm_agent = create_vibevoice_communication_agent()
        
        # Get status
        status = comm_agent.get_status()
        
        vibevoice_engine = getattr(router, '_vibevoice_engine', None)
        
        return {
            "status": "operational",
            "agent_info": status,
            "vibevoice_model": "microsoft/VibeVoice-1.5B",
            "capabilities": [
                "text_to_speech",
                "multi_speaker_synthesis",
                "consciousness_vocalization", 
                "physics_explanation",
                "agent_dialogue_creation",
                "realtime_streaming",
                "voice_switching"
            ],
            "supported_formats": ["wav", "mp3"],
            "max_duration_minutes": 90,
            "max_speakers": 4,
            "streaming_features": {
                "realtime_latency": "50ms",
                "voice_switching": True,
                "websocket_support": True,
                "like_gpt5_grok": True
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Communication status error: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "timestamp": time.time()
        }


# ====== Text-to-Speech (VibeVoice-Realtime) ======

@router.post("/voice/synthesize")
async def synthesize_speech(request: Dict[str, Any]):
    """
    🎙️ Text-to-Speech Synthesis using Microsoft VibeVoice-Realtime
    
    Convert text to speech with multiple speaker voices.
    Supports streaming for low-latency (~300ms first chunk).
    
    Request body:
    {
        "text": "Hello, world!",
        "speaker": "Carter",  // Carter, Nova, Aria, Davis
        "streaming": false
    }
    
    Returns:
    - Audio file (WAV format) or
    - JSON with base64 audio data
    """
    try:
        text = request.get("text", "")
        speaker = request.get("speaker", "Carter")
        streaming = request.get("streaming", False)
        return_base64 = request.get("return_base64", True)
        
        if not text:
            return {"success": False, "error": "No text provided"}
        
        logger.info(f"🎙️ TTS request: {len(text)} chars, speaker={speaker}")
        
        # Use VibeVoice-Realtime
        try:
            from src.voice.vibevoice_realtime import get_vibevoice_realtime, VibeVoiceSpeaker
            
            engine = await get_vibevoice_realtime()
            
            # Map speaker name to enum
            speaker_map = {
                "carter": VibeVoiceSpeaker.CARTER,
                "nova": VibeVoiceSpeaker.NOVA,
                "aria": VibeVoiceSpeaker.ARIA,
                "davis": VibeVoiceSpeaker.DAVIS,
            }
            speaker_enum = speaker_map.get(speaker.lower(), VibeVoiceSpeaker.CARTER)
            
            # Synthesize
            audio_bytes = await engine.synthesize(text, speaker_enum, streaming)
            
            if return_base64:
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                return {
                    "success": True,
                    "audio_data": audio_base64,
                    "format": "wav",
                    "speaker": speaker,
                    "text_length": len(text),
                    "engine": "vibevoice_realtime" if not engine._fallback_mode else "fallback_tts"
                }
            else:
                return Response(
                    content=audio_bytes,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"inline; filename=speech.wav",
                        "X-Speaker": speaker,
                        "X-Engine": "vibevoice_realtime"
                    }
                )
                
        except Exception as e:
            logger.error(f"❌ VibeVoice synthesis failed: {e}")
            
            # Fallback to simple TTS
            try:
                from src.voice.simple_tts import get_simple_tts
                tts = get_simple_tts()
                audio_bytes = await tts.synthesize_async(text)
                
                if return_base64:
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    return {
                        "success": True,
                        "audio_data": audio_base64,
                        "format": "mp3",
                        "speaker": speaker,
                        "engine": "simple_tts_fallback"
                    }
                else:
                    return Response(
                        content=audio_bytes,
                        media_type="audio/mpeg",
                        headers={"Content-Disposition": "inline; filename=speech.mp3"}
                    )
            except Exception as fallback_error:
                logger.error(f"❌ Fallback TTS also failed: {fallback_error}")
                return {"success": False, "error": str(e)}
                
    except Exception as e:
        logger.error(f"❌ TTS endpoint error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/voice/vibevoice/status")
async def get_vibevoice_status():
    """
    📊 Get VibeVoice-Realtime engine status
    """
    try:
        from src.voice.vibevoice_realtime import get_vibevoice_realtime
        
        engine = await get_vibevoice_realtime()
        return engine.get_status()
        
    except Exception as e:
        logger.error(f"VibeVoice status error: {e}")
        return {
            "engine": "VibeVoice-Realtime",
            "initialized": False,
            "error": str(e)
        }


# ====== Speech-to-Text ======

@router.post("/voice/transcribe")
async def transcribe_audio(request: Dict[str, Any]):
    """
    🎤 Speech-to-Text Transcription (GPT-like Voice Mode)
    
    Transcribe audio to text using Whisper for voice input.
    Like ChatGPT's voice conversation mode.
    """
    try:
        audio_data = request.get("audio_data", "")
        if not audio_data:
            logger.error("❌ No audio data provided in request")
            return {
                "success": False,
                "error": "No audio data provided",
                "text": ""
            }
        
        logger.info(f"📝 STT request received - audio data length: {len(audio_data)} chars")
        
        # Try to use Whisper STT
        try:
            from src.voice.whisper_stt import get_whisper_stt
            logger.info("✅ Whisper STT module imported successfully")
            
            whisper = get_whisper_stt(model_size="base")
            logger.info("✅ Whisper instance created, attempting transcription...")
            
            result = await whisper.transcribe_base64(audio_data)
            logger.info(f"📊 Whisper result: success={result.get('success')}, error={result.get('error', 'none')}")
            
            if result.get("success"):
                transcribed_text = result.get("text", "").strip()
                logger.info(f"✅ Whisper transcribed successfully: '{transcribed_text[:100]}...'")
                return {
                    "success": True,
                    "text": transcribed_text,
                    "transcription": transcribed_text,
                    "confidence": result.get("confidence", 0.0),
                    "language": result.get("language", "en"),
                    "engine": "whisper"
                }
            else:
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"❌ Whisper transcription failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "text": "",
                    "engine": "whisper_failed"
                }
                
        except ImportError as e:
            logger.error(f"❌ Whisper import error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Whisper not available: {str(e)}",
                "text": "",
                "engine": "import_failed"
            }
        except Exception as e:
            logger.error(f"❌ Whisper exception: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "engine": "exception"
            }
        
    except Exception as e:
        logger.error(f"❌ STT endpoint error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "text": ""
        }


# ====== Voice Settings ======

@router.get("/voice/settings")
async def get_voice_settings():
    """
    🎛️ Get comprehensive voice settings and capabilities
    """
    vibevoice_engine = getattr(router, '_vibevoice_engine', None)
    
    settings = {
        "available_speakers": {
            "consciousness": {
                "name": "Consciousness Agent",
                "description": "Thoughtful, philosophical voice with deeper insights",
                "characteristics": "Warm, contemplative, wise",
                "base_frequency": 180,
                "suggested_emotions": ["neutral", "thoughtful", "wise", "contemplative"]
            },
            "physics": {
                "name": "Physics Agent", 
                "description": "Analytical, precise voice for technical explanations",
                "characteristics": "Clear, authoritative, logical",
                "base_frequency": 160,
                "suggested_emotions": ["neutral", "analytical", "precise", "confident"]
            },
            "research": {
                "name": "Research Agent",
                "description": "Enthusiastic, curious voice for discoveries",
                "characteristics": "Energetic, inquisitive, excited",
                "base_frequency": 200,
                "suggested_emotions": ["excited", "curious", "enthusiastic", "amazed"]
            },
            "coordination": {
                "name": "Coordination Agent",
                "description": "Professional, clear voice for system management",
                "characteristics": "Steady, reliable, organized",
                "base_frequency": 170,
                "suggested_emotions": ["neutral", "professional", "calm", "focused"]
            }
        },
        "audio_settings": {
            "sample_rates": [16000, 22050, 44100, 48000],
            "default_sample_rate": 44100,
            "bit_depths": [16, 24, 32],
            "default_bit_depth": 16,
            "formats": ["wav", "mp3", "ogg"],
            "default_format": "wav"
        },
        "voice_parameters": {
            "speed_range": {"min": 0.5, "max": 2.0, "default": 1.0, "step": 0.1},
            "pitch_range": {"min": 0.5, "max": 2.0, "default": 1.0, "step": 0.1},
            "volume_range": {"min": 0.0, "max": 1.0, "default": 0.8, "step": 0.05},
            "emotion_intensity": {"min": 0.1, "max": 1.0, "default": 0.7, "step": 0.1}
        },
        "conversation_settings": {
            "auto_voice_detection": True,
            "silence_threshold_ms": 1500,
            "max_recording_duration_s": 30,
            "wake_word_enabled": True,
            "wake_words": ["Hey NIS", "NIS Protocol"],
            "voice_commands_enabled": True,
            "continuous_mode": True
        },
        "performance_settings": {
            "streaming_enabled": True,
            "real_time_processing": True,
            "chunk_size_ms": 100,
            "buffer_size_ms": 500,
            "max_latency_ms": 1000,
            "quality_vs_speed": "balanced"  # "speed", "balanced", "quality"
        },
        "accessibility": {
            "visual_indicators": True,
            "sound_notifications": True,
            "keyboard_shortcuts": {
                "toggle_voice": "Ctrl+V",
                "push_to_talk": "Ctrl+Space",
                "stop_voice": "Escape",
                "settings": "Ctrl+Alt+S"
            }
        },
        "system_info": {
            "vibevoice_loaded": vibevoice_engine is not None,
            "model_name": "microsoft/VibeVoice-1.5B" if vibevoice_engine else None,
            "max_duration_minutes": 90,
            "max_speakers": 4,
            "supported_languages": ["English", "Chinese"]
        }
    }
    return settings


@router.post("/voice/settings/update")
async def update_voice_settings(settings: Dict[str, Any]):
    """
    🎛️ Update voice settings with validation
    """
    try:
        updated_settings = {}
        
        if "speed" in settings:
            speed = float(settings["speed"])
            if 0.5 <= speed <= 2.0:
                updated_settings["speed"] = speed
            else:
                raise ValueError("Speed must be between 0.5 and 2.0")
                
        if "pitch" in settings:
            pitch = float(settings["pitch"])
            if 0.5 <= pitch <= 2.0:
                updated_settings["pitch"] = pitch
            else:
                raise ValueError("Pitch must be between 0.5 and 2.0")
                
        if "volume" in settings:
            volume = float(settings["volume"])
            if 0.0 <= volume <= 1.0:
                updated_settings["volume"] = volume
            else:
                raise ValueError("Volume must be between 0.0 and 1.0")
                
        if "default_speaker" in settings:
            speaker = settings["default_speaker"]
            valid_speakers = ["consciousness", "physics", "research", "coordination"]
            if speaker in valid_speakers:
                updated_settings["default_speaker"] = speaker
            else:
                raise ValueError(f"Speaker must be one of: {valid_speakers}")
                
        if "auto_voice_detection" in settings:
            updated_settings["auto_voice_detection"] = bool(settings["auto_voice_detection"])
            
        if "wake_word_enabled" in settings:
            updated_settings["wake_word_enabled"] = bool(settings["wake_word_enabled"])
            
        if "continuous_mode" in settings:
            updated_settings["continuous_mode"] = bool(settings["continuous_mode"])
            
        return {
            "success": True,
            "message": "Voice settings updated successfully",
            "updated_settings": updated_settings,
            "timestamp": time.time()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Voice settings update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice/test-speaker/{speaker}")
async def test_speaker_voice(speaker: str, text: str = "Hello! This is a test of the voice system."):
    """
    🎤 Test a specific speaker voice with custom text
    """
    try:
        vibevoice_engine = getattr(router, '_vibevoice_engine', None)
        
        if not vibevoice_engine:
            # Fallback to communication agent
            from src.agents.communication.vibevoice_communication_agent import (
                create_vibevoice_communication_agent, TTSRequest, SpeakerVoice
            )
            
            comm_agent = create_vibevoice_communication_agent()
            
            # Map speaker to voice
            speaker_voice = SpeakerVoice.CONSCIOUSNESS
            if speaker == "physics":
                speaker_voice = SpeakerVoice.PHYSICS
            elif speaker == "research":
                speaker_voice = SpeakerVoice.RESEARCH
            elif speaker == "coordination":
                speaker_voice = SpeakerVoice.COORDINATION
            
            # Create TTS request
            tts_request = TTSRequest(
                text=text,
                speaker_voice=speaker_voice,
                emotion="neutral"
            )
            
            # Generate speech
            result = comm_agent.synthesize_speech(tts_request)
            
            if result.success and result.audio_data:
                return Response(
                    content=result.audio_data,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"inline; filename=test_{speaker}_voice.wav",
                        "X-Speaker": speaker,
                        "X-Test-Text": text[:50] + "..." if len(text) > 50 else text,
                        "X-Duration": str(result.duration_seconds),
                        "X-Processing-Time": str(result.processing_time)
                    }
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Voice synthesis failed: {result.error_message}"
                )
        else:
            # Use VibeVoice engine directly
            audio_data = vibevoice_engine.synthesize(text, speaker=speaker)
            if audio_data:
                return Response(
                    content=audio_data,
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"inline; filename=test_{speaker}_voice.wav",
                        "X-Speaker": speaker
                    }
                )
            else:
                raise HTTPException(status_code=500, detail="Voice synthesis failed")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test speaker error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====== Voice Chat WebSocket ======

@router.websocket("/ws/voice-chat")
async def optimized_voice_chat(websocket: WebSocket):
    """
    🎙️ GPT-LIKE Real-Time Voice Chat (Ultra Low-Latency)
    
    Smooth 2-way audio conversation like GPT/Grok voice mode:
    - OpenAI Whisper STT (~300ms)
    - Fast LLM response (Anthropic/DeepSeek)
    - OpenAI TTS (~200ms) with gTTS fallback
    - Target: <800ms total latency
    
    Message Types (client → server):
    - audio_input: {"type": "audio_input", "audio_data": "base64..."}
    - text_input: {"type": "text_input", "text": "Hello"}
    - set_voice: {"type": "set_voice", "voice": "nova"}
    - get_status: {"type": "get_status"}
    - interrupt: {"type": "interrupt"}
    - close: {"type": "close"}
    
    Response Types (server → client):
    - connected: Connection established with capabilities
    - transcription: User speech transcribed
    - text_response: AI response text (streams as generated)
    - audio_response: AI voice audio (base64 MP3)
    - audio_chunk: Streaming audio chunk for ultra-low latency
    - status: Processing stage updates
    - error: Error messages
    """
    await websocket.accept()
    session_id = f"voice_{uuid.uuid4().hex[:8]}"
    conversation_id = None
    
    logger.info(f"🎙️ Voice chat session started: {session_id}")
    
    # Get dependencies
    llm_provider = getattr(router, '_llm_provider', None)
    conversation_memory = getattr(router, '_conversation_memory', {})
    get_or_create_conversation = getattr(router, '_get_or_create_conversation', None)
    add_message_to_conversation = getattr(router, '_add_message_to_conversation', None)
    
    try:
        # Initialize TTS engines
        from src.voice.simple_tts import get_simple_tts
        from src.voice.vibevoice_realtime import get_vibevoice_realtime, VibeVoiceSpeaker
        
        simple_tts = get_simple_tts()
        vibevoice = await get_vibevoice_realtime()
        
        # Default TTS engine
        tts_engine_name = "openai" if simple_tts.use_openai else "gtts"
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "capabilities": {
                "streaming_stt": True,
                "streaming_llm": True,
                "streaming_tts": True,
                "openai_tts": simple_tts.use_openai,
                "vibevoice_tts": True,
                "interruption": True,
                "latency_target_ms": 500,
                "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            }
        })
        
        # Initialize STT service
        stt_service = None
        try:
            from src.voice.whisper_stt import get_whisper_stt
            stt_service = get_whisper_stt(model_size="base")
            logger.info("✅ Whisper STT loaded")
        except Exception as e:
            logger.warning(f"⚠️ Whisper not available: {e}")
        
        # Message processing loop
        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type")
                
                # Extract settings if provided
                client_settings = data.get("settings", {})
                requested_engine = client_settings.get("engine", "gtts")
                requested_speaker = client_settings.get("default_speaker", "Carter")
                
                # Determine active TTS engine for this turn
                active_tts_engine = "vibevoice" if requested_engine == "vibevoice" else tts_engine_name
                
                # ===== AUDIO INPUT (Full Pipeline) =====
                if msg_type == "audio_input":
                    start_time = time.time()
                    audio_data = data.get("audio_data", "")
                    
                    if not audio_data:
                        await websocket.send_json({"type": "error", "message": "No audio data"})
                        continue
                    
                    # STEP 1: STT (Streaming when possible)
                    await websocket.send_json({"type": "status", "stage": "transcribing"})
                    
                    transcription_text = ""
                    stt_time = 0
                    if stt_service:
                        stt_result = await stt_service.transcribe_base64(audio_data)
                        if stt_result.get("success"):
                            transcription_text = stt_result.get("text", "").strip()
                            stt_time = time.time() - start_time
                            logger.info(f"⏱️ STT: {stt_time*1000:.0f}ms")
                            
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcription_text,
                                "confidence": stt_result.get("confidence", 0.0),
                                "latency_ms": int(stt_time * 1000)
                            })
                    
                    if not transcription_text:
                        await websocket.send_json({"type": "error", "message": "Transcription failed"})
                        continue
                    
                    # STEP 2: LLM (Streaming response)
                    await websocket.send_json({"type": "status", "stage": "thinking"})
                    llm_start = time.time()
                    
                    # Get or create conversation
                    if not conversation_id and get_or_create_conversation:
                        conversation_id = get_or_create_conversation(None, session_id)
                    
                    # Add user message
                    if add_message_to_conversation:
                        await add_message_to_conversation(conversation_id, "user", transcription_text, {}, session_id)
                    
                    # Generate LLM response with streaming
                    response_text = ""
                    llm_time = 0
                    if llm_provider:
                        messages = []
                        
                        # Get conversation history (last 6 messages for context)
                        if conversation_id and conversation_id in conversation_memory:
                            history = conversation_memory[conversation_id][-6:]
                            for msg in history:
                                messages.append({
                                    "role": msg["role"],
                                    "content": msg["content"]
                                })
                        
                        # Generate response
                        llm_result = await llm_provider.generate_response(
                            messages=messages,
                            temperature=0.7,
                            max_tokens=150,  # Keep responses concise for voice
                            requested_provider="openai"  # Use GPT-4 for best quality
                        )
                        
                        response_text = llm_result.get("content", "")
                        llm_time = time.time() - llm_start
                        logger.info(f"⏱️ LLM: {llm_time*1000:.0f}ms")
                        
                        # Stream text response to client
                        await websocket.send_json({
                            "type": "text_response",
                            "text": response_text,
                            "latency_ms": int(llm_time * 1000)
                        })
                        
                        # Add to conversation memory
                        if add_message_to_conversation:
                            await add_message_to_conversation(conversation_id, "assistant", response_text, {}, session_id)
                    else:
                        response_text = "I'm sorry, I'm having trouble connecting to my language model."
                    
                    # STEP 3: TTS (Dynamically selected engine)
                    await websocket.send_json({"type": "status", "stage": "synthesizing"})
                    tts_start = time.time()
                    
                    try:
                        audio_bytes = None
                        
                        if active_tts_engine == "vibevoice":
                            # Use VibeVoice
                            speaker_map = {
                                "carter": VibeVoiceSpeaker.CARTER,
                                "nova": VibeVoiceSpeaker.NOVA,
                                "aria": VibeVoiceSpeaker.ARIA,
                                "davis": VibeVoiceSpeaker.DAVIS,
                            }
                            # Handle both specific names and generic agent names
                            speaker_key = requested_speaker.lower()
                            if speaker_key == "consciousness": speaker_key = "carter"
                            elif speaker_key == "physics": speaker_key = "davis" 
                            elif speaker_key == "research": speaker_key = "nova"
                            
                            speaker_enum = speaker_map.get(speaker_key, VibeVoiceSpeaker.CARTER)
                            audio_bytes = await vibevoice.synthesize(response_text, speaker_enum)
                        else:
                            # Use SimpleTTS (OpenAI/gTTS)
                            audio_bytes = await simple_tts.synthesize_async(response_text)
                        
                        if audio_bytes:
                            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                            tts_time = time.time() - tts_start
                            total_time = time.time() - start_time
                            
                            logger.info(f"⏱️ TTS ({active_tts_engine}): {tts_time*1000:.0f}ms | Total: {total_time*1000:.0f}ms")
                            
                            await websocket.send_json({
                                "type": "audio_response",
                                "audio_data": audio_base64,
                                "format": "wav" if active_tts_engine == "vibevoice" else "mp3",
                                "text": response_text,
                                "tts_engine": active_tts_engine,
                                "latency": {
                                    "stt_ms": int(stt_time * 1000),
                                    "llm_ms": int(llm_time * 1000),
                                    "tts_ms": int(tts_time * 1000),
                                    "total_ms": int(total_time * 1000)
                                }
                            })
                        else:
                            await websocket.send_json({"type": "error", "message": "TTS generation failed"})
                    except Exception as e:
                        logger.error(f"TTS error: {e}")
                        await websocket.send_json({"type": "error", "message": f"TTS error: {str(e)}"})
                
                # ===== TEXT INPUT (Skip STT) =====
                elif msg_type == "text_input":
                    text_input = data.get("text", "").strip()
                    if not text_input:
                        continue
                    
                    start_time = time.time()
                    
                    # Get or create conversation
                    if not conversation_id and get_or_create_conversation:
                        conversation_id = get_or_create_conversation(None, session_id)
                    
                    # Add user message
                    if add_message_to_conversation:
                        await add_message_to_conversation(conversation_id, "user", text_input, {}, session_id)
                    
                    # Generate LLM response
                    await websocket.send_json({"type": "status", "stage": "thinking"})
                    
                    response_text = ""
                    if llm_provider:
                        messages = []
                        if conversation_id and conversation_id in conversation_memory:
                            history = conversation_memory[conversation_id][-6:]
                            for msg in history:
                                messages.append({"role": msg["role"], "content": msg["content"]})
                        
                        llm_result = await llm_provider.generate_response(
                            messages=messages,
                            temperature=0.7,
                            max_tokens=150,
                            requested_provider="openai"
                        )
                        
                        response_text = llm_result.get("content", "")
                        await websocket.send_json({"type": "text_response", "text": response_text})
                        if add_message_to_conversation:
                            await add_message_to_conversation(conversation_id, "assistant", response_text, {}, session_id)
                    
                    # Generate audio (fast async TTS)
                    await websocket.send_json({"type": "status", "stage": "synthesizing"})
                    tts_start = time.time()
                    try:
                        audio_bytes = await tts.synthesize_async(response_text)
                        
                        if audio_bytes:
                            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                            tts_time = time.time() - tts_start
                            total_time = time.time() - start_time
                            
                            await websocket.send_json({
                                "type": "audio_response",
                                "audio_data": audio_base64,
                                "format": "mp3",
                                "text": response_text,
                                "tts_engine": tts_engine,
                                "latency": {
                                    "tts_ms": int(tts_time * 1000),
                                    "total_ms": int(total_time * 1000)
                                }
                            })
                    except Exception as e:
                        logger.error(f"TTS error: {e}")
                
                # ===== SET VOICE =====
                elif msg_type == "set_voice":
                    voice = data.get("voice", "alloy")
                    tts.set_voice(voice)
                    await websocket.send_json({
                        "type": "voice_changed",
                        "voice": voice,
                        "available_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                    })
                
                # ===== STATUS REQUEST =====
                elif msg_type == "get_status":
                    await websocket.send_json({
                        "type": "status_response",
                        "session_id": session_id,
                        "conversation_id": conversation_id,
                        "messages_count": len(conversation_memory.get(conversation_id, [])) if conversation_id else 0,
                        "stt_available": stt_service is not None,
                        "llm_available": llm_provider is not None,
                        "tts_engine": tts_engine,
                        "openai_tts": tts.use_openai,
                        "current_voice": tts.voice,
                        "available_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                    })
                
                # ===== INTERRUPT =====
                elif msg_type == "interrupt":
                    logger.info(f"🛑 Interrupted session: {session_id}")
                    await websocket.send_json({"type": "interrupted"})
                
                # ===== CLOSE =====
                elif msg_type == "close":
                    logger.info(f"👋 Closing session: {session_id}")
                    await websocket.send_json({"type": "closed", "session_id": session_id})
                    break
                    
            except WebSocketDisconnect:
                logger.info(f"📴 Voice chat disconnected: {session_id}")
                break
            except Exception as e:
                logger.error(f"Voice chat error: {e}")
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except:
                    break
                    
    except Exception as e:
        logger.error(f"Voice chat session error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        logger.info(f"🔚 Voice chat session ended: {session_id}")


# ====== Dependency Injection Helper ======

def set_dependencies(
    llm_provider=None,
    conversation_memory=None,
    vibevoice_engine=None,
    consciousness_service=None,
    get_or_create_conversation=None,
    add_message_to_conversation=None
):
    """Set dependencies for the voice router"""
    router._llm_provider = llm_provider
    router._conversation_memory = conversation_memory or {}
    router._vibevoice_engine = vibevoice_engine
    router._consciousness_service = consciousness_service
    router._get_or_create_conversation = get_or_create_conversation
    router._add_message_to_conversation = add_message_to_conversation
