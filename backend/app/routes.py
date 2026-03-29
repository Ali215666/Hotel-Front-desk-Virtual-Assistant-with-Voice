"""
API routes for Hotel Front Desk conversational AI system.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import asyncio
from contextlib import suppress
import json
import logging
import re

from .websocket_manager import WebSocketManager
from .dependencies import (
    get_websocket_manager,
    get_session_manager,
    get_ollama_client,
    get_memory_manager,
    get_prompt_builder,
    get_audio_converter,
    get_moonshine_asr,
    get_piper_tts,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_greeting_from_response(response: str, has_history: bool) -> str:
    """
    Remove greeting patterns from assistant responses if conversation history exists.
    
    Args:
        response: The assistant's response text
        has_history: Whether conversation history exists
        
    Returns:
        Cleaned response text
    """
    if not has_history or not response:
        return response
    
    # Patterns to remove (case insensitive)
    greeting_patterns = [
        r'^Hello\s+\w+,?\s*',  # "Hello Name," or "Hello Name "
        r'^Hi\s+\w+,?\s*',      # "Hi Name," or "Hi Name "
        r'^Hey\s+\w+,?\s*',     # "Hey Name," or "Hey Name "
        r'^Hello,?\s*',         # "Hello," or "Hello "
        r'^Hi,?\s*',            # "Hi," or "Hi "
        r'^Hey,?\s*',           # "Hey," or "Hey "
    ]
    
    cleaned = response
    for pattern in greeting_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove leading whitespace after cleaning
    cleaned = cleaned.lstrip()
    
    return cleaned


router = APIRouter()


# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    reply: str = Field(..., description="Assistant's response")


@router.post("/sessions")
async def create_session() -> Dict[str, Any]:
    """
    Create a new conversation session.
    
    Returns:
        Dict containing session_id and metadata
    """
    # TODO: Implement session creation logic
    pass


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    """
    Retrieve session information.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dict containing session data
    """
    # TODO: Implement session retrieval logic
    pass


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """
    Delete a session and its history.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dict with deletion confirmation
    """
    # TODO: Implement session deletion logic
    pass


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str) -> Dict[str, Any]:
    """
    Retrieve conversation history for a session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dict containing conversation history
    """
    # TODO: Implement history retrieval logic
    pass


@router.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest,
    session_manager=Depends(get_session_manager),
    memory_manager=Depends(get_memory_manager),
    prompt_builder=Depends(get_prompt_builder),
    ollama_client=Depends(get_ollama_client)
) -> ChatResponse:
    """
    REST endpoint for synchronous chat interaction.
    
    Accepts POST requests with JSON payload:
    {
        "session_id": "string",
        "message": "string"
    }
    
    Returns JSON response:
    {
        "reply": "string"
    }
    
    Args:
        request: ChatRequest containing session_id and message
        session_manager: Session manager dependency
        memory_manager: Memory manager dependency
        prompt_builder: Prompt builder dependency
        ollama_client: Ollama client dependency
        
    Returns:
        ChatResponse containing the assistant's reply
        
    Raises:
        HTTPException: 400 for invalid requests, 500 for server errors
    """
    try:
        session_id = request.session_id
        user_message = request.message
        
        # Validate inputs
        if not session_id or not session_id.strip():
            raise HTTPException(
                status_code=400,
                detail="Invalid session_id: must be a non-empty string"
            )
        
        if not user_message or not user_message.strip():
            raise HTTPException(
                status_code=400,
                detail="Invalid message: must be a non-empty string"
            )
        
        logger.info(f"REST API: Processing message for session {session_id}: {user_message[:50]}...")
        
        # Ensure session exists in session manager
        if not session_manager.get_session(session_id):
            session_manager.create_session()
            from datetime import datetime
            session_manager.sessions[session_id] = {
                'created_at': datetime.now(),
                'last_active': datetime.now()
            }
            logger.info(f"Created new session: {session_id}")
        
        # Ensure memory session exists
        if not memory_manager.session_exists(session_id):
            memory_manager.create_session(session_id)
            logger.info(f"Created new memory session: {session_id}")
        
        # Get conversation history
        history = memory_manager.get_history(session_id)
        active_context = memory_manager.get_active_context(history, session_id=session_id)
        
        # Build prompt with context
        prompt = prompt_builder.build_prompt(active_context, user_message)
        
        # Generate response from LLM (non-streaming)
        logger.info(f"Generating response for session {session_id}")
        response = ollama_client.generate(prompt)
        
        # Check if response is an error message
        if response.startswith("Error:"):
            logger.error(f"LLM error for session {session_id}: {response}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {response}"
            )
        
        # Clean greeting from response if conversation history exists
        cleaned_response = clean_greeting_from_response(response, len(active_context) > 0)
        
        # Store conversation in memory
        memory_manager.add_message(session_id, "user", user_message)
        memory_manager.add_message(session_id, "assistant", cleaned_response)
        
        logger.info(f"REST API: Response generated for session {session_id}")
        
        return ChatResponse(reply=cleaned_response)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except ValueError as ve:
        logger.error(f"Validation error in chat endpoint: {ve}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(ve)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    WebSocket endpoint for real-time hotel assistant conversation.
    
    Accepts JSON messages with format:
    {
        "session_id": "string",
        "message": "string"
    }
    
    Args:
        websocket: WebSocket connection
        ws_manager: WebSocket connection manager
    """
    # Import dependencies
    from .dependencies import get_session_manager, get_ollama_client, get_memory_manager, get_prompt_builder
    
    session_manager = get_session_manager()
    ollama_client = get_ollama_client()
    memory_manager = get_memory_manager()
    prompt_builder = get_prompt_builder()
    
    # Initially accept the connection without session_id
    await websocket.accept()
    logger.info("WebSocket connection accepted, awaiting session_id")
    
    current_session_id = None
    
    try:
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Validate message format
                if not isinstance(message_data, dict):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid message format. Expected JSON object."
                    })
                    continue
                
                session_id = message_data.get("session_id")
                user_message = message_data.get("message")
                
                # Validate required fields
                if not session_id or not user_message:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Missing required fields: 'session_id' and 'message'"
                    })
                    continue
                
                # Register connection with session_id if first message or session changed
                if current_session_id != session_id:
                    # Update session tracking without closing the WebSocket
                    # (same connection, different session ID)
                    async with ws_manager._lock:
                        # Remove old session ID from tracking (if exists)
                        if current_session_id in ws_manager.active_connections:
                            # Only remove from dict, don't close the connection
                            ws_manager.active_connections.pop(current_session_id)
                            logger.info(f"Removed old session tracking: {current_session_id}")
                        
                        # If new session ID already has a connection, close that old one
                        if session_id in ws_manager.active_connections:
                            old_ws = ws_manager.active_connections[session_id]
                            if old_ws != websocket:  # Only close if it's a different connection
                                try:
                                    await old_ws.close()
                                except:
                                    pass
                        
                        # Register current WebSocket with new session ID
                        ws_manager.active_connections[session_id] = websocket
                    
                    # Update current session ID
                    current_session_id = session_id
                    logger.info(f"WebSocket session updated to: {session_id}")
                
                # Ensure session exists in session manager
                if not session_manager.get_session(session_id):
                    session_manager.create_session()
                    session_manager.sessions[session_id] = {
                        'created_at': session_manager.sessions.get(session_id, {}).get('created_at'),
                        'last_active': session_manager.sessions.get(session_id, {}).get('last_active')
                    }
                
                # Ensure memory session exists
                if not memory_manager.session_exists(session_id):
                    memory_manager.create_session(session_id)
                
                # Handle init/handshake messages - just acknowledge, don't process
                if user_message == "__INIT__" or message_data.get("type") == "init":
                    logger.info(f"Received init handshake for session {session_id}")
                    await websocket.send_json({
                        "type": "status",
                        "message": "Session registered"
                    })
                    continue
                
                logger.info(f"Processing message for session {session_id}: {user_message[:50]}...")
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "status",
                    "message": "Processing your request..."
                })
                
                # Get conversation history
                history = memory_manager.get_history(session_id)
                active_context = memory_manager.get_active_context(history, session_id=session_id)
                
                # Build prompt
                prompt = prompt_builder.build_prompt(active_context, user_message)
                
                # Store user message
                memory_manager.add_message(session_id, "user", user_message)
                
                # Stream response from Ollama
                full_response = ""
                
                try:
                    async for token in ollama_client.generate_stream(prompt):
                        if token:
                            full_response += token
                            # Send each token to client
                            await websocket.send_text(token)
                    
                    # Send completion signal
                    await websocket.send_json({
                        "type": "done",
                        "message": "Response complete"
                    })
                    
                    # Clean greeting from response if conversation history exists
                    cleaned_response = clean_greeting_from_response(full_response, len(active_context) > 0)
                    
                    # Store cleaned assistant response
                    memory_manager.add_message(session_id, "assistant", cleaned_response)
                    logger.info(f"Response completed for session {session_id}")
                
                except Exception as stream_error:
                    logger.error(f"Error during streaming: {stream_error}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error generating response: {str(stream_error)}"
                    })
            
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error: {json_error}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
                continue
            
            except Exception as msg_error:
                logger.error(f"Error processing message: {msg_error}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(msg_error)}"
                })
                continue
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {current_session_id}")
        if current_session_id:
            await ws_manager.disconnect(current_session_id)
    
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket endpoint: {e}")
        if current_session_id:
            await ws_manager.disconnect(current_session_id)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Unexpected error: {str(e)}"
            })
        except:
            pass


async def _stream_tts_for_text(
    websocket: WebSocket,
    piper_tts,
    fragment: str,
    sequence_start: int
) -> int:
    """
    Stream synthesized audio chunks for a single text fragment.

    Returns:
        int: Next sequence id after the streamed chunks.
    """
    seq = sequence_start
    async for audio_chunk_b64 in piper_tts.synthesize_chunked_b64(fragment):
        await websocket.send_json({
            "type": "audio_chunk",
            "audio": audio_chunk_b64,
            "format": "wav",
            "sequence": seq,
        })
        seq += 1
    return seq


@router.websocket("/ws/voice_chat")
async def websocket_voice_chat_endpoint(
    websocket: WebSocket,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    WebSocket endpoint for voice-first conversational interaction.

    Pipeline:
    Audio input -> Mooshine ASR -> Conversation manager/prompt -> LLM stream
    -> sentence-buffered Piper TTS -> audio chunks back to browser.

    Message contract:
    - JSON init: {"type":"init","session_id":"..."}
    - Binary frames: microphone/uploaded audio chunks
    - JSON control: {"type":"audio_end","session_id":"...","mime_type":"audio/webm"}

    Outbound events:
    - transcript, token, done, audio_chunk, audio_done, status, error.
    """
    # Core dependencies are the same as /ws/chat to preserve existing
    # conversation state, history memory, and prompt orchestration behavior.
    session_manager = get_session_manager()
    ollama_client = get_ollama_client()
    memory_manager = get_memory_manager()
    prompt_builder = get_prompt_builder()
    audio_converter = get_audio_converter()
    moonshine_asr = get_moonshine_asr()
    piper_tts = get_piper_tts()

    await websocket.accept()
    current_session_id = None
    current_connection_key = None
    buffered_audio = bytearray()
    current_mime_type = "audio/webm"

    logger.info("Voice WebSocket accepted, waiting for init")

    async def register_session_if_needed(new_session_id: str):
        nonlocal current_session_id, current_connection_key
        new_connection_key = f"voice:{new_session_id}"

        if current_session_id == new_session_id:
            return

        async with ws_manager._lock:
            # Voice and text sockets share the same manager instance, so we
            # namespace voice keys to avoid closing `/ws/chat` connections.
            if current_connection_key in ws_manager.active_connections:
                ws_manager.active_connections.pop(current_connection_key)

            if new_connection_key in ws_manager.active_connections:
                old_ws = ws_manager.active_connections[new_connection_key]
                if old_ws != websocket:
                    try:
                        await old_ws.close()
                    except Exception:
                        pass

            ws_manager.active_connections[new_connection_key] = websocket

        current_session_id = new_session_id
        current_connection_key = new_connection_key
        logger.info("Voice session updated to: %s", current_session_id)

    async def ensure_state_for_session(session_id: str):
        # Keep session manager behavior aligned with existing websocket text flow.
        if not session_manager.get_session(session_id):
            session_manager.create_session()
            session_manager.sessions[session_id] = {
                'created_at': session_manager.sessions.get(session_id, {}).get('created_at'),
                'last_active': session_manager.sessions.get(session_id, {}).get('last_active')
            }

        if not memory_manager.session_exists(session_id):
            memory_manager.create_session(session_id)

    async def process_audio_turn(session_id: str, audio_bytes: bytes, mime_type: str):
        if not audio_bytes:
            await websocket.send_json({
                "type": "error",
                "message": "No audio data received"
            })
            return

        await websocket.send_json({
            "type": "status",
            "message": "Transcribing audio..."
        })

        source_ext = (mime_type.split("/")[-1] if mime_type and "/" in mime_type else "webm").split(";")[0]
        wav_audio = await audio_converter.to_wav_16k(audio_bytes, source_ext)
        transcript = await moonshine_asr.transcribe(wav_audio)

        if not transcript:
            await websocket.send_json({
                "type": "error",
                "message": "ASR produced empty transcription"
            })
            return

        await websocket.send_json({
            "type": "transcript",
            "text": transcript,
            "final": True
        })

        history = memory_manager.get_history(session_id)
        active_context = memory_manager.get_active_context(history, session_id=session_id)
        prompt = prompt_builder.build_prompt(active_context, transcript)
        memory_manager.add_message(session_id, "user", transcript)

        # Low-latency voice output with deterministic chunking:
        # - stream text tokens immediately to UI
        # - first audio chunk is emitted at first sentence-ending punctuation
        # - second chunk (optional) contains the remaining response
        # - total chunks per reply: max 2
        full_response = ""
        first_audio_buffer = ""
        trailing_audio_buffer = ""
        audio_seq = 0
        first_audio_task = None
        first_audio_started = False

        await websocket.send_json({
            "type": "status",
            "message": "Generating response..."
        })

        async for token in ollama_client.generate_stream(prompt):
            if not token:
                continue
            full_response += token
            await websocket.send_json({
                "type": "token",
                "content": token
            })


            first_audio_buffer += token
            # Continuously check for sentence-ending punctuation and emit TTS chunk for each sentence
            while True:
                split_idx = -1
                for punct in (".", "?", "!", "\n"):
                    idx = first_audio_buffer.find(punct)
                    if idx != -1 and (split_idx == -1 or idx > split_idx):
                        split_idx = idx
                if split_idx == -1:
                    break
                fragment = first_audio_buffer[: split_idx + 1].strip()
                if fragment:
                    audio_seq = await _stream_tts_for_text(websocket, piper_tts, fragment, audio_seq)
                first_audio_buffer = first_audio_buffer[split_idx + 1:]


        # After all tokens, flush any remaining text as a final chunk
        remaining = first_audio_buffer.strip()
        if remaining:
            audio_seq = await _stream_tts_for_text(websocket, piper_tts, remaining, audio_seq)

        cleaned_response = clean_greeting_from_response(full_response, len(active_context) > 0)

        memory_manager.add_message(session_id, "assistant", cleaned_response)

        await websocket.send_json({"type": "done", "message": "Response complete"})
        await websocket.send_json({"type": "audio_done", "chunks": audio_seq})

    try:
        while True:
            packet = await websocket.receive()

            if "bytes" in packet and packet["bytes"] is not None:
                # Binary frames are audio chunks from live mic or upload stream.
                buffered_audio.extend(packet["bytes"])
                continue

            if "text" not in packet or packet["text"] is None:
                continue

            try:
                message_data = json.loads(packet["text"])
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                continue

            if not isinstance(message_data, dict):
                await websocket.send_json({"type": "error", "message": "Invalid JSON payload"})
                continue

            msg_type = message_data.get("type", "")
            session_id = message_data.get("session_id")

            if not session_id:
                await websocket.send_json({"type": "error", "message": "Missing session_id"})
                continue

            await register_session_if_needed(session_id)
            await ensure_state_for_session(session_id)

            if msg_type == "init":
                await websocket.send_json({"type": "status", "message": "Voice session registered"})
                continue

            if msg_type == "audio_chunk_meta":
                # Metadata can be sent ahead of binary chunks to identify mime/container.
                current_mime_type = message_data.get("mime_type", current_mime_type)
                continue

            if msg_type == "audio_end":
                current_mime_type = message_data.get("mime_type", current_mime_type)
                audio_payload = bytes(buffered_audio)
                buffered_audio.clear()

                try:
                    await process_audio_turn(session_id, audio_payload, current_mime_type)
                except Exception as turn_error:
                    logger.error("Voice turn failed for %s: %s", session_id, turn_error, exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Voice pipeline error: {turn_error}"
                    })
                continue

            await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected for session: %s", current_session_id)
        if current_connection_key:
            await ws_manager.disconnect(current_connection_key)
    except Exception as exc:
        logger.error("Unexpected voice websocket error: %s", exc, exc_info=True)
        if current_connection_key:
            await ws_manager.disconnect(current_connection_key)
        try:
            await websocket.send_json({"type": "error", "message": f"Unexpected error: {exc}"})
        except Exception:
            pass

