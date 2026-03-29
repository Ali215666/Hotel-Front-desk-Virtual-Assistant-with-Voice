import React, { useState, useRef, useEffect } from 'react'
import './InputBox.css'

function InputBox({
  onSendMessage,
  isConnected,
  isTyping,
  sessionId,
  isRecording,
  isVoiceProcessing,
  isVoiceEnabled,
  onStartRecording,
  onStopRecording
}) {
  const [message, setMessage] = useState('')
  const textareaRef = useRef(null)

  // Clear input when session changes (New Chat clicked)
  useEffect(() => {
    setMessage('')
    if (textareaRef.current) {
      textareaRef.current.style.height = '52px'
    }
  }, [sessionId])

  const handleSubmit = (e) => {
    e.preventDefault()
    
    // Validate message and connection status
    if (!message.trim()) {
      console.log('Cannot send empty message')
      return
    }
    
    if (!isConnected) {
      console.log('Cannot send message: not connected')
      return
    }
    
    if (isTyping) {
      console.log('Cannot send message: assistant is typing')
      return
    }
    
    // Send message to backend
    console.log('Sending message from input box:', {
      session_id: sessionId,
      message: message.trim()
    })
    
    onSendMessage(message.trim())
    
    // Clear input after sending
    setMessage('')
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = '52px'
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // Auto-resize textarea as user types
  const handleChange = (e) => {
    setMessage(e.target.value)
    
    if (textareaRef.current) {
      textareaRef.current.style.height = '52px'
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px'
    }
  }

  return (
    <form className="input-box" onSubmit={handleSubmit}>
      <div className="input-container">
        <textarea
          ref={textareaRef}
          className="message-input"
          placeholder={isConnected ? "Type your message here..." : "Connecting..."}
          value={message}
          onChange={handleChange}
          onKeyPress={handleKeyPress}
          disabled={!isConnected || isTyping}
          rows={1}
        />

        <div className="input-actions">
          <button
            type="button"
            className={`icon-button mic-button ${isRecording ? 'recording' : ''}`}
            onClick={isRecording ? onStopRecording : onStartRecording}
            disabled={!isVoiceEnabled || isVoiceProcessing || isTyping}
            title={isRecording ? 'Stop recording' : 'Start recording'}
            aria-label={isRecording ? 'Stop recording' : 'Start recording'}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M12 15a3 3 0 0 0 3-3V7a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M19 11a7 7 0 0 1-14 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 18v3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M8 21h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          <button
            type="submit"
            className="icon-button send-button"
            disabled={!message.trim() || !isConnected || isTyping}
            title="Send message"
            aria-label="Send message"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path>
            </svg>
          </button>
        </div>
      </div>
    </form>
  )
}

export default InputBox
