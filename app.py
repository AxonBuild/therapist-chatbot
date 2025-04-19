import os
import streamlit as st
import json
import time
import base64
import tempfile
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Import your chatbot backend
from backend import MedicalChatbot

# Load environment variables
load_dotenv()

# Create OpenAI client for TTS
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Function to convert text to speech using OpenAI
def text_to_speech(text, voice="alloy"):
    """Convert text to speech using OpenAI's TTS API."""
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            # Call OpenAI TTS API with the correct streaming method
            with openai_client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text
            ) as audio_response:
                audio_response.stream_to_file(temp_audio_file.name)
            print(f"Audio saved to {temp_audio_file.name}")
            
            # Return the path to the temporary file
            return temp_audio_file.name
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

# Function to create an HTML audio player
def get_audio_player(audio_file_path, autoplay=True):
    """Create an HTML audio player for the given audio file."""
    if audio_file_path:
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        # Encode the audio bytes as base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Create an HTML audio element with autoplay
        autoplay_attr = "autoplay" if autoplay else ""
        audio_html = f"""
        <audio {autoplay_attr} controls style="width: 100%; display: none;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        <script>
            // This ensures the audio plays even if the controls are hidden
            document.addEventListener('DOMContentLoaded', (event) => {{
                const audioElements = document.getElementsByTagName('audio');
                const latestAudio = audioElements[audioElements.length-1];
                if({str(autoplay).lower()}) {{
                    latestAudio.play();
                }}
            }});
        </script>
        """
        return audio_html
    return ""

# Page configuration
st.set_page_config(
    page_title="EkoMindAI Therapeutic Assistant",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #5E3B50;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subheader {
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #777;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #ddd;
    }
    .stChatMessage .message-container .avatar.user {
        background-color: #8E617F !important;
    }
    .stChatMessage .message-container .avatar.assistant {
        background-color: #5E3B50 !important;
    }
    button[data-testid="chatInputSubmitButton"] {
        background-color: #5E3B50 !important;
        color: white !important;
    }
    
    div[data-testid="stChatInput"] {
        border-color: #5E3B50;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #8E617F;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .logo {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background-color: #5E3B50;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Style for clear chat button */
    .clear-button {
        background-color: #f0f0f0;
        color: #333;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.3s;
        float: right;
        margin-bottom: 10px;
    }
    
    .clear-button:hover {
        background-color: #8E617F;
        color: white;
        border-color: #8E617F;
    }
    
    .therapy-script {
        background-color: #f8f4f8;
        border-left: 5px solid #5E3B50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .status-indicator {
        margin-bottom: 20px;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    
    .phase-1 {
        background-color: #e1f5fe;
        color: #0277bd;
    }
    
    .phase-2 {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    
    .debug-info {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = MedicalChatbot()

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "speech_enabled" not in st.session_state:
    st.session_state.speech_enabled = True

if "current_phase" not in st.session_state:
    st.session_state.current_phase = "Identifying your concerns"

if "therapeutic_script" not in st.session_state:
    st.session_state.therapeutic_script = None

# Logo and App title
st.markdown('<div class="logo-container"><div class="logo">EM</div></div>', unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>EkoMindAI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your mental wellness companion</p>", unsafe_allow_html=True)

# Function to clear chat history
def clear_chat_history():
    """Clear the chat history and reset the chatbot session"""
    st.session_state.messages = []
    st.session_state.chatbot.reset_session()
    st.session_state.current_phase = "Identifying your concerns"
    st.session_state.therapeutic_script = None
    st.rerun()

# Settings in sidebar
with st.sidebar:
    st.title("Settings")
    
    # Debug mode toggle
    debug_toggle = st.toggle("Debug Mode", value=st.session_state.debug_mode)
    if debug_toggle != st.session_state.debug_mode:
        st.session_state.debug_mode = debug_toggle
        st.rerun()
    
    # Text-to-speech toggle
    speech_toggle = st.toggle("Text-to-Speech", value=st.session_state.speech_enabled)
    if speech_toggle != st.session_state.speech_enabled:
        st.session_state.speech_enabled = speech_toggle
        st.rerun()
    
    # Voice selection for OpenAI TTS
    if st.session_state.speech_enabled:
        voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        if "selected_voice" not in st.session_state:
            st.session_state.selected_voice = "nova"  # Default to nova as it's good for therapeutic content
            
        selected_voice = st.selectbox(
            "Voice", 
            options=voice_options,
            index=voice_options.index(st.session_state.selected_voice)
        )
        
        if selected_voice != st.session_state.selected_voice:
            st.session_state.selected_voice = selected_voice
            st.rerun()
    
    # Reset button
    if st.button("Reset Conversation", use_container_width=True):
        clear_chat_history()

# Display current phase
phase_color = "phase-1" if st.session_state.current_phase == "Identifying your concerns" else "phase-2"
st.markdown(f'<div class="status-indicator {phase_color}">{st.session_state.current_phase}</div>', unsafe_allow_html=True)

# Button to clear chat
if st.session_state.messages:
    cols = st.columns([4, 1])
    with cols[1]:
        if st.button("Clear Chat", type="secondary", use_container_width=True):
            clear_chat_history()

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Check if this is a therapeutic script message
        if msg.get("is_script", False):
            st.markdown(f"<div class='therapy-script'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])
    
    # Display debug information if enabled
    if st.session_state.debug_mode and msg.get("debug_info"):
        with st.expander("Debug Info"):
            st.json(msg["debug_info"])

# Chat input
if user_input := st.chat_input("Share what's on your mind..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
   
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
   
    # Process the message through the chatbot backend
    response = st.session_state.chatbot.process_message(user_input)
    
    # Update phase based on chatbot state
    if st.session_state.chatbot.session.identified_disorder and st.session_state.current_phase == "Identifying your concerns":
        st.session_state.current_phase = f"Exploring your {st.session_state.chatbot.session.identified_disorder} concerns"
    
    # Determine if the response is a script
    is_script = response.startswith("SCRIPT:::")
    if is_script:
        # Strip the marker
        display_content = response[len("SCRIPT:::"):]
        # Use a calmer voice for scripts
        script_voice = "nova" if st.session_state.selected_voice != "nova" else "shimmer"
        tts_voice = script_voice
    else:
        display_content = response
        tts_voice = st.session_state.selected_voice

    # Generate speech if enabled
    audio_path = None
    if st.session_state.speech_enabled and display_content: # Check if content exists
        audio_path = text_to_speech(display_content, voice=tts_voice)
    
    # Display the response (either script or normal message)
    with st.chat_message("assistant"):
        if is_script:
            st.markdown(f"<div class='therapy-script'>{display_content}</div>", unsafe_allow_html=True)
        else:
            # Use a different class maybe for non-script audio messages if needed
            st.markdown(f"<div class='assistant-audio-message'>{display_content}</div>", unsafe_allow_html=True)
            
        # Play audio if available
        if audio_path:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

    # Add the message to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": display_content, # Use the processed content
        "is_script": is_script, # Mark if it was a script
        "debug_info": { # Keep debug info consistent
            "phase": st.session_state.current_phase,
            "disorder": st.session_state.chatbot.session.identified_disorder,
            "confidence": st.session_state.chatbot.session.disorder_confidence if st.session_state.chatbot.session.identified_disorder else 0.0,
            "current_node": st.session_state.chatbot.session.current_node_id
        } if st.session_state.debug_mode else None
    })

# Disclaimer
st.markdown("""
<div class="disclaimer">
    EkoMindAI provides support based on therapeutic methods, but is not a replacement for professional mental health care. 
    If you're experiencing a crisis or need immediate help, please contact a mental health professional.
</div>
""", unsafe_allow_html=True)