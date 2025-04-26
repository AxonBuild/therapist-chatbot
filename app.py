import os
import streamlit as st
import json
import time
import base64
import tempfile
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from openai import OpenAI
import glob

# Import the therapeutic chatbot from backend
from backend_new import TherapeuticChatbot, OPENROUTER_API_KEY

# Load environment variables
load_dotenv()

# Create OpenAI client for TTS
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_script_audio_path(script_id: str, disorder_key: str, base_scripts_dir: str) -> str | None:
    """
    Returns the path to the pre-generated audio file for a script, if it exists.
    
    Args:
        script_id: Format like "SCRIPT_A" (base identifier without extension)
        disorder_key: Folder name like "phobic" or "sleep"
        base_scripts_dir: Base directory for all script files
    
    Returns:
        Full path to the audio file if found, None otherwise
    """
    # Normalize script_id by removing any file extension
    if "." in script_id:
        script_id = script_id.split(".")[0]  # Remove extension if present
    
    # Ensure script_id starts with "SCRIPT_"
    if not script_id.startswith("SCRIPT_"):
        script_id = f"SCRIPT_{script_id}"
    
    # Exact path structure: scripts/[disorder_key]/scripts/
    script_dir = os.path.join("scripts", disorder_key, "scripts")
    print(f"DEBUG: Looking for audio files in {script_dir}")
    
    # Check if the directory exists
    if not os.path.exists(script_dir):
        print(f"WARNING: Directory does not exist: {script_dir}")
        return None
    
    # List all audio files in the directory
    try:
        audio_files = [f for f in os.listdir(script_dir) if f.endswith('.mp3')]
        
        # Print all available audio files for debugging
        if audio_files:
            print(f"DEBUG: Available audio files in {script_dir}:")
            for file in audio_files:
                print(f"  - {file}")
        else:
            print(f"DEBUG: No audio files found in {script_dir}")
            return None
        
        # Look for any file that starts with the script_id
        for file in audio_files:
            if file.startswith(f"{script_id}"):
                path = os.path.join(script_dir, file)
                print(f"DEBUG: Found match starting with {script_id}: {path}")
                return path
        
        print(f"WARNING: No audio file starting with {script_id} found in {script_dir}")
        return None
        
    except Exception as e:
        print(f"ERROR: Failed to list audio files in {script_dir}: {e}")
        return None

def debug_audio_files(base_dir="."):
    """
    List all audio files in the scripts directories to help with debugging.
    Can be called from debug mode to see what files are available.
    """
    audio_files = {}
    
    # Check for all disorder directories
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "scripts":
                # This is a scripts directory
                scripts_path = os.path.join(root, dir_name)
                parent_dir = os.path.basename(os.path.dirname(scripts_path))
                
                # Find all MP3 files in this directory
                mp3_files = glob.glob(os.path.join(scripts_path, "*.mp3"))
                if mp3_files:
                    audio_files[parent_dir] = mp3_files
    
    return audio_files

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
    
    .assistant-audio-message {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Check for OpenRouter API Key (essential for backend)
if not OPENROUTER_API_KEY:
    st.error("FATAL: OpenRouter API key not found. The chatbot backend cannot function. Please set the OPENROUTER_API_KEY environment variable.")
    st.stop() # Stop execution if the key is missing

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = TherapeuticChatbot()
        # Add an initial greeting if messages are empty
        if not st.session_state.messages:
             st.session_state.messages.append({
                 "role": "assistant",
                 "content": "Hello! I'm here to listen. How are you feeling today?",
                 "is_script": False # Mark explicitly
             })
    except Exception as e:
        st.error(f"Failed to initialize the chatbot backend: {e}")
        st.stop()

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "speech_enabled" not in st.session_state:
    st.session_state.speech_enabled = True

if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = "nova"  # Default voice

# Logo and App title
st.markdown('<div class="logo-container"><div class="logo">EM</div></div>', unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>EkoMindAI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your mental wellness companion</p>", unsafe_allow_html=True)

# Function to clear chat history
def clear_chat_history():
    """Clear the chat history and reset the chatbot session"""
    st.session_state.messages = []
    # Reset backend session if possible (assuming a method exists or implicitly handled)
    # If the backend manages sessions by ID, starting fresh might be enough.
    # Let's try deleting the specific session if the backend supports it
    session_id = "streamlit_user_session" # Assuming this is the ID used
    if "chatbot" in st.session_state and hasattr(st.session_state.chatbot, 'sessions') and session_id in st.session_state.chatbot.sessions:
        try:
            del st.session_state.chatbot.sessions[session_id]
            print(f"INFO: Cleared backend session '{session_id}'")
        except Exception as e:
            print(f"WARNING: Could not clear backend session '{session_id}': {e}")

    # Add the initial greeting back
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm here to listen. How are you feeling today?",
        "is_script": False
    })
    st.rerun()

# Settings in sidebar
if os.getenv("ENV") != "production":
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
            
            selected_voice = st.selectbox(
                "Voice", 
                options=voice_options,
                index=voice_options.index(st.session_state.selected_voice)
            )
            
            if selected_voice != st.session_state.selected_voice:
                st.session_state.selected_voice = selected_voice
                st.rerun()
        
        # Debug audio files button (only shown in debug mode)
        if st.session_state.debug_mode:
            if st.button("Debug Audio Files"):
                audio_files = debug_audio_files()
                st.json(audio_files)
        
        # Reset button
        if st.button("Reset Conversation", use_container_width=True):
            clear_chat_history()

# Display chat messages using standard Streamlit chat message approach
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Check if this is a therapeutic script message
        if msg.get("is_script", False):
            st.markdown(f"<div class='therapy-script'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-audio-message'>{msg['content']}</div>", unsafe_allow_html=True)
        
        # Play audio if available
        if msg["role"] == "assistant" and st.session_state.speech_enabled and msg.get("audio_path"):
            with open(msg["audio_path"], "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3", autoplay=msg.get("autoplay", False))
    
    # Display debug information if enabled
    if st.session_state.debug_mode and msg.get("debug_info"):
        with st.expander("Debug Info"):
            st.json(msg["debug_info"])

# Chat input
if user_input := st.chat_input("Share what's on your mind..."):
    with st.chat_message("user"):
        st.write(user_input)
    
    # Add user message to frontend chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
   
    # Process the message through the chatbot backend
    session_id = "streamlit_user_session"
    try:
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.process_message(user_input, session_id=session_id)

            # --- Handle Backend Response ---
            is_script = False
            script_id = None
            disorder_key = None
            script_title = None
            display_content = ""
            
            # Check if response is a dictionary (contains script metadata)
            if isinstance(response, dict):
                print("DEBUG: Received dictionary response from backend:", response)
                display_content = response.get("content", "")
                is_script = response.get("is_script", False)
                script_id = response.get("script_id")
                disorder_key = response.get("disorder_key")
                script_title = response.get("script_title")
                
                # Still check for script markers to extract the actual content
                script_start_marker = "SCRIPT_START"
                script_end_marker = "SCRIPT_END"
                
                if script_start_marker in display_content and script_end_marker in display_content:
                    lead_in_end = display_content.find(script_start_marker)
                    lead_in = display_content[:lead_in_end].strip()
                    script_content_start = lead_in_end + len(script_start_marker)
                    script_content_end = display_content.find(script_end_marker)
                    script_content = display_content[script_content_start:script_content_end].strip()
                    
                    # Format for display while keeping metadata
                    display_content = f"{lead_in}\n\n{script_content}"
                    display_content_for_tts = f"{lead_in} {script_content}"  # Flattened for TTS
                else:
                    # Fallback if somehow the markers are missing
                    display_content_for_tts = display_content
            else:
                # Response is a string
                display_content = response
                display_content_for_tts = response
                
                # Check for embedded script markers in string response (fallback case)
                script_start_marker = "SCRIPT_START"
                script_end_marker = "SCRIPT_END"
                
                if script_start_marker in display_content and script_end_marker in display_content:
                    is_script = True
                    lead_in_end = display_content.find(script_start_marker)
                    lead_in = display_content[:lead_in_end].strip()
                    script_content_start = lead_in_end + len(script_start_marker)
                    script_content_end = display_content.find(script_end_marker)
                    script_content = display_content[script_content_start:script_content_end].strip()
                    
                    display_content = f"{lead_in}\n\n{script_content}"  # Content for display
                    display_content_for_tts = f"{lead_in} {script_content}"  # Flattened for TTS
                    
                    # Try to extract script_id from content as fallback
                    script_pattern = r'SCRIPT_[A-Z]_[A-Za-z_]+'
                    script_matches = re.findall(script_pattern, display_content)
                    if script_matches:
                        script_id = script_matches[0]
                        
                        # Try to determine disorder key as fallback
                        if "sleep" in display_content.lower() or "night" in display_content.lower():
                            disorder_key = "sleep"
                        elif "phobic" in display_content.lower() or "anxiety" in display_content.lower():
                            disorder_key = "phobic"
                        else:
                            disorder_key = "general"  # default fallback

            # Set voice based on whether this is a script
            if is_script:
                tts_voice = "nova" if st.session_state.selected_voice != "nova" else "shimmer"
            else:
                tts_voice = st.session_state.selected_voice

        with st.spinner("Generating Audio..."):
            # Generate speech if enabled
            audio_path = None
            if is_script and st.session_state.speech_enabled:
                if script_id and disorder_key:
                    print(f"DEBUG: Looking for audio for script_id={script_id}, disorder_key={disorder_key}")
                    # Try to get pre-generated audio
                    audio_path = get_script_audio_path(script_id, disorder_key, ".")
                    
                    if audio_path:
                        print(f"DEBUG: Found pre-generated audio at {audio_path}")
                    else:
                        print(f"DEBUG: No pre-generated audio found for {script_id} in {disorder_key}")
                    
                # If still no audio path, fallback to TTS
                if not audio_path and display_content_for_tts:
                    print(f"DEBUG: Falling back to TTS for script")
                    try:
                        # Validate display_content_for_tts is a string
                        if not isinstance(display_content_for_tts, str):
                            display_content_for_tts = str(display_content_for_tts)
                        audio_path = text_to_speech(display_content_for_tts, voice=tts_voice)
                    except Exception as e:
                        print(f"ERROR: TTS generation failed: {e}")
                        
            elif st.session_state.speech_enabled and display_content_for_tts:
                # For non-script messages, use TTS
                try:
                    audio_path = text_to_speech(display_content_for_tts, voice=tts_voice)
                except Exception as e:
                    print(f"ERROR: TTS generation failed: {e}")

            # Add the assistant message to chat history
            debug_info = None
            if st.session_state.debug_mode:
                if hasattr(st.session_state.chatbot, 'sessions') and session_id in st.session_state.chatbot.sessions:
                    backend_session = st.session_state.chatbot.sessions[session_id]
                    debug_info = {
                        "phase": "Identifying your concerns",
                        "true_conditions": list(backend_session.get_true_conditions_set()),
                        "message_count": backend_session.script_message_count,
                        "is_script": is_script,
                        "script_id": script_id,
                        "disorder_key": disorder_key,
                        "script_title": script_title,
                        "response_type": "dictionary" if isinstance(response, dict) else "string"
                    }
                else:
                    debug_info = {
                        "phase": "Identifying your concerns",
                        "note": "No active backend session found"
                    }

            st.session_state.messages.append({
                "role": "assistant", 
                "content": display_content,
                "audio_path": audio_path,
                "autoplay": True,  # Set autoplay for this new message
                "is_script": is_script,
                "debug_info": debug_info
            })

            # Rerun to update the chat display
            st.rerun()

    except Exception as e:
        st.error(f"An error occurred while processing your message: {e}")
        if st.session_state.debug_mode:
            import traceback
            st.error(traceback.format_exc())

# Disclaimer
st.markdown("""
<div class="disclaimer">
    EkoMindAI provides support based on therapeutic methods, but is not a replacement for professional mental health care. 
    If you're experiencing a crisis or need immediate help, please contact a mental health professional.
</div>
""", unsafe_allow_html=True)