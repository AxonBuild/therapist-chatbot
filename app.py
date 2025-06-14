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

model_options = [
    "openai/gpt-4o",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat-v3-0324",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.0-flash-001",
    "google/gemini-flash-1.5",
    "anthropic/claude-3.7-sonnet",
    "meta-llama/llama-4-maverick",
]

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

def get_git_commit_id(length=7):
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", f"HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            return result.stdout.strip()[:length]
        else:
            return "unknown"
    except Exception as e:
        print(f"Error getting git commit ID: {e}")
        return "unknown"

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
    
    /* Updated script styling */
    .therapy-script {
        background-color: #f9f4f9;
        border-left: 5px solid #5E3B50;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    .therapy-script h1, .therapy-script h2, .therapy-script h3 {
        color: #5E3B50;
        margin-bottom: 1rem;
    }
    
    .script-metadata {
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #666;
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

if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_options[0]

if "speech_enabled" not in st.session_state:
    st.session_state.speech_enabled = False

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
with st.sidebar:
    st.title("Settings")
    # Template customization section
    st.subheader("Prompt Customization")
    use_custom_template = st.checkbox("Use Custom Prompt Template")
    
    custom_template = None
    if use_custom_template:
        st.write("**Edit System Prompt:**")
        
        # Initialize default template if not in session state
        if "custom_template_text" not in st.session_state:
            st.session_state.custom_template_text = "You are a supportive, empathetic therapist. Your goal is to respond to the user in a warm, validating, and thoughtful way.\n\n{clinical_guidance_placeholder}"
        
        # Single editable text area for the complete template with key parameter
        template_text = st.text_area(
            "System Prompt Template:",
            value=st.session_state.custom_template_text,
            height=200,
            key="template_text_area",
            help="Edit the complete system prompt. Use {clinical_guidance_placeholder} where you want clinical guidance to be inserted."
        )
        
        # Only update session state if the value actually changed
        if template_text != st.session_state.custom_template_text:
            st.session_state.custom_template_text = template_text
        
        custom_template = {
            "template_text": st.session_state.custom_template_text
        }
        
        # Reset to defaults button
        if st.button("Reset to Default Template"):
            st.session_state.custom_template_text = "You are a supportive, empathetic therapist. Your goal is to respond to the user in a warm, validating, and thoughtful way.\n\n{clinical_guidance_placeholder}"
            st.rerun()
    

    # Debug mode toggle
    if os.getenv("ENV") != "production":
        debug_toggle = st.toggle("Debug Mode", value=st.session_state.debug_mode)
        if debug_toggle != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_toggle
            st.rerun()
    
    # Text-to-speech toggle
    speech_toggle = st.toggle("Text-to-Speech", value=st.session_state.speech_enabled)
    if speech_toggle != st.session_state.speech_enabled:
        st.session_state.speech_enabled = speech_toggle
        st.rerun()
        
    # Model selection
    selected_model = st.selectbox(
        "Model", 
        options=model_options,
        index=model_options.index(st.session_state.selected_model)
    )
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
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

# --- In the message display section ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Check if this is a therapeutic script message
        if msg.get("is_script", False):
            # MODIFIED: Display different styling for scripts without content
            if "script_id" in msg:
                st.markdown(f"""
                <div class='therapy-script'>
                    <h3>🎯 Script Selected: {msg.get("script_id")}</h3>
                    <p>This script has been selected based on the conversation.</p>
                    <p><em>The script content will be available in a future version.</em></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='therapy-script'>{msg['content']}</div>", unsafe_allow_html=True)
            
            # Display script metadata if available
            if msg.get("metadata"):
                meta = msg["metadata"]
                # Extract values with proper fallbacks
                level = meta.get('Level', meta.get('level', 'N/A'))
                target = meta.get('Target Population', meta.get('target_population', 'N/A'))
                duration = meta.get('Estimated Duration (min)', meta.get('estimated_duration', 'N/A'))
                
                st.markdown(f"""<div class="script-metadata">
                    <div>Level: {level} | Target: {target}</div>
                    <div>Duration: ~{duration} min</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])
        
        # Play audio if available
        if msg["role"] == "assistant" and st.session_state.speech_enabled and msg.get("audio_path"):
            with open(msg["audio_path"], "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3", autoplay=msg.get("autoplay", False))
    
    # Display debug information if enabled
    if st.session_state.debug_mode and msg.get("debug_info"):
        with st.expander("Debug Info"):
            st.json(msg["debug_info"])

# --- Inside the chat input section where responses are handled ---
if user_input := st.chat_input("Share what's on your mind..."):
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Add user message to frontend chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
   
    # Process the message through the chatbot backend
    session_id = "streamlit_user_session"
    try:
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.process_message(
                user_input, 
                session_id=session_id,
                model=st.session_state.selected_model,
                custom_template=custom_template
            )

            # Handle structured response (new format with script)
            if isinstance(response, dict) and "script_offered" in response:
                # First display the normal response
                normal_response = response.get("response", "")
                
                # Generate audio for normal response if enabled
                normal_audio_path = None
                if st.session_state.speech_enabled:
                    normal_audio_path = text_to_speech(normal_response, voice=st.session_state.selected_voice)
                
                # Add normal response to session
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": normal_response,
                    "audio_path": normal_audio_path,
                    "autoplay": True
                })
                
                # Get script info
                script_id = response.get("script_id", "")
                
                # MODIFIED: Display script filename instead of content
                script_display = f"**Script File: {script_id}**\n\nScript would play here when available."
                
                # For debugging, keep script content available but don't display it
                script_content = response.get("script_content", "")
                
                # Generate audio for script (commented out)
                script_audio_path = None
                # if st.session_state.speech_enabled and script_content:
                #     script_audio_path = text_to_speech(script_content, voice=st.session_state.selected_voice)
                
                # Debug info for script
                debug_info = None
                if st.session_state.debug_mode:
                    debug_info = {
                        "script_id": script_id,
                        "metadata": response.get("metadata", {}),
                        # Add the first 100 chars of script content for verification
                        "script_content_preview": script_content[:100] + "..." if script_content else "No content"
                    }
                    
                    # Add guidance delivery status if available
                    if hasattr(st.session_state.chatbot, 'sessions'):
                        session = st.session_state.chatbot.sessions.get(session_id)
                        if hasattr(session, "delivered_guidance"):
                            debug_info["delivered_guidance"] = session.delivered_guidance
                    
                # Add script to messages
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": script_display,  # Display filename instead
                    "is_script": True,
                    "script_id": script_id,  # Store script ID for reference
                    "audio_path": script_audio_path,
                    "autoplay": False,
                    "metadata": response.get("metadata", {}),
                    "debug_info": debug_info
                })
                
            else:
                # Handle normal text response (old format)
                audio_path = None
                if st.session_state.speech_enabled:
                    audio_path = text_to_speech(response, voice=st.session_state.selected_voice)
                
                # Debug info for normal response
                debug_info = None
                if st.session_state.debug_mode:
                    debug_info = {
                        "response_type": "text"
                    }
                    
                    # Add guidance delivery status if available
                    if hasattr(st.session_state.chatbot, 'sessions'):
                        session = st.session_state.chatbot.sessions.get(session_id)
                        if hasattr(session, "delivered_guidance"):
                            debug_info["delivered_guidance"] = session.delivered_guidance
                
                # Store normal response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "audio_path": audio_path,
                    "autoplay": True,
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
commit_id = get_git_commit_id()
st.markdown(f"""
<div class="disclaimer">
    EkoMindAI provides support based on therapeutic methods, but is not a replacement for professional mental health care. 
    If you're experiencing a crisis or need immediate help, please contact a mental health professional.
    <br><br>
    <small>App Version: {commit_id}</small>
</div>
""", unsafe_allow_html=True)