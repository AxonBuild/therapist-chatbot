import os
import streamlit as st
import json
import time
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Import your chatbot backend
#from streaming  import StreamingMedicalChatbot as MedicalChatbot
from backend import MedicalChatbot
# Load environment variables
load_dotenv()

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

# Debug mode toggle in sidebar
with st.sidebar:
    st.title("Settings")
    debug_toggle = st.toggle("Debug Mode", value=st.session_state.debug_mode)
    if debug_toggle != st.session_state.debug_mode:
        st.session_state.debug_mode = debug_toggle
        st.rerun()
    
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
    
    # Check if we have a therapeutic script
    if st.session_state.chatbot.session.therapeutic_script and not st.session_state.therapeutic_script:
        # Save the script to session state
        st.session_state.therapeutic_script = st.session_state.chatbot.session.therapeutic_script
        
        # Display regular response in a chat message
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add the regular response to the chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "debug_info": {
                "phase": st.session_state.current_phase,
                "disorder": st.session_state.chatbot.session.identified_disorder,
                "confidence": st.session_state.chatbot.session.disorder_confidence,
                "current_node": st.session_state.chatbot.session.current_node_id
            } if st.session_state.debug_mode else None
        })
        
        # Add a brief pause for effect
        time.sleep(1)
        
        # Display script in a separate chat message
        with st.chat_message("assistant"):
            st.markdown(f"<div class='therapy-script'>{st.session_state.therapeutic_script}</div>", unsafe_allow_html=True)
        
        # Add the script to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": st.session_state.therapeutic_script,
            "is_script": True
        })
    else:
        # Normal response - display in chat message
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "debug_info": {
                "phase": st.session_state.current_phase,
                "disorder": st.session_state.chatbot.session.identified_disorder,
                "confidence": st.session_state.chatbot.session.disorder_confidence if st.session_state.chatbot.session.identified_disorder else 0.0,
                "current_node": st.session_state.chatbot.session.current_node_id
            } if st.session_state.debug_mode else None
        })
                
            
            
        # Force a rerun to update the interface with any state changes
        st.rerun()
    
            


# Disclaimer
st.markdown("""
<div class="disclaimer">
    EkoMindAI provides support based on therapeutic methods, but is not a replacement for professional mental health care. 
    If you're experiencing a crisis or need immediate help, please contact a mental health professional.
</div>
""", unsafe_allow_html=True)