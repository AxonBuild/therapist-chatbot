import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

# Import the real implementation
from backend import MedicalChatbot

class StreamingMedicalChatbot:
    """Wrapper class around MedicalChatbot to provide streaming responses"""
    
    def __init__(self):
        """Initialize the streaming medical chatbot."""
        self.chatbot = MedicalChatbot()
    
    def process_message(self, user_message: str) -> str:
        """Process a user message and return the response (non-streaming)."""
        return self.chatbot.process_message(user_message)
    
    def process_message_stream(self, user_message: str) -> Iterator[Dict[str, Any]]:
        """Process a user message and yield streaming response chunks."""
        # Add message to the chatbot's session
        self.chatbot.session.add_message("user", user_message)
        
        # Get the current phase
        is_first_phase = not self.chatbot.session.identified_disorder or self.chatbot.session.disorder_confidence < 0.7
        
        # Get base response
        if is_first_phase:
            response = self._handle_first_phase_stream(user_message)
        else:
            response = self._handle_second_phase_stream(user_message)
        
        # Add assistant response to the chatbot's session
        self.chatbot.session.add_message("assistant", response)
        
        # First chunk with metadata
        yield {
            "chunk": response[:20],  # First part of the response
            "is_first_chunk": True,
            "is_last_chunk": False,
            "phase": "identification" if is_first_phase else "navigation",
            "disorder": self.chatbot.session.identified_disorder,
            "confidence": self.chatbot.session.disorder_confidence,
            "at_leaf_node": self.chatbot.session.at_leaf_node,
            "current_node": self.chatbot.session.current_node
        }
        
        # Stream the rest of the response in small chunks
        chunk_size = 10  # Small chunk size for smoother streaming
        remaining = response[20:]
        chunks = [remaining[i:i+chunk_size] for i in range(0, len(remaining), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            yield {
                "chunk": chunk,
                "is_first_chunk": False,
                "is_last_chunk": is_last
            }
            
    def _handle_first_phase_stream(self, user_message: str) -> str:
        """Handle the first phase (disorder identification) with streaming."""
        response = self.chatbot._handle_disorder_identification()
        return response
    
    def _handle_second_phase_stream(self, user_message: str) -> str:
        """Handle the second phase (tree navigation) with streaming."""
        response = self.chatbot._handle_tree_navigation(user_message)
        return response
    
    def get_therapeutic_script(self) -> Optional[str]:
        """Get the therapeutic script if one has been identified."""
        return self.chatbot.session.therapeutic_script
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the chatbot."""
        return {
            "identified_disorder": self.chatbot.session.identified_disorder,
            "disorder_confidence": self.chatbot.session.disorder_confidence,
            "current_node": self.chatbot.session.current_node,
            "at_leaf_node": self.chatbot.session.at_leaf_node,
            "has_therapeutic_script": bool(self.chatbot.session.therapeutic_script)
        }
    
    def reset_session(self) -> None:
        """Reset the chatbot session."""
        self.chatbot.reset_session()