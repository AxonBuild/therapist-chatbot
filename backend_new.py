import json
import os
import re
import time # For searching files
from openai import OpenAI # Use OpenAI library structure for OpenRouter
from dotenv import load_dotenv

from rag import fetch_guidance_notes
load_dotenv()
# --- Configuration ---

# IMPORTANT: Set your OpenRouter API key as an environment variable
# or replace os.getenv("OPENROUTER_API_KEY") with your actual key string.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Specify the model hosted on OpenRouter
OPENROUTER_MODEL_ID = "openai/gpt-4.1-mini"

def compute_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\n>> Function {func.__name__} took {end_time - start_time:.2f} seconds to execute.\n")
        return result
    return wrapper

# --- LLM Client ---

class LLMClient:
    """Handles communication with the LLM via OpenRouter."""
    def __init__(self, api_key: str, model_id: str):
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        self.model_id = model_id
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=OPENROUTER_API_KEY)

    @compute_time
    def send_prompt(self, prompt: str, temperature: float = 0.5, extract_json: bool = False, messages=[]) -> str | None:
        """Sends a prompt to the LLM and returns the text response."""
        messages = messages.copy()
        if len(messages) == 0:
            messages.append({"role": "user", "content": prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature
            )
            response = completion.choices[0].message.content
            response = response.strip() if response else None
            
            if extract_json and response:
                response = self._extract_json(response)                
            return response
        except Exception as e:
            print(f"ERROR: LLM API call failed: {e}")
            return None

    def _extract_json(self, response: str):
        try:
            extracted = re.findall(r'```(json)*([\s\S]*?)```', response)
            if len(extracted) == 0:
                extracted = re.findall(r'({[\s\S]*})', response)

            if len(extracted) == 0:
                raise ValueError
            elif len(extracted) != 1:
                print("*important* Incorrect format of the response. More than 1 json found. Using the first one.", response)
                response = extracted[0]

            data = extracted[0][-1] if type(extracted[0]) in [list, tuple] else extracted[0]
            return json.loads(data)
        except:
            print(f"\Failed to extract json from:\n{response}\n\n-------")
            raise

# --- Session State ---

class ChatSession:
    """Stores the state of a single user conversation."""
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.conversation_history: list[dict] = []
        self.identified_conditions: dict[str, dict[str, bool]] = {}
        self.suggested_follow_ups: list[str] = []
        self.script_message_count: int = 0
        self.offering_script: dict | None = None
        self.delivered_scripts: set[tuple[str, str]] = set()

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        self.script_message_count += 1

    def get_true_conditions_set(self) -> set[str]:
        true_set = set()
        for disorder, conditions in self.identified_conditions.items():
            for key, value in conditions.items():
                if value:
                    true_set.add(key)
        return true_set

    def format_history_for_prompt(self) -> str:
        return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in self.conversation_history if m['role'] != 'assistant'])

    
# --- Main Chatbot Logic ---

class TherapeuticChatbot:
    """Orchestrates the chatbot flow."""
    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}

    def _get_or_create_session(self, session_id: str) -> ChatSession:
        """Retrieves an existing session or creates a new one with all conditions."""
        if session_id not in self.sessions:
            print(f"INFO: Creating new session: {session_id}")
            self.sessions[session_id] = ChatSession(session_id)
        return self.sessions[session_id]

    def _build_keyword_identifier_prompt(self, user_messages_list: str) -> str:
        prompt = f"""
            # Task
            You are given messages from a patient talking to a therapist. You task is to extract ALL the symptoms and conditions that the patient is experiencing from the list of messages.

            # Messages
            {user_messages_list}

            # Instructions for answering:
            Keep your output precise and short.

            # Response Format Guidelines
            Give your output in the following JSON format:
            {{
                "symptoms": ["list of symptoms"]
            }}
        """

        return prompt
    
    def _build_system_prompt(self, guidance_notes: str) -> str:
        flag = True if guidance_notes else False
        
        prompt = f"""
            You are a supportive, empathetic therapist. Your goal is to respond to the user in a warm, validating, and thoughtful way.

            {"Helpful material you can use:" if flag else ""}
            {guidance_notes}
            """
        # print("## SYSTEM PROMPT:\n", prompt, "\n##\n")
        return prompt
    
    @compute_time
    def process_message(self, user_message: str, session_id: str = "default", model=None) -> dict | str:
        """Processes a user message and returns the assistant's response.
        
        Returns:
            For regular responses: a string containing the assistant's message
            For script responses: a dictionary with script metadata including content, script_id, and disorder_key
        """
        # 1. Get/Create Session & Update History
        llm = LLMClient(api_key=OPENROUTER_API_KEY, model_id=model)
        
        session = self._get_or_create_session(session_id)
        session.add_message("user", user_message)

        history_str = session.format_history_for_prompt()
        
        keyword_prompt = self._build_keyword_identifier_prompt(history_str)
        keyword_response = llm.send_prompt(keyword_prompt, temperature=0.7, extract_json=True)
        symptoms = keyword_response.get("symptoms", [])
        
        guidance_notes = fetch_guidance_notes(symptoms)
        
        messages = [
            {"role": "system", "content": self._build_system_prompt(guidance_notes=guidance_notes)},
            *session.conversation_history
        ]
        
        ai_response_content = llm.send_prompt(prompt=None, messages=messages, temperature=0.8)
        if not ai_response_content:
            print("ERROR: Failed to generate manual response from LLM. Using fallback.")
            ai_response_content = "I understand. It sounds like a difficult situation. Could you tell me a little more about that?"  # Generic fallback
                
        
        session.add_message("assistant", ai_response_content)
        return ai_response_content
        
# --- Example Usage ---
if __name__ == "__main__":
    print("Initializing Chatbot...")
    # Ensure you have OPENROUTER_API_KEY set in your environment
    if not OPENROUTER_API_KEY:
        print("\n!!! WARNING: OPENROUTER_API_KEY environment variable not set. LLM calls will fail. !!!\n")

    # Ensure python-docx is installed if you have .docx scripts
    try:
        import docx
    except ImportError:
        print("\n--- NOTE: 'python-docx' not installed. .docx scripts cannot be loaded. Run: pip install python-docx ---\n")