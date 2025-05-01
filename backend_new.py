import json
import os
import glob
import random
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

# Constants
MAPPINGS_DIR = "mappings"
SCRIPTS_DIR = "scripts"

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
                
            # print(f"\n--- LLM Request ---\nModel: {self.model_id}\nTemp: {temperature}\nPrompt:\n{prompt}\n--- LLM Response ---\n{response}\n---\n") # Debug
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
    def __init__(self, session_id: str, all_initial_conditions: dict[str, dict[str, str]]):
        self.session_id: str = session_id
        self.conversation_history: list[dict] = []
        self.identified_conditions: dict[str, dict[str, bool]] = {}
        for disorder, conditions in all_initial_conditions.items():
            self.identified_conditions[disorder] = {key: False for key in conditions}
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

# --- Condition and Script Management ---

class ConditionScriptManager:
    """Loads and manages all disorder conditions and script mappings in a merged structure."""
    def __init__(self, mappings_dir: str):
        self.mappings_dir = mappings_dir
        self.disorder_data = {}  # {disorder_key: {"conditions": {...}, "scriptMappings": {...}}}
        self.merged_conditions = {}  # {key: {"description": ..., "disorder": ...}}
        self._load_and_merge_mappings()

    def _load_and_merge_mappings(self):
        """Loads all JSON mapping files and merges conditions into a single dict."""
        for filename in os.listdir(self.mappings_dir):
            if filename.endswith(".json"):
                base = os.path.splitext(filename)[0]  # e.g., 'sleep_mappings'
                if base.endswith("_mappings"):
                    disorder_key = base[:-len("_mappings")]
                else:
                    disorder_key = base
                conditions_key = f"{disorder_key}_conditions"
                with open(os.path.join(self.mappings_dir, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print("DEBUG: Looking for conditions_key:", conditions_key, "in", filename)
                    if conditions_key in data and "scriptMappings" in data:
                        # Merge conditions
                        for key, desc in data[conditions_key].items():
                            if key in self.merged_conditions:
                                print(f"WARNING: Duplicate condition key '{key}' found in '{disorder_key}'.")
                            self.merged_conditions[key] = {
                                "description": desc,
                                "disorder": disorder_key
                            }
                        # Store full mapping for script logic
                        self.disorder_data[disorder_key] = {
                            "conditions": data[conditions_key],
                            "scriptMappings": data["scriptMappings"]
                        }
                        #print("disorder data",self.disorder_data[disorder_key])

                        print(f"INFO: Loaded and merged mappings for '{disorder_key}'")
                    else:
                        print(f"WARNING: Could not find valid keys in {filename}. Found keys: {list(data.keys())}")

    def get_merged_conditions(self) -> dict:
        return self.merged_conditions

    def get_disorder_keys(self) -> list[str]:
        return list(self.disorder_data.keys())

    def get_conditions_for_disorder(self, disorder_key: str) -> dict:
        return self.disorder_data.get(disorder_key, {}).get("conditions", {})
    
# --- Main Chatbot Logic ---

class TherapeuticChatbot:
    """Orchestrates the chatbot flow."""
    def __init__(self):
        self.condition_manager = ConditionScriptManager(MAPPINGS_DIR)
        self.sessions: dict[str, ChatSession] = {}
        # No default disorder key needed here anymore

        if not self.condition_manager.get_disorder_keys():
             raise ValueError("No disorder mappings loaded successfully. Cannot initialize chatbot.")

    def _get_or_create_session(self, session_id: str) -> ChatSession:
        """Retrieves an existing session or creates a new one with all conditions."""
        if session_id not in self.sessions:
            print(f"INFO: Creating new session: {session_id} with all loaded conditions.")
            # Initialize with conditions from *all* loaded disorders
            all_initial_conditions = {}
            loaded_disorders = self.condition_manager.get_disorder_keys()
            if not loaded_disorders:
                 raise RuntimeError("Failed to load any disorder conditions during session creation.")

            for disorder_key in loaded_disorders:
                 conditions = self.condition_manager.get_conditions_for_disorder(disorder_key)
                 if conditions:
                     all_initial_conditions[disorder_key] = conditions
                 else:
                     print(f"WARNING: Failed to load conditions for '{disorder_key}' during session init.")

            if not all_initial_conditions:
                 raise RuntimeError("Failed to gather any initial conditions for the session.")

            self.sessions[session_id] = ChatSession(session_id, all_initial_conditions)
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
        print("## SYSTEM PROMPT:\n", prompt, "\n##\n")
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
        
        print(f"\nSYMPTOMS IDENTIFIED: {symptoms}\n")
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


    chatbot = TherapeuticChatbot()
    session_id = "user_test_123"

    print("\n--- Turn 1 ---")
    response1 = chatbot.process_message("I can't sleep", session_id)
    print(f"\nAssistant Response 1:\n{response1}")

    print("\n--- Turn 2 ---")
    response2 = chatbot.process_message("my mind seems too active. there is no rest or calm", session_id)
    print(f"\nAssistant Response 2:\n{response2}")

    # Simulate more turns to potentially trigger script
    print("\n--- Turn 3 (Simulated) ---")
    response3 = chatbot.process_message("It's like worries just race around.", session_id)
    print(f"\nAssistant Response 3:\n{response3}")

    print("\n--- Turn 4 (Simulated) ---")
    response4 = chatbot.process_message("Yeah, mostly about work stuff and things I forgot to do.", session_id)
    print(f"\nAssistant Response 4:\n{response4}")

    # Turn 5 - Message count reaches 9 (4 user + 4 AI + 1 user). Should be >= 5
    print("\n--- Turn 5 (Check Script Trigger) ---")
    response5 = chatbot.process_message("It makes it impossible to fall asleep.", session_id)
    print(f"\nAssistant Response 5:\n{response5}")

    # Optional: Inspect final session state
    # final_session = chatbot.sessions.get(session_id)
    # if final_session:
    #     print("\n--- Final Session State ---")
    #     print("Identified Conditions:")
    #     print(json.dumps(final_session.identified_conditions, indent=2))
    #     print("\nSuggested Follow-ups:")
    #     print(final_session.suggested_follow_ups)
    #     print(f"\nMessage Count: {final_session.message_count}")
