import json
import os
import glob
import random
import re
import time # For searching files
from openai import OpenAI # Use OpenAI library structure for OpenRouter
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---

# IMPORTANT: Set your OpenRouter API key as an environment variable
# or replace os.getenv("OPENROUTER_API_KEY") with your actual key string.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Specify the model hosted on OpenRouter

OPENROUTER_MODEL_ID = "openai/gpt-4.1-mini"

# Constants
MIN_MESSAGES_FOR_SCRIPT = 10
MAPPINGS_DIR = "mappings"
SCRIPTS_DIR = "scripts"

# Therapeutic Guidelines (for manual responses)
THERAPEUTIC_GUIDELINES = """
Act as an empathetic, supportive assistant. Follow these guidelines:

1. **Warm Welcome + Emotional Validation:** Begin with genuine empathy and validation. Examples: "I hear how difficult this has been for you.", "That sounds really challenging to go through.", "I can understand why you'd feel that way."

2. **Create a Safe Space:** Explicitly communicate safety and confidentiality. Examples: "This is a safe space to share whatever's on your mind.", "Feel free to express yourself openly here."

3. **Reflective Listening:** Demonstrate deep understanding by thoughtfully reflecting back what the user shared, including both content and emotions.

4. **Normalize Feelings:** Help users understand their reactions are natural. Examples: "Many people experience similar feelings in your situation.", "What you're describing is a common response to stress."

5. **Gentle Exploration:** Ask open-ended questions that invite deeper sharing. Examples: "Could you tell me more about how that experience affected you?", "What thoughts come up for you when that happens?"

6. **Avoid Giving Advice:** Do not provide direct advice, medical diagnoses, or definitive solutions. Focus on listening, validating, and gentle exploration.

7. **Balanced Responses:** Provide substantive, thoughtful responses (typically 3-5 sentences) that show you're truly engaged with what they've shared.

8. **Safety First:** Do not engage with harmful or crisis situations directly. Provide general crisis resource information if necessary.

9. **Warm Tone:** Maintain a consistently warm, supportive tone that conveys genuine care and interest.

10. **Use Follow-ups:** Incorporate the provided follow-up questions naturally, adapting them to the conversation flow.
"""

# use random responses to present scripts
script_responses = [
    "Based on what you've shared, I think a guided exercise called '{script_title}' might be helpful for addressing some of the challenges you're experiencing. Would you like me to share this exercise with you now? It's completely up to you, and we can continue our conversation either way.",
    
    "From what I understand about your situation, an exercise called '{script_title}' could be beneficial for the difficulties you've mentioned. Would you be interested in trying this exercise together? No pressure - we can also keep talking about other approaches.",
    
    "I wonder if an exercise known as '{script_title}' might help with what you're going through right now. Would you like to explore this technique? Feel free to say yes or no - I'm here to support whatever direction feels right for you.",
    
    "Given what you've described, an exercise called '{script_title}' might offer some relief for these challenges. Would you be open to learning about this exercise? We can certainly continue our discussion regardless of your choice.",
    
    "As I listen to your experience, I'm thinking an activity called '{script_title}' could be particularly relevant for your situation. Would you like me to walk you through it? It's entirely your decision, and we can proceed however feels most comfortable.",
    
    "Something that might help with what you're describing is a method called '{script_title}'. Would this be something you'd like to try now? There's no obligation - I'm here to follow your lead on what would be most supportive.",
    
    "Based on our conversation, I believe an activity called '{script_title}' could address some of these concerns. Would you be interested in hearing more about this exercise? We can continue our discussion either way - it's completely your choice.",
    
    "I think an activity known as '{script_title}' might be valuable for working through what you've shared. Would now be a good time to introduce this exercise? Please feel free to decline if you'd prefer to continue in another direction.",
    
    "After reflecting on what you've told me, an approach called '{script_title}' seems like it could be helpful here. Would you like me to guide you through this exercise? Whatever you decide is perfectly fine - I'm here to support your journey.",
    
    "Your experiences suggest that an exercise called '{script_title}' might provide some helpful tools for what you're facing. Would you be open to exploring this together? The choice is yours, and there's no pressure either way."
]

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
        messages = messages
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
        return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in self.conversation_history])

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

    def find_matching_script(self, identified_conditions_set: set[str]) -> tuple[str | None, str | None]:
        for disorder_key, data in self.disorder_data.items():
            mappings = data.get("scriptMappings")
            if not mappings:
                continue
            for script_id, script_data in mappings.items():
                required = set(script_data.get("required_conditions", []))
                if required and required.issubset(identified_conditions_set):
                    print(f"DEBUG: Matched script '{script_id}' from disorder '{disorder_key}'")
                    return script_id, disorder_key
        return None, None
    
    def remove_matched_script(self, script_id: str, disorder_key: str):
        mappings = self.disorder_data.get(disorder_key, {}).get("scriptMappings")
        if mappings and script_id in mappings:
            del mappings[script_id]
            print(f"DEBUG: Removed script '{script_id}' from disorder '{disorder_key}'")

    def get_script_title(self, script_id: str, disorder_key: str) -> str | None:
        return self.disorder_data.get(disorder_key, {}).get("scriptMappings", {}).get(script_id, {}).get("title")

    def get_script_content(self, script_id: str, disorder_key: str, base_scripts_dir: str) -> str | None:
        script_dir = os.path.join(base_scripts_dir, disorder_key, "scripts")
        # Match any file that starts with the script_id and ends with .txt or .docx
        pattern_txt = os.path.join(script_dir, f"{script_id}*.txt")
        pattern_docx = os.path.join(script_dir, f"{script_id}*.docx")
        matches = glob.glob(pattern_txt) + glob.glob(pattern_docx)
        if not matches:
            print(f"WARNING: Script file not found: {pattern_txt} or {pattern_docx}")
            return None
        script_path = matches[0]  # Use the first match
        # If it's a .docx, you may want to extract text (requires python-docx)
        if script_path.endswith(".docx"):
            try:
                from docx import Document
                doc = Document(script_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                print("ERROR: python-docx not installed. Cannot read .docx files.")
                return None
        else:
            with open(script_path, "r", encoding="utf-8") as f:
                return f.read()

# --- Main Chatbot Logic ---

class TherapeuticChatbot:
    """Orchestrates the chatbot flow."""
    def __init__(self):
        self.condition_manager = ConditionScriptManager(MAPPINGS_DIR)
        self.sessions: dict[str, ChatSession] = {}
        # No default disorder key needed here anymore

        if not self.condition_manager.get_disorder_keys():
             raise ValueError("No disorder mappings loaded successfully. Cannot initialize chatbot.")

    def _check_user_accepted_script(self, user_input: str) -> bool:
        prompt = f"""
You are an agent whose job is to decide whether the user has accepted the script offered to them.
This is the last user message: {user_input}

Respond with JSON in the following format:
{{
    "accepted": true | false
}}
"""
        llm = LLMClient(OPENROUTER_API_KEY, "openai/gpt-4.1-mini")
        
        response = llm.send_prompt(prompt, extract_json=True)
        print(f"INFO: User Script Acceptance Response: {response}")
        if not response:
            return False
        return response["accepted"]

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

    def _build_condition_prompt(self, history_str: str) -> str | None:
        """Builds the prompt for the condition identification LLM call using ALL conditions."""
        merged_conditions = self.condition_manager.get_merged_conditions()
        if not merged_conditions:
            print("ERROR: No merged conditions found.")
            return None
        conditions_list_str = "\n".join([f"- {key}: {val['description']}" for key, val in merged_conditions.items()])
        #print("condition_list_str: ",conditions_list_str)
        prompt = f"""
Analyze the following conversation history and determine if the user exhibits any of the listed conditions.

# Conversation History
{history_str}

# Conditions to Evaluate:
{conditions_list_str}

# Instructions:
1. For each condition, if you are confident the user is experiencing it, include its key in your response.
2. Respond ONLY with a JSON object with:
   - "present_conditions": a list of the keys that are present/true for the user.
   - "follow-ups": a list of 5 relevant, open-ended follow-up questions.

# Response (JSON only):
{{
  "present_conditions": ["key1", "key2"],
  "follow-ups": ["question 1", "question 2", ...]
}}
"""
        return prompt.strip()

    def _parse_condition_response(self, data: str | None, session: ChatSession):
        """Parses the LLM's JSON response and updates the session state across disorders."""
        if not data:
            print("WARNING: Received empty response from condition identification LLM.")
            return
        try:
            present_keys = data.get("present_conditions", [])
            # Reset all to False first
            for disorder, conditions in session.identified_conditions.items():
                for key in conditions:
                    conditions[key] = False
            # Set only present keys to True
            true_conditions_printout = []
            for key in present_keys:
                for disorder, conditions in session.identified_conditions.items():
                    if key in conditions:
                        conditions[key] = True
                        true_conditions_printout.append((disorder, key))
            # Print all true conditions after update
            if true_conditions_printout:
                print("TRUE CONDITIONS THIS CALL:")
                for disorder, key in true_conditions_printout:
                    print(f"  - [{disorder}] {key}")
            else:
                print("No conditions marked as true in this call.")
            # Handle follow-ups
            session.suggested_follow_ups = data.get("follow-ups", [])
        except Exception as e:
            print(f"ERROR: Failed to parse condition response: {e}\n{data}")

    def _build_manual_response_prompt(self, session: ChatSession) -> str:
        """Builds the prompt for generating a manual, empathetic response."""
        history_str = session.format_history_for_prompt()
        follow_ups_str = "\n".join([f"- {q}" for q in session.suggested_follow_ups]) if session.suggested_follow_ups else "None available."

        prompt = f"""
You are a supportive, empathetic therapist. Your goal is to respond to the user in a warm, validating, and thoughtful way.

# Conversation History:
{history_str}

# Your Response:
"""
        return prompt.strip()

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

        # 2. Identify Conditions across ALL disorders (LLM Call 1)
        history_str = session.format_history_for_prompt()
        condition_prompt = self._build_condition_prompt(history_str)
        condition_response = None
        if condition_prompt:
            condition_response = llm.send_prompt(condition_prompt, temperature=0.7, extract_json=True)
        self._parse_condition_response(condition_response, session)

        # 3. Check for Script Match across ALL disorders
        true_conditions_set = session.get_true_conditions_set()
        matched_script_id, matched_disorder_key = self.condition_manager.find_matching_script(true_conditions_set)

        # 4. Response Decision Logic
        ai_response_content = None
        script_rejected = False
        final_response_type = "manual"  # Default

        # Check if we're in script offering mode from previous turn
        if session.offering_script:
            # Retrieve details from the stored offer
            offered_script_id = session.offering_script.get("script_id")
            offered_script_title = session.offering_script.get("script_title")
            offered_disorder_key = session.offering_script.get("disorder_key")

            # Clear the offering state regardless of user's answer
            session.offering_script = None

            user_accepted = self._check_user_accepted_script(user_message)

            if user_accepted and offered_script_id and offered_disorder_key:
                # User accepted, deliver the script using the stored disorder key
                script_content = self.condition_manager.get_script_content(offered_script_id, offered_disorder_key, SCRIPTS_DIR)
                if script_content:
                    lead_in = f"I'm glad you're open to trying this. Here's the '{offered_script_title}' exercise:\n\n---\n"
                    formatted_content = f"{lead_in}SCRIPT_START\n{script_content}\nSCRIPT_END"
                    final_response_type = "script"
                    print(f"INFO: Delivering script '{offered_script_id}' (Disorder: {offered_disorder_key}) after user acceptance")
                    session.script_message_count = 0
                    # Mark this script as delivered
                    session.delivered_scripts.add((offered_script_id, offered_disorder_key))
                    
                    # Update the session history with just the formatted content for conversation record
                    session.add_message("assistant", formatted_content)
                    
                    # Return a dictionary with metadata instead of just the content
                    return {
                        "content": formatted_content,
                        "is_script": True,
                        "script_id": offered_script_id,
                        "script_title": offered_script_title,
                        "disorder_key": offered_disorder_key
                    }
                else:
                    print(f"WARNING: Failed to load script '{offered_script_id}' (Disorder: {offered_disorder_key}) content after user acceptance")
                    ai_response_content = f"I apologize, but I'm having trouble retrieving the exercise I mentioned. Let's continue our conversation instead. How have you been feeling lately?"
                    final_response_type = "manual"
            else:
                # User declined or gave ambiguous response
                session.script_message_count = 0
                self.condition_manager.remove_matched_script(offered_script_id, offered_disorder_key)
                
                script_rejected = True
                final_response_type = "manual"

        # Normal flow (not responding to script offer)
        elif matched_script_id and matched_disorder_key and session.script_message_count >= MIN_MESSAGES_FOR_SCRIPT:
            # Only offer if not already delivered
            if (matched_script_id, matched_disorder_key) not in session.delivered_scripts:
                print(f"INFO: Condition match for script '{matched_script_id}' (Disorder: {matched_disorder_key}) and message count ({session.script_message_count}) threshold met.")
                script_title = self.condition_manager.get_script_title(matched_script_id, matched_disorder_key) or matched_script_id
                session.offering_script = {
                    "script_id": matched_script_id,
                    "script_title": script_title,
                    "disorder_key": matched_disorder_key
                }
                ai_response_content = script_responses[random.randint(0, len(script_responses) - 1)].format(script_title=script_title)
                final_response_type = "manual"
                print(f"INFO: Offering script '{matched_script_id}' (Disorder: {matched_disorder_key}) to user")
            else:
                # Script already delivered, do not re-offer
                ai_response_content = (
                    "We've already explored the main exercise I can offer for your situation. "
                    "Let's continue our conversation and see how else I can support you."
                )
                final_response_type = "manual"

        elif matched_script_id:
            # Script matched but message count too low
            print(f"INFO: Condition match for script '{matched_script_id}' (Disorder: {matched_disorder_key}) but message count ({session.script_message_count}) is less than {MIN_MESSAGES_FOR_SCRIPT}. Generating manual response.")
            final_response_type = "manual"
        else:
            # No script conditions met anywhere
            final_response_type = "manual"

        # 5. Generate Manual Response if needed (LLM Call 2)
        if final_response_type == "manual" and not ai_response_content:
            # Manual prompt generation remains the same, using the general guidelines
            messages = [
                {"role": "system", "content": "You are a supportive, empathetic therapist. Your goal is to respond to the user in a warm, validating, and thoughtful way."},
                *session.conversation_history,
                {"role": "user", "content": user_message}
            ]
            ai_response_content = llm.send_prompt(prompt=None, messages=messages, temperature=0.8)
            if not ai_response_content:
                print("ERROR: Failed to generate manual response from LLM. Using fallback.")
                ai_response_content = "I understand. It sounds like a difficult situation. Could you tell me a little more about that?"  # Generic fallback
                
            if script_rejected:
                ai_response_content = f"It is alright, we can continue our conversation without the exercise.\n{ai_response_content}"

        # 6. Update session history and return for manual responses
        if final_response_type == "manual":
            session.add_message("assistant", ai_response_content)
            return ai_response_content
        
        # For script responses, we already returned earlier
        # This should not be reached if we have a script, but just in case:
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
