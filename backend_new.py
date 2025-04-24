import json
import os
import glob # For searching files
from openai import OpenAI # Use OpenAI library structure for OpenRouter
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---

# IMPORTANT: Set your OpenRouter API key as an environment variable
# or replace os.getenv("OPENROUTER_API_KEY") with your actual key string.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Specify the model hosted on OpenRouter

OPENROUTER_MODEL_ID = "openai/gpt-4.1-mini"
OPENROUTER_SITE_NAME = "your_site_name" # Added site name

# Constants
MIN_MESSAGES_FOR_SCRIPT = 5
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

# --- LLM Client ---

class LLMClient:
    """Handles communication with the LLM via OpenRouter."""
    def __init__(self, api_key: str, model_id: str):
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        self.model_id = model_id
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=OPENROUTER_API_KEY)

    def send_prompt(self, prompt: str, temperature: float = 0.5, system_message: str | None = None) -> str | None:
        """Sends a prompt to the LLM and returns the text response."""
        messages = []
        if system_message:
             messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature
            )
            response = completion.choices[0].message.content
            # print(f"\n--- LLM Request ---\nModel: {self.model_id}\nTemp: {temperature}\nPrompt:\n{prompt}\n--- LLM Response ---\n{response}\n---\n") # Debug
            return response.strip() if response else None
        except Exception as e:
            print(f"ERROR: LLM API call failed: {e}")
            return None

# --- Session State ---

class ChatSession:
    """Stores the state of a single user conversation."""
    def __init__(self, session_id: str, all_initial_conditions: dict[str, dict[str, str]]):
        self.session_id: str = session_id
        self.conversation_history: list[dict] = []
        # Store conditions nested by disorder: {disorder_key: {condition_key: bool}}
        self.identified_conditions: dict[str, dict[str, bool]] = {}
        for disorder, conditions in all_initial_conditions.items():
            self.identified_conditions[disorder] = {key: False for key in conditions}

        self.suggested_follow_ups: list[str] = []
        self.message_count: int = 0 # Total messages (user + assistant)
        self.offering_script: dict | None = None # Track if we're offering a script {script_id: str, script_title: str, disorder_key: str}

    def add_message(self, role: str, content: str):
        """Adds a message to the history and increments count."""
        self.conversation_history.append({"role": role, "content": content})
        self.message_count += 1

    def get_true_conditions_set(self) -> set[str]:
        """Returns a flat set of condition keys currently marked as True across all disorders."""
        true_set = set()
        for disorder, conditions in self.identified_conditions.items():
            for key, value in conditions.items():
                if value:
                    true_set.add(key)
        return true_set

    def format_history_for_prompt(self) -> str:
        """Formats the conversation history into a simple string for the LLM."""
        formatted = []
        for msg in self.conversation_history:
            role = "AI" if msg["role"] == "assistant" else "USER"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

# --- Condition and Script Management ---

class ConditionScriptManager:
    """Loads and manages disorder conditions and script mappings."""
    def __init__(self, mappings_dir: str):
        self.mappings_dir = mappings_dir
        self.disorder_data = {} # Stores data like {"sleep": {"conditions": {...}, "scriptMappings": {...}}, "anxiety": {...}}
        self._all_conditions_map: dict[str, str] = {} # Map: {description: disorder_key}
        self._load_all_mappings()

    def _load_all_mappings(self):
        """Loads all JSON mapping files and builds the all_conditions_map."""
        if not os.path.isdir(self.mappings_dir):
            print(f"WARNING: Mappings directory not found: {self.mappings_dir}")
            return
        for filename in os.listdir(self.mappings_dir):
            if filename.endswith("_mappings.json"):
                disorder_key = filename.replace("_mappings.json", "")
                filepath = os.path.join(self.mappings_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Basic validation
                        if isinstance(data, dict) and "scriptMappings" in data:
                            # Find the conditions key (e.g., "sleep_conditions")
                            conditions_key = f"{disorder_key}_conditions"
                            if conditions_key in data and isinstance(data[conditions_key], dict):
                                self.disorder_data[disorder_key] = {
                                    "conditions": data[conditions_key],
                                    "scriptMappings": data["scriptMappings"]
                                }
                                # Add conditions to the global map
                                for key, desc in data[conditions_key].items():
                                    if desc in self._all_conditions_map:
                                        print(f"WARNING: Duplicate condition description found: '{desc}' in '{disorder_key}' and '{self._all_conditions_map[desc]}'. Using first found.")
                                    else:
                                        self._all_conditions_map[desc] = disorder_key
                                print(f"INFO: Loaded mappings for '{disorder_key}'")
                            else:
                                print(f"WARNING: Could not find or invalid conditions key '{conditions_key}' in {filename}")
                        else:
                            print(f"WARNING: Invalid format or missing keys in {filename}")
                except json.JSONDecodeError:
                    print(f"ERROR: Failed to decode JSON from {filename}")
                except Exception as e:
                    print(f"ERROR: Failed to load or process {filename}: {e}")

    def get_all_condition_descriptions(self) -> list[str]:
        """Gets a list of all condition descriptions across all disorders."""
        return list(self._all_conditions_map.keys())

    def get_disorder_and_key_from_description(self, description: str) -> tuple[str | None, str | None]:
        """Finds the disorder and internal condition key based on its description."""
        disorder_key = self._all_conditions_map.get(description)
        if disorder_key:
            conditions = self.get_conditions_for_disorder(disorder_key)
            if conditions:
                for key, desc in conditions.items():
                    if desc == description:
                        return disorder_key, key
        return None, None # Return None for both if not found

    def find_matching_script(self, identified_conditions_set: set[str]) -> tuple[str | None, str | None]:
        """
        Finds the first script whose requirements are met by the identified conditions.
        Checks across all loaded disorders.
        Returns (script_id, disorder_key) or (None, None).
        """
        for disorder_key, data in self.disorder_data.items():
            mappings = data.get("scriptMappings")
            if not mappings:
                continue

            # Get conditions relevant *only* to this disorder for checking its scripts
            # Note: identified_conditions_set contains keys like 'sleep_onset_insomnia'
            # We need to filter this set based on the current disorder_key if keys are prefixed,
            # or assume the set contains globally unique keys if not prefixed.
            # Assuming keys are unique across disorders (e.g., 'sleep_onset_insomnia', 'anxiety_panic_attack')
            # If keys are NOT unique (e.g., 'difficulty_sleeping' in multiple files), this needs refinement.

            for script_id, script_data in mappings.items():
                required = set(script_data.get("required_conditions", []))
                # Check if the required conditions for *this script* are a subset
                # of the *globally identified* true conditions.
                if required and required.issubset(identified_conditions_set):
                    print(f"DEBUG: Matched script '{script_id}' from disorder '{disorder_key}'")
                    return script_id, disorder_key # Return the first match

        return None, None # No match found across all disorders

    def get_disorder_keys(self) -> list[str]:
        """Returns a list of loaded disorder keys (e.g., ['sleep', 'anxiety'])."""
        return list(self.disorder_data.keys())

    def get_conditions_for_disorder(self, disorder_key: str) -> dict | None:
        """Gets the condition dictionary {key: description} for a disorder."""
        return self.disorder_data.get(disorder_key, {}).get("conditions")

    def get_condition_descriptions(self, disorder_key: str) -> list[str] | None:
        """Gets a list of condition descriptions for a disorder."""
        conditions = self.get_conditions_for_disorder(disorder_key)
        return list(conditions.values()) if conditions else None

    def get_condition_key_from_description(self, description: str, disorder_key: str) -> str | None:
        """Finds the internal condition key based on its description."""
        conditions = self.get_conditions_for_disorder(disorder_key)
        if conditions:
            for key, desc in conditions.items():
                if desc == description:
                    return key
        return None

    def get_script_title(self, script_id: str, disorder_key: str) -> str | None:
        """Gets the title of a specific script."""
        return self.disorder_data.get(disorder_key, {}).get("scriptMappings", {}).get(script_id, {}).get("title")

    def get_script_content(self, script_id: str, disorder_key: str, script_dir: str) -> str | None:
        """Loads script content by searching the relevant disorder directory."""
        script_dir = os.path.join(script_dir, disorder_key, "scripts")

        if not os.path.isdir(script_dir):
            print(f"ERROR: Script directory not found: {script_dir}")
            return None

        found_file_path = None
        try:
            # Use glob to find files starting with the script_id
            # This handles potential variations after the ID more easily
            search_pattern = os.path.join(script_dir, f"{script_id}*")
            possible_files = glob.glob(search_pattern)

            if not possible_files:
                print(f"ERROR: No file starting with '{script_id}' found in {script_dir}")
                return None

            # Filter for actual files (glob might return directories if names match)
            found_files = [f for f in possible_files if os.path.isfile(f)]

            if not found_files:
                 print(f"ERROR: Pattern '{script_id}*' matched directories but no files in {script_dir}")
                 return None

            found_file_path = found_files[0] # Use the first file found
            if len(found_files) > 1:
                 print(f"WARNING: Multiple files found starting with '{script_id}' in {script_dir}. Using: {found_file_path}")

            # --- Loading Logic ---
            print(f"INFO: Loading script content from: {found_file_path}")
            if found_file_path.endswith(".txt"):
                with open(found_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif found_file_path.endswith(".docx"):
                try:
                    import docx # Local import to avoid making it a hard dependency if not used
                    doc = docx.Document(found_file_path)
                    full_text = [para.text for para in doc.paragraphs]
                    return '\n'.join(full_text)
                except ImportError:
                    print("ERROR: 'python-docx' library not installed. Cannot read .docx file. Please run: pip install python-docx")
                    return None
                except Exception as e:
                    print(f"ERROR: Failed to read .docx file {found_file_path}: {e}")
                    return None
            else:
                print(f"WARNING: Found file {found_file_path} has unsupported extension. Cannot load.")
                return None

        except Exception as e:
            print(f"ERROR: Failed to list/access/read script file in {script_dir} for {script_id}: {e}")
            return None

# --- Main Chatbot Logic ---

class TherapeuticChatbot:
    """Orchestrates the chatbot flow."""
    def __init__(self):
        self.llm_client = LLMClient(
            api_key=OPENROUTER_API_KEY,
            model_id=OPENROUTER_MODEL_ID,
        )
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

    def _build_condition_prompt(self, history_str: str) -> str | None:
        """Builds the prompt for the condition identification LLM call using ALL conditions."""
        all_descriptions = self.condition_manager.get_all_condition_descriptions()
        if not all_descriptions:
            print("ERROR: No condition descriptions found across any disorders.")
            return None

        # Sort descriptions for consistent prompt order
        all_descriptions.sort()
        conditions_list_str = "\n".join([f"- {desc}" for desc in all_descriptions])

        prompt = f"""
Analyze the following conversation history and determine if the user exhibits any of the listed conditions based *only* on the information present in the conversation.

# Conversation History
{history_str}

# Conditions to Evaluate (Across all potential areas):
{conditions_list_str}

# Instructions:
1. Evaluate each condition independently based on the *entire* conversation history provided.
2. Respond *only* in JSON format. Do not add any introductory text or explanations before or after the JSON object.
3. The JSON object must have:
    - Keys for *every* condition description listed above.
    - Values must be boolean (`true` or `false`).
    - A key named "follow-ups" with a value that is a JSON array of exactly 5 potential, relevant, open-ended follow-up questions the assistant could ask the user next, based on the conversation.

# Response (JSON only):
```json
{{
  "condition description 1 (from list above)": true/false,
  "condition description 2 (from list above)": true/false,
  ...
  "follow-ups": ["question 1", "question 2", "question 3", "question 4", "question 5"]
}}
```
"""
        return prompt.strip()

    def _parse_condition_response(self, response_str: str | None, session: ChatSession):
        """Parses the LLM's JSON response and updates the session state across disorders."""
        if not response_str:
            print("WARNING: Received empty response from condition identification LLM.")
            return

        # Clean potential markdown code fences
        if response_str.startswith("```json"):
            response_str = response_str[7:]
        if response_str.endswith("```"):
            response_str = response_str[:-3]
        response_str = response_str.strip()
        print(f"INFO: Parsing condition response (length: {len(response_str)})") # Avoid printing potentially huge responses

        try:
            data = json.loads(response_str)
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object.")

            updated_count = 0
            # Reset all conditions to False before update? Or just update based on response?
            # Let's just update based on response for now.
            # If a condition isn't in the response, it retains its previous state.

            # Update conditions
            for desc, value in data.items():
                if desc == "follow-ups": # Skip the follow-ups key here
                    continue

                # Find which disorder this description belongs to
                disorder_key, condition_key = self.condition_manager.get_disorder_and_key_from_description(desc)

                if disorder_key and condition_key:
                    if disorder_key in session.identified_conditions and condition_key in session.identified_conditions[disorder_key]:
                        if isinstance(value, bool):
                            # Update the specific condition within the specific disorder
                            session.identified_conditions[disorder_key][condition_key] = value
                            updated_count += 1
                        else:
                            print(f"WARNING: Invalid boolean value for condition '{desc}' (Disorder: {disorder_key}, Key: {condition_key}): {value}")
                    else:
                         print(f"WARNING: Condition '{desc}' (Disorder: {disorder_key}, Key: {condition_key}) found in response but not initialized in session state.")
                # else: # Don't warn if key not found, might be extra keys in response or mapping issue
                     # print(f"WARNING: Condition description '{desc}' from LLM response not found in known conditions map.")

            # Update follow-ups (remains the same)
            if "follow-ups" in data and isinstance(data["follow-ups"], list):
                session.suggested_follow_ups = [str(q) for q in data["follow-ups"]]
            else:
                print("WARNING: 'follow-ups' key missing or not a list in LLM response.")
                session.suggested_follow_ups = [] # Clear old ones if new ones are invalid

            print(f"INFO: Parsed condition response. Updated {updated_count} conditions across disorders.")
            # print(f"DEBUG: Session conditions after update: {json.dumps(session.identified_conditions, indent=2)}") # Debug

        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to decode JSON response for conditions: {e}\nResponse snippet:\n{response_str[:500]}...") # Print snippet
            session.suggested_follow_ups = []
        except Exception as e:
            print(f"ERROR: Failed to parse condition response: {e}\nResponse snippet:\n{response_str[:500]}...") # Print snippet
            session.suggested_follow_ups = []


    def _build_manual_response_prompt(self, session: ChatSession) -> str:
        """Builds the prompt for generating a manual, empathetic response."""
        history_str = session.format_history_for_prompt()
        follow_ups_str = "\n".join([f"- {q}" for q in session.suggested_follow_ups]) if session.suggested_follow_ups else "None available."

        prompt = f"""
You are a supportive, empathetic therapeutic assistant. Your goal is to respond to the user's last message in a warm, validating, and thoughtful way, following the provided guidelines.

# Conversation History:
{history_str}

# Your Guidelines:
{THERAPEUTIC_GUIDELINES}

# Potential Follow-up Questions (for inspiration, adapt as needed):
{follow_ups_str}

# Task:
Generate a supportive, empathetic response to the *last USER message* in the history. Your response should:
1. Validate their feelings and experiences
2. Demonstrate that you've truly heard and understood them
3. Create a safe, non-judgmental space for them to continue sharing
4. Include thoughtful reflection on what they've shared
5. End with a gentle, open-ended question that invites deeper exploration

Your response should be substantive (typically 3-5 sentences) to show genuine engagement, but not overwhelmingly long. Focus on emotional support rather than problem-solving or advice.

# Your Response:
"""
        return prompt.strip()

    def process_message(self, user_message: str, session_id: str = "default") -> str:
        """Processes a user message and returns the assistant's response."""
        # 1. Get/Create Session & Update History
        session = self._get_or_create_session(session_id)
        session.add_message("user", user_message)

        # --- No single disorder context needed here anymore ---

        # 2. Identify Conditions across ALL disorders (LLM Call 1)
        history_str = session.format_history_for_prompt()
        condition_prompt = self._build_condition_prompt(history_str) # No disorder key needed
        condition_response = None
        if condition_prompt:
            condition_response = self.llm_client.send_prompt(condition_prompt, temperature=0.3)
        self._parse_condition_response(condition_response, session) # No disorder key needed

        # 3. Check for Script Match across ALL disorders
        true_conditions_set = session.get_true_conditions_set()
        # find_matching_script now returns (script_id, disorder_key)
        matched_script_id, matched_disorder_key = self.condition_manager.find_matching_script(true_conditions_set)

        # 4. Response Decision Logic
        ai_response_content = None
        final_response_type = "manual" # Default

        # Check if we're in script offering mode from previous turn
        if session.offering_script:
            # Retrieve details from the stored offer
            offered_script_id = session.offering_script.get("script_id")
            offered_script_title = session.offering_script.get("script_title")
            offered_disorder_key = session.offering_script.get("disorder_key") # Get the disorder context of the offered script

            # Clear the offering state regardless of user's answer
            session.offering_script = None

            acceptance_keywords = ["yes", "yeah", "sure", "okay", "ok", "fine", "alright", "go ahead", "please", "let's try"]
            user_accepted = any(keyword in user_message.lower() for keyword in acceptance_keywords)

            if user_accepted and offered_script_id and offered_disorder_key:
                # User accepted, deliver the script using the stored disorder key
                script_content = self.condition_manager.get_script_content(offered_script_id, offered_disorder_key, SCRIPTS_DIR)
                if script_content:
                    lead_in = f"I'm glad you're open to trying this. Here's the '{offered_script_title}' exercise:\n\n---\n"
                    ai_response_content = f"{lead_in}SCRIPT_START\n{script_content}\nSCRIPT_END"
                    final_response_type = "script"
                    print(f"INFO: Delivering script '{offered_script_id}' (Disorder: {offered_disorder_key}) after user acceptance")
                else:
                    print(f"WARNING: Failed to load script '{offered_script_id}' (Disorder: {offered_disorder_key}) content after user acceptance")
                    ai_response_content = f"I apologize, but I'm having trouble retrieving the exercise I mentioned. Let's continue our conversation instead. How have you been feeling lately?"
                    final_response_type = "manual"
            else:
                # User declined or gave ambiguous response
                ai_response_content = f"That's completely fine. We can continue our conversation without the exercise. Let's focus on what you'd like to discuss today. How have you been feeling lately?"
                final_response_type = "manual"

        # Normal flow (not responding to script offer)
        elif matched_script_id and matched_disorder_key and session.message_count >= MIN_MESSAGES_FOR_SCRIPT:
            print(f"INFO: Condition match for script '{matched_script_id}' (Disorder: {matched_disorder_key}) and message count ({session.message_count}) threshold met.")

            # Offer the matched script
            script_title = self.condition_manager.get_script_title(matched_script_id, matched_disorder_key) or matched_script_id

            # Store the script info *and its disorder context* for next turn
            session.offering_script = {
                "script_id": matched_script_id,
                "script_title": script_title,
                "disorder_key": matched_disorder_key # Store the context
            }

            # Create a thoughtful offer message
            ai_response_content = f"Based on what you've shared, I think a guided exercise called '{script_title}' might be helpful for addressing some of the challenges you're experiencing. Would you like me to share this exercise with you now? It's completely up to you, and we can continue our conversation either way."
            final_response_type = "manual" # Offering is a manual-type response
            print(f"INFO: Offering script '{matched_script_id}' (Disorder: {matched_disorder_key}) to user")

        elif matched_script_id:
            # Script matched but message count too low
            print(f"INFO: Condition match for script '{matched_script_id}' (Disorder: {matched_disorder_key}) but message count ({session.message_count}) is less than {MIN_MESSAGES_FOR_SCRIPT}. Generating manual response.")
            final_response_type = "manual"
        else:
            # No script conditions met anywhere
            # print("INFO: No script conditions met across any disorder. Generating manual response.")
            final_response_type = "manual"

        # 5. Generate Manual Response if needed (LLM Call 2)
        if final_response_type == "manual" and not ai_response_content:
            # Manual prompt generation remains the same, using the general guidelines
            manual_prompt = self._build_manual_response_prompt(session)
            ai_response_content = self.llm_client.send_prompt(manual_prompt, temperature=0.8)
            if not ai_response_content:
                print("ERROR: Failed to generate manual response from LLM. Using fallback.")
                ai_response_content = "I understand. It sounds like a difficult situation. Could you tell me a little more about that?" # Generic fallback

        # 6. Update session history and return
        session.add_message("assistant", ai_response_content)
        # print(f"Assistant Response ({final_response_type}):\n{ai_response_content}") # Debug
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
