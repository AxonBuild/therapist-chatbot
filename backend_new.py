import json
import os
import re
import time # For searching files
from openai import OpenAI # Use OpenAI library structure for OpenRouter
from dotenv import load_dotenv
from typing import Iterable, List, Set, Dict, Any, Tuple
from clinical_cases_rag import fetch_clinical_cases, analyze_solution_delivery
from script_loader import ScriptLoader
import threading
load_dotenv()
# --- Configuration ---

# IMPORTANT: Set your OpenRouter API key as an environment variable
# or replace os.getenv("OPENROUTER_API_KEY") with your actual key string.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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
            print(f"LLM Response: {response}")
            
            if extract_json and response:
                response = self._extract_json(response)                
            return response
        except Exception as e:
            print(f"ERROR: LLM API call failed: {e}")
            return None

    def _extract_json(self, response: str):
        """Extract JSON from the response or wrap plain text in JSON format."""
        try:
            # First attempt: look for JSON inside code blocks
            code_blocks = re.findall(r'```(?:json)?([\s\S]*?)```', response)
            if code_blocks:
                for block in code_blocks:
                    try:
                        return json.loads(block.strip())
                    except:
                        continue
            
            # Second attempt: look for JSON enclosed in braces
            json_candidates = re.findall(r'({[\s\S]*?})', response)
            if json_candidates:
                for candidate in json_candidates:
                    try:
                        return json.loads(candidate)
                    except:
                        continue
            
            # Third attempt: try the entire response as JSON
            return json.loads(response)
        except:
            # If all extraction attempts fail, wrap the response in our expected format
            print(f"Failed to extract json from:\n{response}\n\n-------")
            
            # Instead of raising an error, wrap the response in the expected JSON format
            return {
                "Response": response,
                "Activity_offered": False  # Default to false when parsing fails
            }


# --- Session State ---

class ChatSession:
    """Stores the state of a single user conversation."""
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.conversation_history: list[dict] = []
        
        #Track which guidance note sections have been delivered
        #self.delivered_guidance: dict[str, dict[str, bool]] = {}

        # New: Clinical case tracking
        self.matched_limiting_beliefs: Dict[str, Dict[str, Any]] = {}  # All matched beliefs over time
        self.message_belief_history: List[Dict[str, Any]] = []  # Track which message had which beliefs
        self.delivered_solutions: dict[str, dict] = {}  # Maps case_id to solution delivery status
        self.current_cases: set[str] = set() # Track the currently chosen cases
        self.completed_cases: set[str] = set()  # Track which cases have been completed

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def format_history_for_prompt(self) -> str:
        # Only include user messages
        return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in self.conversation_history])
    
    # def update_guidance_delivery(self, note_filename: str, primary_delivered: bool, secondary_delivered: bool):
    #     """Legacy: Update which guidance components have been delivered"""
    #     if note_filename not in self.delivered_guidance:
    #         self.delivered_guidance[note_filename] = {"primary": False, "secondary": False}
        
    #     if primary_delivered:
    #         self.delivered_guidance[note_filename]["primary"] = True
    #     if secondary_delivered:
    #         self.delivered_guidance[note_filename]["secondary"] = True
            
    def update_limiting_belief_matches(self, belief_matches: Dict[str, Dict[str, Any]], message_index: int):
            """Track limiting beliefs with message context and accumulation"""
            current_message_beliefs = []
            
            for belief, info in belief_matches.items():
                # Add to overall tracking (don't overwrite)
                if belief not in self.matched_limiting_beliefs:
                    self.matched_limiting_beliefs[belief] = info
                    self.matched_limiting_beliefs[belief]["first_detected"] = message_index
                    self.matched_limiting_beliefs[belief]["occurrences"] = 1
                else:
                    # Increment occurrence count
                    self.matched_limiting_beliefs[belief]["occurrences"] += 1
                    self.matched_limiting_beliefs[belief]["last_detected"] = message_index
                
                current_message_beliefs.append(belief)
            
            # Track which beliefs appeared in this message
            self.message_belief_history.append({
                "message_index": message_index,
                "beliefs": current_message_beliefs
            })
    def get_all_limiting_beliefs_context(self) -> str:
        """Get formatted context of all limiting beliefs detected"""
        if not self.matched_limiting_beliefs:
            return ""
        
        context = "Previously identified limiting belief patterns:\n"
        for belief, info in self.matched_limiting_beliefs.items():
            occurrences = info.get("occurrences", 1)
            context += f"- \"{belief}\" (mentioned {occurrences} time{'s' if occurrences > 1 else ''})\n"
        
        return context
    
    def update_solution_delivery(self, case_id: str, solution_status: Dict[str, bool]):
        """Update which solutions have been delivered for a case"""
        # Update status
        print(f"\nUpdating solution delivery for case {case_id}: {solution_status}")
        print(f"Current delivery status: {self.delivered_solutions.get(case_id, {})}")
        
        for solution_type, delivered in solution_status.items():
            if self.delivered_solutions[case_id][solution_type] == True:
                print(f"Solution type '{solution_type}' for case {case_id} already delivered. Skipping update to: {delivered}")
                continue
            if delivered:
                self.delivered_solutions[case_id][solution_type] = True
        
        print(f"Updated delivery status: {self.delivered_solutions[case_id]}")
        print("Checking if all solutions delivered for case:", all(self.delivered_solutions[case_id].values()))
        
        if all(self.delivered_solutions[case_id].values()):
            print(f"Case {case_id} delivery completed!")
            self.completed_cases.add(case_id)
            self.current_cases.remove(case_id)
    
    def add_current_cases(self, case_ids: Iterable[str]):
        for case in case_ids:
            self.current_cases.add(case)
            self.delivered_solutions[case] = {
                "immediate": False,
                "intermediate": False,
                "long_term": False
            }

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
    def _analyze_solutions_async(self, response_text: str, matched_cases: Set[str], llm: 'LLMClient', session):
            """Analyze solution delivery in a separate thread to avoid blocking the main response."""
            try:
                if matched_cases:
                    print("Starting async solution delivery analysis...")
                    delivery_analysis = analyze_solution_delivery(response_text, matched_cases, llm)
                    
                    # Ensure delivery_analysis is a dictionary
                    if isinstance(delivery_analysis, dict):
                        # Update session with solution delivery status
                        for case_id, status in delivery_analysis.items():
                            if isinstance(status, dict):
                                session.update_solution_delivery(case_id, status)
                            else:
                                print(f"Warning: Invalid status format for case {case_id}: {status}")
                    else:
                        print(f"Warning: delivery_analysis is not a dict: {type(delivery_analysis)}")
                        
                    print("Completed async solution delivery analysis")
            except Exception as e:
                print(f"Error in async solution delivery analysis: {e}")
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
    
   
    def _build_system_prompt(self, clinical_guidance: str, json_output: bool = True, custom_template: dict = None) -> str:
        """Build system prompt with clinical case guidance and JSON output instructions."""
        has_guidance = True if clinical_guidance else False
        
        # Use custom template if provided, otherwise use default
        if custom_template:
            prompt = self._build_custom_prompt(clinical_guidance, custom_template)
        else:
            # Default prompt (existing logic)
            prompt = f"""You are a supportive, empathetic therapist. Your goal is to respond to the user in a warm, validating, and thoughtful way.

{"Here are clinical patterns and therapeutic approaches that may help with this conversation:" if has_guidance else ""}
{clinical_guidance}

"""
        
        if json_output:
            prompt += """
Important: Return your response in JSON format with these keys:
{
    "Response": "Your response to the user",
    "Activity_offered": true/false (Set to true when appropriate to offer a guided exercise otherwise false)
}

For the "Activity_offered" field:
- Set to TRUE ONLY when all of these conditions are met:
    1. The conversation has established good rapport
    2. You have a clear understanding of the user's specific issue
    3. A structured activity would be therapeutic at this moment
    4. You've already provided some initial validation and support
- Otherwise, set to FALSE

DO NOT offer an activity in the first few exchanges. Focus on understanding and validating the user first.
            """
        
        return prompt
    
    def _build_custom_prompt(self, clinical_guidance: str, custom_template: dict) -> str:
        """Build prompt using custom template values."""
        try:
            # Load template file
            template_path = os.path.join("templates", "system_prompt_template.txt")
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Replace placeholders
            therapist_style = custom_template.get("therapist_style", "supportive, empathetic")
            response_tone = custom_template.get("response_tone", "warm, validating, and thoughtful")
            
            # Build clinical guidance section
            has_guidance = True if clinical_guidance else False
            clinical_section = ""
            if has_guidance:
                clinical_section = f"Here are clinical patterns and therapeutic approaches that may help with this conversation:\n{clinical_guidance}\n"
            
            # Replace placeholders in template
            prompt = template.format(
                therapist_style=therapist_style,
                response_tone=response_tone,
                clinical_guidance_placeholder=clinical_section
            )
            
            return prompt
            
        except Exception as e:
            print(f"Error loading custom template: {e}")
            # Fallback to default
            return f"""You are a supportive, empathetic therapist. Your goal is to respond to the user in a warm, validating, and thoughtful way.

{"Here are clinical patterns and therapeutic approaches that may help with this conversation:" if clinical_guidance else ""}
{clinical_guidance}

"""


    def _get_appropriate_script(self, symptoms: List[str], session: ChatSession, model=None) -> dict:
        """Select appropriate script based on symptoms and user profile."""
        # First extract user profile from conversation
        user_profile = self._extract_user_profile(session, model=model)
        
        # Then find matching script
        script_result = self._find_matching_script(symptoms, user_profile)
        
        return script_result
    
    def _find_matching_script(self, symptoms: List[str], user_profile: dict) -> dict:
        """Find matching script based on symptoms and user profile."""
        script_loader = ScriptLoader()
        
        # Find the best matching script
        matching_script, match_score = script_loader.find_matching_script(symptoms, user_profile)
        
        if not matching_script:
            print("No matching script found")
            return None
            
        script_id = matching_script.get('Script ID', matching_script.get('Filename', ''))
        
        # Get the script content
        script_content = script_loader.get_script_content(script_id)
        
        return {
            "script_id": script_id,
            "script_content": script_content,
            "metadata": matching_script,
            "match_score": match_score  # Include the score
        }
        
    def _extract_user_profile(self, session: ChatSession, model=None) -> dict:
        """Extract user age group, emotional intensity, and specific concerns."""
        conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in session.conversation_history])
        
        llm = LLMClient(api_key=OPENROUTER_API_KEY, model_id=model or "openai/gpt-4o-mini")
        
        profile_prompt = f"""
Based on this conversation, analyze the user's profile and emotional state.

Conversation:
{conversation}

Determine:
1. The user's likely age group (child/teen/adult/senior)
2. Their emotional intensity level (mild/moderate/intense)
3. Top 3 specific therapeutic concerns (like "grief release", "self-worth", "body awareness", etc.)

Return as JSON:
{{
    "age_group": "adult|teen|child|senior",
    "emotional_intensity": "mild|moderate|intense",
    "specific_concerns": ["concern1", "concern2", "concern3"]
}}
        """
        
        try:
            profile = llm.send_prompt(profile_prompt, extract_json=True)
            if profile:
                return profile
        except Exception as e:
            print(f"Error extracting user profile: {e}")
        
        # Default fallback
        return {
            "age_group": "adult", 
            "emotional_intensity": "moderate",
            "specific_concerns": []
        }

    # @compute_time
    # def process_message(self, user_message: str, session_id: str = "default", model=None, custom_template: dict = None) -> dict | str:
    #     """Processes a user message and returns the assistant's response."""
    #     # Initialize LLM client and model
    #     model = model or "openai/gpt-4o-mini"
    #     print("Using model:", model)
    #     llm = LLMClient(api_key=OPENROUTER_API_KEY, model_id=model)
        
    #     # Get/create session and update history
    #     session = self._get_or_create_session(session_id)
    #     session.add_message("user", user_message)

    #     # Extract conversation history for keyword identification
    #     history_str = session.format_history_for_prompt()
        
    #     # Extract symptoms from the conversation
    #     if len(session.current_cases) < 3:
    #         keyword_prompt = self._build_keyword_identifier_prompt(history_str)
    #         keyword_response = llm.send_prompt(keyword_prompt, temperature=0.7, extract_json=True)
    #         symptoms = keyword_response.get("symptoms", [])
    #     else:
    #         symptoms = []
        
    #     # Fetch clinical cases based on symptoms and latest message
    #     clinical_guidance = fetch_clinical_cases(
    #         symptoms=symptoms, 
    #         user_message=user_message,
    #         session=session
    #     )
        
    #     print(f"Matched clinical cases: {session.current_cases}")
    #     print(f"Clinical guidance: {clinical_guidance}")
        
    #     # Build system prompt with clinical guidance
    #     system_prompt = self._build_system_prompt(
    #         clinical_guidance=clinical_guidance, 
    #         json_output=True,
    #         custom_template=custom_template
    #     )        
    #     print(f"System prompt: {system_prompt}")
    #     # Construct messages for LLM
    #     messages = [
    #         {"role": "system", "content": system_prompt},
    #         *session.conversation_history
    #     ]
        
    #     # Get response from LLM
    #     ai_response = llm.send_prompt(prompt=None, messages=messages, temperature=0.8, extract_json=True)
    #     print(f"AI Response: {ai_response}")
        
    #     # Handle potential failure
    #     if not ai_response:
    #         print("ERROR: Failed to generate response from LLM. Using fallback.")
    #         response_text = "I understand. It sounds like a difficult situation. Could you tell me a little more about that?"
    #         activity_offered = False
    #     else:
    #         # Extract data from JSON response
    #         try:
    #             if isinstance(ai_response, str):
    #                 ai_response = json.loads(ai_response)
                    
    #             response_text = ai_response.get("Response", "")
    #             activity_offered = ai_response.get("Activity_offered", False)
    #         except Exception as e:
    #             print(f"Error parsing LLM response as JSON: {e}")
    #             response_text = ai_response if isinstance(ai_response, str) else "I understand. Could you tell me more?"
    #             activity_offered = False
        
    #     # Analyze which solutions were delivered
    #     if len(session.current_cases) > 0:
    #         delivery_analysis = analyze_solution_delivery(response_text, session.current_cases, llm)
            
    #         # Update session with solution delivery status
    #         for case_id, status in delivery_analysis.items():
    #             session.update_solution_delivery(case_id, status)
        
    #     # Add assistant response to session history
    #     session.add_message("assistant", response_text)
        
    #     # Check message count for activity offering
    #     message_count = len(session.conversation_history) // 2  # Count exchanges, not individual messages
    #     if activity_offered and message_count <= 2:
    #         print(f"Script offering suppressed - too early in conversation (message count: {message_count})")
    #         activity_offered = False
        
    #     # Handle script/activity if offered
    #     if activity_offered:
    #         # Get script based on symptoms and user profile
    #         script_result = self._get_appropriate_script(symptoms, session, model)
            
    #         # Check if script has a good enough score
    #         if script_result and script_result.get("match_score", 0) >= 10:  # Minimum threshold
    #             return {
    #                 "response": response_text,
    #                 "script_id": script_result.get("script_id"),
    #                 "script_content": script_result.get("script_content"),
    #                 "script_offered": True,
    #                 "metadata": script_result.get("metadata", {})
    #             }
    #         else:
    #             print(f"Script rejected - match score too low: {script_result.get('match_score', 0) if script_result else 0}")
                
    #     # Return normal response if no script offered or match score too low
    #     return response_text
    @compute_time
    def process_message(self, user_message: str, session_id: str = "default", model=None, custom_template: dict = None) -> dict | str:
        """Processes a user message and returns the assistant's response."""
        # Initialize LLM client and model
        model = model or "openai/gpt-4o-mini"
        print("Using model:", model)
        llm = LLMClient(api_key=OPENROUTER_API_KEY, model_id=model)
        
        # Get/create session and update history
        session = self._get_or_create_session(session_id)
        session.add_message("user", user_message)

        # Extract conversation history for keyword identification
        history_str = session.format_history_for_prompt()
        
        # Extract symptoms from the conversation
        if len(session.current_cases) < 3:
            keyword_prompt = self._build_keyword_identifier_prompt(history_str)
            keyword_response = llm.send_prompt(keyword_prompt, temperature=0.7, extract_json=True)
            symptoms = keyword_response.get("symptoms", [])
        else:
            symptoms = []
        
        # Fetch clinical cases based on symptoms and latest message
        clinical_guidance, matched_cases = fetch_clinical_cases(
            symptoms=symptoms, 
            user_message=user_message,
            session=session
        )
        
        print(f"Matched clinical cases: {session.current_cases}")
        print(f"Clinical guidance: {clinical_guidance}")
        
        # Build system prompt with clinical guidance
        system_prompt = self._build_system_prompt(clinical_guidance=clinical_guidance, json_output=True)
        
        # Construct messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            *session.conversation_history
        ]
        
        # Get response from LLM
        ai_response = llm.send_prompt(prompt=None, messages=messages, temperature=0.8, extract_json=True)
        print(f"AI Response: {ai_response}")
        
        # Handle potential failure
        if not ai_response:
            print("ERROR: Failed to generate response from LLM. Using fallback.")
            response_text = "I understand. It sounds like a difficult situation. Could you tell me a little more about that?"
            activity_offered = False
        else:
            # Extract data from JSON response
            try:
                if isinstance(ai_response, str):
                    ai_response = json.loads(ai_response)
                    
                response_text = ai_response.get("Response", "")
                activity_offered = ai_response.get("Activity_offered", False)
            except Exception as e:
                print(f"Error parsing LLM response as JSON: {e}")
                response_text = ai_response if isinstance(ai_response, str) else "I understand. Could you tell me more?"
                activity_offered = False
        
        # Add assistant response to session history
        session.add_message("assistant", response_text)
        
        # Start solution delivery analysis in background thread (non-blocking)
        if matched_cases:
            analysis_thread = threading.Thread(
                target=self._analyze_solutions_async,
                args=(response_text, matched_cases.copy(), llm, session),
                daemon=True  # Thread will die when main program exits
            )
            analysis_thread.start()
            print("Started solution delivery analysis in background thread")
        
        # Check message count for activity offering
        message_count = len(session.conversation_history) // 2  # Count exchanges, not individual messages
        if activity_offered and message_count <= 2:
            print(f"Script offering suppressed - too early in conversation (message count: {message_count})")
            activity_offered = False
        
        # Handle script/activity if offered
        if activity_offered:
            # Get script based on symptoms and user profile
            script_result = self._get_appropriate_script(symptoms, session, model)
            
            # Check if script has a good enough score
            if script_result and script_result.get("match_score", 0) >= 10:  # Minimum threshold
                return {
                    "response": response_text,
                    "script_id": script_result.get("script_id"),
                    "script_content": script_result.get("script_content"),
                    "script_offered": True,
                    "metadata": script_result.get("metadata", {})
                }
            else:
                print(f"Script rejected - match score too low: {script_result.get('match_score', 0) if script_result else 0}")
                
        # Return normal response if no script offered or match score too low
        return response_text