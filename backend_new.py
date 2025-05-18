import json
import os
import re
import time # For searching files
from openai import OpenAI # Use OpenAI library structure for OpenRouter
from dotenv import load_dotenv
from typing import List
from rag import fetch_guidance_notes
from script_loader import ScriptLoader

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
        self.identified_conditions: dict[str, dict[str, bool]] = {}
        self.suggested_follow_ups: list[str] = []
        self.script_message_count: int = 0
        self.offering_script: dict | None = None
        self.delivered_scripts: set[tuple[str, str]] = set()
        
        # New: Track which guidance note sections have been delivered
        self.delivered_guidance: dict[str, dict[str, bool]] = {}

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        self.script_message_count += 1

    def format_history_for_prompt(self) -> str:
        # only include user messages
        return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in self.conversation_history if m['role'] != 'assistant'])
    
    # New: Method to update guidance note delivery status
    def update_guidance_delivery(self, note_filename: str, primary_delivered: bool, secondary_delivered: bool):
        """Update which guidance components have been delivered"""
        if note_filename not in self.delivered_guidance:
            self.delivered_guidance[note_filename] = {"primary": False, "secondary": False}
        
        if primary_delivered:
            self.delivered_guidance[note_filename]["primary"] = True
        if secondary_delivered:
            self.delivered_guidance[note_filename]["secondary"] = True
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
    
    def _build_system_prompt(self, guidance_notes: str, json_output: bool = True) -> str:
        """Build system prompt with guidance notes and JSON output instructions if needed."""
        flag = True if guidance_notes else False
        
        prompt = f"""
            You are a supportive, empathetic therapist. Your goal is to respond to the user in a warm, validating, and thoughtful way.
    
            {"Helpful material you can use:" if flag else ""}
            {guidance_notes}
            """
        
        if json_output:
            prompt += """
            Important: Return your response in JSON format with these keys:
            {
                "Response": "Your therapeutic response here, addressing the user's concerns warmly and empathetically",
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
    def _analyze_content_delivery(self, response_text: str, session_id: str, symptoms: List[str],model=None) -> dict:
        """Analyze which guidance note content was delivered in the response."""
        session = self._get_or_create_session(session_id)
        
        # Fetch the same guidance notes again to analyze them
        # (This could be optimized by passing the notes directly)
        guidance_notes = fetch_guidance_notes(symptoms)
        
        llm = LLMClient(api_key=OPENROUTER_API_KEY, model_id=model)
        
        analysis_prompt = f"""
        Analyze if the following therapeutic response addressed the key points from the guidance notes.
        
        Response:
        {response_text}
        
        Guidance Notes:
        {guidance_notes}
        
        For each guidance note, determine if the primary content and secondary content were addressed.
        Return your analysis as a JSON object with the following format:
        
        {{
            "note_filename1": {{
                "primary_delivered": true/false,
                "secondary_delivered": true/false
            }},
            "note_filename2": {{
                "primary_delivered": true/false,
                "secondary_delivered": true/false
            }}
        }}
        """
        
        try:
            delivery_analysis = llm.send_prompt(analysis_prompt, extract_json=True)
            return delivery_analysis or {}
        except Exception as e:
            print(f"Error analyzing content delivery: {e}")
            return {}
        
    def _get_appropriate_script(self, symptoms: List[str], session: ChatSession,model=None) -> dict:
        """Select appropriate script based on symptoms and user profile."""
        # First extract user profile from conversation
        user_profile = self._extract_user_profile(session,model=model)
        
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
        print(f"Script ID: {script_id}, Match Score: {match_score}")
        print(f"Script Content: {script_content}")
        
        return {
            "script_id": script_id,
            "script_content": script_content,
            "metadata": matching_script,
            "match_score": match_score  # Include the score
        }
    def _extract_user_profile(self, session: ChatSession, model=None) -> dict:
        """Extract user age group, emotional intensity, and specific concerns."""
        conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in session.conversation_history])
        
        llm = LLMClient(api_key=OPENROUTER_API_KEY, model_id=model)
        
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
        
        # Extract symptoms
        keyword_prompt = self._build_keyword_identifier_prompt(history_str)
        keyword_response = llm.send_prompt(keyword_prompt, temperature=0.7, extract_json=True)
        symptoms = keyword_response.get("symptoms", [])
        
        # Fetch guidance notes with delivery status
        guidance_notes = fetch_guidance_notes(symptoms, session)
        
        # Build updated system prompt for JSON output
        system_prompt = self._build_system_prompt(guidance_notes=guidance_notes, json_output=True)
        
        messages = [
            {"role": "system", "content": system_prompt},
            *session.conversation_history
        ]
        
        # Request JSON response
        ai_response = llm.send_prompt(prompt=None, messages=messages, temperature=0.8, extract_json=True)
        print(f"AI Response: {ai_response}")
        
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
        
        # Analyze which content was delivered
        if guidance_notes:
            delivery_analysis = self._analyze_content_delivery(response_text, session_id, symptoms,model)
            
            # Update session with delivery status
            for note_file, status in delivery_analysis.items():
                session.update_guidance_delivery(
                    note_file,
                    status.get("primary_delivered", False),
                    status.get("secondary_delivered", False)
                )
        
        # Add assistant response to session history
        session.add_message("assistant", response_text)
        message_count = len(session.conversation_history)
    
        # Only offer scripts after a certain number of exchanges to build rapport first
        if activity_offered and message_count <= 2:  # Adjust this threshold as needed
            print(f"Script offering suppressed - too early in conversation (message count: {message_count})")
            activity_offered = False
        
        # Handle script/activity if offered
        if activity_offered:
            # Get script based on symptoms and user profile
            script_result = self._get_appropriate_script(symptoms, session, model)
            
            # ADDED: Check if script has a good enough score
            if script_result and script_result.get("match_score", 0) >= 10:  # Set minimum threshold
                return {
                    "response": response_text,
                    "script_id": script_result.get("script_id"),
                    "script_content": script_result.get("script_content"),
                    "script_offered": True,
                    "metadata": script_result.get("metadata", {})
                }
            else:
                print(f"Script rejected - match score too low: {script_result.get('match_score', 0) if script_result else 0}")
                # Return just the response without script
                return response_text
        
        # Return normal response if no activity offered or no script found
        return response_text