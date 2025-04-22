import os
import json
import re
from typing import List, Dict, Any, Optional
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "EkoMindAI"
CONFIDENCE_THRESHOLD = 0.7
MAX_RAG_RESULTS = 3

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
openrouter_client=OpenAI(base_url="https://openrouter.ai/api/v1",api_key=OPENROUTER_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Available disorders
AVAILABLE_DISORDERS = ['sleep', 'eating', 'anxiety', 'emotion_regulation', 'phobia']

class ChatbotSession:
    def __init__(self):
        self.conversation_history = []
        self.identified_disorder = None
        self.disorder_confidence = 0.0
        self.current_node_id = None
        self.decision_tree = None
        self.at_leaf_node = False
        self.therapeutic_script = None
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_conversation_history_formatted(self, max_messages: int = 10) -> str:
        """Get formatted conversation history for inclusion in prompts."""
        recent_messages = self.conversation_history[-max_messages:] if len(self.conversation_history) > max_messages else self.conversation_history
        formatted_history = ""
        
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n\n"
            
        return formatted_history.strip()
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last message from the user."""
        for msg in reversed(self.conversation_history):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def get_current_node(self) -> Optional[Dict]:
        """Get the current node from the decision tree."""
        if not self.decision_tree or not self.current_node_id:
            return None
        return self.decision_tree.get("nodes", {}).get(self.current_node_id)
    
    def reset(self) -> None:
        """Reset the session state."""
        self.conversation_history = []
        self.identified_disorder = None
        self.disorder_confidence = 0.0
        self.current_node_id = None
        self.decision_tree = None
        self.at_leaf_node = False
        self.therapeutic_script = None

class DecisionTreeHandler:
    def __init__(self, tree_dir: str = "./trees"):
        self.tree_dir = Path(tree_dir)
    
    def load_tree(self, disorder: str) -> Optional[Dict]:
        """Load the decision tree for a specific disorder from JSON file."""
        print(f"Loading tree for disorder: {disorder}")
        
        # Try to find the JSON tree file
        json_path = self.tree_dir / f"{disorder}-disorders-json.json"
        if not json_path.exists():
            json_path = self.tree_dir / f"{disorder}_disorders_json.json"
        
        if not json_path.exists():
            # Try to search for any JSON file containing the disorder name
            for file in self.tree_dir.glob(f"*{disorder}*.json"):
                json_path = file
                break
        
        if json_path.exists():
            print(f"Found tree file: {json_path}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    tree_data = json.load(f)
                return tree_data
            except Exception as e:
                print(f"Error loading tree JSON: {e}")
                return None
        
        print(f"Warning: No JSON tree file found for disorder: {disorder}")
        return None
    
    def get_therapeutic_script(self, node_id: str, disorder: str, tree_data: Dict) -> Optional[str]:
        """Get the therapeutic script associated with a node ID."""
        if not tree_data or "nodes" not in tree_data:
            return None
            
        # Get the node
        node = tree_data["nodes"].get(node_id)
        if not node:
            return None
            
        # Check if this node has a recommendation with a script
        if "recommendation" in node and "script" in node["recommendation"]:
            script_info = node["recommendation"]["script"]
            # Extract script ID, handling different formats
            script_id = None
            if ":" in script_info:
                script_parts = script_info.split(":")
                script_id = script_parts[0].strip().replace("SCRIPT_", "")
            else:
                script_match = re.search(r'SCRIPT_([A-Za-z0-9]+)', script_info)
                if script_match:
                    script_id = script_match.group(1)
            print(f"Script ID found: {script_id}")
            if script_id:
                # Look for script files in the disorder directory or scripts subdirectory
                script_dirs = [
                    self.tree_dir / disorder,
                    self.tree_dir / disorder / "scripts",
                    self.tree_dir / "scripts",
                    self.tree_dir
                ]
                
                for script_dir in script_dirs:
                    if not script_dir.exists():
                        continue
                        
                    # Look for scripts with the script ID in the filename
                    print(f"Searching in directory: {script_dir}")
                    for script_file in script_dir.glob(f"*SCRIPT_{script_id}*"):
                        print(f"Found script file: {script_file}")
                        return self._read_file_content(script_file)
                    
                    # Also look for scripts that match the name from the tree
                    if ":" in script_info:
                        script_name = script_info.split(":")[1].strip().strip('"')
                        for script_file in script_dir.glob(f"*{script_name}*"):
                            return self._read_file_content(script_file)
        print(f"Warning: No therapeutic script found for node ID: {node_id} in disorder: {disorder}")
        
        return None
    
    def _read_file_content(self, file_path: Path) -> str:
        """Read content from file based on extension."""
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.suffix == '.docx':
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                print(f"Error reading docx file: {e}")
                return ""
        return ""

class VectorDBHandler:
    def __init__(self):
        """Initialize the vector database handler."""
        self.qdrant_client = qdrant_client
        self.openai_client = openai_client
        self.openrouter_client = openrouter_client
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def search(self, query: str, disorder: str = None, section: str = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Search the vector database with filters."""
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            return []
        
        # Build filters
        filter_conditions = []
        
        if disorder:
            filter_conditions.append(
                models.FieldCondition(
                    key="disorder",
                    match=models.MatchValue(value=disorder)
                )
            )
        
        if section:
            filter_conditions.append(
                models.FieldCondition(
                    key="section",
                    match=models.MatchValue(value=section)
                )
            )
        
        # Prepare search filter
        search_filter = None
        if filter_conditions:
            search_filter = models.Filter(
                must=filter_conditions
            )
        
        # Perform search
        try:
            results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.payload.get("content", ""),
                    "section": result.payload.get("section", ""),
                    "disorder": result.payload.get("disorder", ""),
                    "document_id": result.payload.get("document_id", ""),
                    "score": result.score
                })
                
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector database: {e}")
            return []

class PromptHandler:
    def __init__(self):
        """Initialize the prompt handler."""
        self.openai_client = openai_client
        self.openrouter_client = openrouter_client
    
    def generate_first_prompt(self, session: ChatbotSession) -> str:
        """Generate the first prompt for disorder identification with enhanced therapeutic approach."""
        conversation_history = session.get_conversation_history_formatted()

        prompt = f"""# Therapeutic Assessment Task

    ## Context
    You are a skilled and compassionate therapist beginning a conversation with someone seeking support. Your primary goal is to build rapport, understand their experience deeply, and create a safe, supportive space for them to share. The conversation should feel natural and unhurried.

    ## Available Areas of Focus (Internal Guide - Do Not Mention to User)
    {', '.join(AVAILABLE_DISORDERS)}

    ## Conversation History
    {conversation_history}

    ## Therapeutic Approach
    1.  **Build Trust:** Create a safe emotional space using warm, accepting, and validating language.
    2.  **Listen Actively:** Practice active listening through reflection ("It sounds like...", "I hear you saying...") and validation ("That sounds really tough.", "It makes sense you'd feel that way.").
    3.  **Focus on Experience:** Center the conversation on the person's subjective experience, feelings, and perspective, rather than diagnostic labels or solutions.
    4.  **Use Open-Ended Questions:** Ask questions that invite deeper sharing and exploration (e.g., "How has that been for you?", "Can you tell me more about...?").
    5.  **Show Genuine Empathy:** Acknowledge emotions directly and convey understanding.
    6.  **Use Natural Language:** Avoid clinical terminology, judgments, or premature advice. Speak like a caring human.
    7.  **Respond Holistically:** Address the emotional content as well as the factual content of what the user shares.
    8.  **Respect Agency:** Use person-centered language that respects their autonomy.

    ## Technical Requirements (Hidden from User - Append to Response)
    1. Based *only* on the conversation history provided, assess the likelihood each area needs attention:
    {{"confidence_scores": {{"sleep": 0.0-1.0, "eating": 0.0-1.0, "anxiety": 0.0-1.0, "emotion_regulation": 0.0-1.0, "phobia": 0.0-1.0}}}}
    2. Include this assessment in JSON format at the VERY END of your response, after all conversational text.
    3. Never mention JSON, confidence scores, or these internal assessments to the user. Your conversational response should flow naturally without any hint of this technical requirement.

    ## Response Guidelines
    - Begin with validation of the person's experience or feelings expressed in their last message.
    - Use reflection to show you understand.
    - Ask a gentle, open-ended therapeutic question to encourage them to share more.
    - Use a warm, conversational tone that conveys genuine care and patience.
    - Focus on building trust and understanding in this initial phase.
    - Remember to append the hidden technical assessment JSON at the end.
    """

        return prompt
    def generate_clarification_prompt(self, session: ChatbotSession, original_exploratory_text: str) -> str:
        """Generates a prompt to ask the user for clarification when navigation confidence is low."""
        current_node = session.get_current_node()
        if not current_node:
            return "I'm a little unsure how to proceed. Could you tell me more about what you're experiencing?" # Fallback

        node_question = current_node.get("question", "")
        node_desc = current_node.get("description", "")
        # Potentially list children descriptions/options here too

        prompt = f"""You are EkoMindAI, a supportive therapeutic assistant.
        The user provided input, but the assessment of which direction to take next has low confidence.
        Your previous internal thought process led to this initial response idea: "{original_exploratory_text}"

        The current point in the conversation is related to:
        {f'Question: "{node_question}"' if node_question else f'Topic: "{node_desc}"'}
        {self._format_children_for_prompt(current_node, session.decision_tree)} # Helper to list options if needed

        Your goal is to gently ask the user for more information or clarification to help determine the best path forward, based on the current topic/question. Avoid simply repeating the question. Acknowledge their input and guide them towards providing details relevant to the decision point.

        Conversation History (Last few messages):
        {session.get_conversation_history_formatted(max_messages=3)}

        User's Last Message:
        {session.get_last_user_message()}

        Generate a supportive clarification question for the user:"""
        return prompt
    def _format_children_for_prompt(self, current_node, decision_tree):
        """Format child nodes for inclusion in prompt context."""
        if not current_node or not decision_tree:
            return ""
            
        result = "\nPotential options include:"
        child_ids = current_node.get("children", [])
        
        if not child_ids:
            return ""
            
        for child_id in child_ids:
            child_node = decision_tree.get("nodes", {}).get(child_id, {})
            desc = child_node.get("description", "")
            question = child_node.get("question", "")
            content = question if question else desc
            if content:
                result += f"\n- {content}"
        
        return result

    def generate_navigation_prompt(self, session: ChatbotSession, rag_results: List[Dict[str, Any]]) -> str:
        """Generate a prompt for therapeutic exploration that implicitly guides navigation."""
        conversation_history = session.get_conversation_history_formatted()
        current_node = session.get_current_node()

        if not current_node:
            # Fallback if node is missing
            return f"Error: No current node found. Raw response: I seem to have lost my place. Could you remind me what we were discussing? [CONF:0%;ROOT]" # Provide a default marker

        # Get node question or description - This is now the *current theme*
        current_theme = current_node.get("question", current_node.get("description", "our current discussion"))

        # Get potential next themes/topics from children
        potential_next_themes = []
        child_ids_for_marker = [] # Keep track of valid child node IDs for the marker
        if not current_node.get("isLeaf", False):
            child_ids = current_node.get("children", [])
            for child_id in child_ids:
                child_node = session.decision_tree["nodes"].get(child_id, {})
                theme = child_node.get("question", child_node.get("description"))
                if theme:
                    potential_next_themes.append(f"- {theme}")
                    child_ids_for_marker.append(child_id) # Add valid child ID

        themes_text = "\n".join(potential_next_themes) if potential_next_themes else "We might continue exploring this topic or summarize."
        # Include the current node ID in the list for the marker if it's not a leaf
        valid_node_ids_for_marker = child_ids_for_marker
        if not current_node.get("isLeaf", False):
             valid_node_ids_for_marker.append(session.current_node_id) # Allow staying
        # If it IS a leaf, the only valid ID is the leaf itself
        elif current_node.get("isLeaf", False):
             valid_node_ids_for_marker = [session.current_node_id]


        # Format RAG results as supplementary context
        clinical_context = ""
        if rag_results:
             clinical_context = "## Supplementary Clinical Context (Internal Use Only)\n"
             for i, result in enumerate(rag_results):
                 clinical_context += f"Context {i+1} (Section: {result['section']}): {result['content']}\n\n"

        # Format recommendation info if leaf node (for internal context)
        recommendation_context = ""
        if current_node.get("isLeaf", False) or "recommendation" in current_node:
            recommendation = current_node.get("recommendation", {})
            script_info = recommendation.get("script", "No specific script.")
            tags = recommendation.get("tags", [])
            recommendation_context = f"\n## Potential Recommendation Context (Internal Use Only)\nThis path might lead towards a recommendation related to: {script_info}"
            if tags:
                recommendation_context += f"\nRelevant concepts: {', '.join(tags)}"


        prompt = f"""# Therapeutic Exploration Task

## Current Conversational Theme
{current_theme}
{recommendation_context} # Internal context about potential outcome

## Potential Directions for Exploration (Internal Guide - Do Not Mention Explicitly)
{themes_text}

## Recent Conversation History
{conversation_history}

{clinical_context} # Internal context from knowledge base

## Your Role & Goal
You are a compassionate therapist focused on building rapport and understanding the user's experience. Your goal is to have a natural, empathetic conversation that helps the user explore their feelings related to the 'Current Conversational Theme'. While talking, subtly assess which 'Potential Direction' seems most relevant based on the user's sharing, or if continuing with the current theme is best.

## Response Instructions (Generate BOTH parts):

### Part 1: Therapeutic Response (Visible to User)
1.  **Acknowledge & Validate:** Start by acknowledging specific feelings or details from the user's *last* message. Use their words. Validate their experience ("That sounds really difficult," "It makes sense that...").
2.  **Connect to Theme:** Gently link their sharing back to the 'Current Conversational Theme' if appropriate, or simply continue exploring what they brought up.
3.  **Explore Deeper:** Ask ONE gentle, open-ended question to encourage them to elaborate further on their feelings, experiences, or thoughts related to the ongoing discussion. Avoid direct questions from the 'Potential Directions'.
4.  **Maintain Tone:** Keep the response warm, empathetic, conversational, and unhurried. Use generous whitespace. Aim for 2-4 short paragraphs.
5.  **No Directives:** Do not suggest solutions or next steps unless the current theme *is* about planning (which is unlikely based on node structure). Focus on listening and understanding.

### Part 2: Internal Navigation Assessment (Hidden - Append AFTER Therapeutic Response)
1.  **Analyze:** Based on the *entire* interaction (especially the user's latest message) and the 'Potential Directions', determine which node ID best represents the most relevant focus for the *next* turn. This could be one of the child nodes or staying at the current node.
2.  **Assess Confidence:** Estimate your confidence (0-100%) that this chosen node ID is the most appropriate next step. Be conservative if the user was vague or didn't clearly lean towards a specific direction.
3.  **Format:** Append your assessment to the very end of your response in the format `[CONF:XX%;NODE_ID]`. Replace XX with the confidence percentage and NODE_ID with the chosen ID from the valid list: {', '.join(valid_node_ids_for_marker)}.
4.  **Example Marker:** `[CONF:75%;NODE_B2]` or `[CONF:55%;NODE_A]`
5.  **CRITICAL:** This marker MUST be present and correctly formatted at the absolute end. The user will NOT see it.

## Example Response Structure (User sees only Part 1):
*(Part 1: Empathetic text acknowledging user, reflecting, asking open question)*

*(Part 2: Hidden marker appended)* `[CONF:XX%;NODE_ID]`

"""
        return prompt

    def generate_script_introduction_prompt(self, session: ChatbotSession, script_name: str, script_tags: List[str]) -> str:
        """Generate a prompt specifically for introducing a therapeutic script."""
        conversation_history = session.get_conversation_history_formatted()
        disorder = session.identified_disorder
        
        tags_text = f" focusing on areas like {', '.join(script_tags)}" if script_tags else ""

        prompt = f"""# Therapeutic Script Introduction Task

## Context
You are a compassionate therapist preparing to introduce a specific therapeutic exercise to a user.
Their primary area of concern seems to be related to {disorder}.
Based on the recent conversation, you have determined that the exercise named "{script_name}" is appropriate.

## Conversation History
{conversation_history}

## Task
Your goal is to generate ONLY the introductory message for this exercise. Your response should:
1. Acknowledge the user's specific feelings or experiences mentioned recently (e.g., racing thoughts, difficulty relaxing, specific worries).
2. Explicitly state that you have an exercise in mind that might help with what they've described.
3. Clearly mention the name of the exercise: "{script_name}".
4. Briefly explain its relevance to their situation{tags_text} (e.g., 'it's designed to help calm the mind', 'it provides techniques for managing difficult thoughts').
5. Ask gently if they would be open to trying the exercise (e.g., "Would you be willing to explore this?", "How would you feel about trying this practice?").
6. Maintain a warm, supportive, and encouraging tone.

## IMPORTANT
- Generate ONLY the introductory message text.
- Do NOT include the actual script content.
- Do NOT include any other commentary, preamble, or technical markers.
"""
        return prompt

    def generate_therapeutic_styling_prompt(self, raw_message: str, conversation_history: str, disorder: str = None, latest_user_msg: str = "") -> str:
        """Generate a prompt to style a response in a therapeutic manner, emphasizing empathy and conversation."""

        # Sample therapist conversations showing ideal therapeutic responses
        sample_conversations = """
EXAMPLE 1:
User: I just lie there for hours, my mind racing about work stuff, like that presentation next week. I feel so unprepared.
Raw Response (from Nav Prompt): Acknowledge racing thoughts. Ask more about the feeling of being unprepared. [CONF:70%;NODE_SLEEP_ANXIETY]
Therapeutic Response:
It sounds incredibly frustrating lying there for hours while your mind races, especially with worries about work and that upcoming presentation. Feeling unprepared on top of that must be really draining.

That feeling of your mind just not switching off can be exhausting.

Could you tell me a bit more about what that feeling of being unprepared is like for you when you're trying to rest?

EXAMPLE 2:
User: I don't know, I just haven't felt like eating much since that argument with my partner last weekend. Food just doesn't appeal.
Raw Response (from Nav Prompt): Validate lack of appetite post-argument. Explore connection between mood and eating. [CONF:80%;NODE_EATING_MOOD_LINK]
Therapeutic Response:
It sounds like that argument with your partner last weekend has really impacted you, to the point where even food doesn't feel appealing right now. That loss of appetite must feel quite unsettling.

It's common for our emotional state, especially after conflict, to affect us physically like that.

How have you been feeling emotionally since the argument, aside from the lack of appetite?

EXAMPLE 3:
User: I managed to sleep okay last night but I still feel so tired today, like I haven't rested at all.
Raw Response (from Nav Prompt): Acknowledge sleep quality vs feeling rested. Ask about daytime fatigue. [CONF:65%;NODE_SLEEP_QUALITY]
Therapeutic Response:
I hear you – it's confusing and frustrating when you technically sleep, but still wake up feeling tired, like the rest didn't quite 'take'. That daytime fatigue can really weigh on you.

Sometimes the *quality* of sleep matters just as much as the hours.

What is that tiredness like for you during the day? Does it feel more physical or mental?
"""

        prompt = f"""# Therapeutic Communication Styling

## Context
You are refining a response drafted by another part of the system. Your goal is to ensure it sounds like a genuinely skilled, empathetic therapist crafting a warm, supportive message for someone experiencing difficulties related to {disorder if disorder else "mental wellbeing"}. The aim is to build rapport, encourage continued sharing, and make the user feel heard.

## User's Latest Message (CRITICAL - Ensure Response Acknowledges This)
"{latest_user_msg}"

## Raw Message to Restyle (This was generated based on the user's message and internal logic)
```
{raw_message}
```

## Recent Conversation History
{conversation_history}

## Sample Therapeutic Styling (Demonstrating Tone, Empathy, Open Questions)
{sample_conversations}

## Style Requirements
1.  **Empathy & Validation First:** Start by directly acknowledging and validating specific feelings, experiences, or words from the *user's latest message*. Show you've truly heard them.
2.  **Conversational & Warm:** Use a natural, human, and caring tone. Avoid jargon, clinical language, or sounding robotic.
3.  **Concise & Readable:** Keep responses relatively brief (2-4 short paragraphs). Use generous white space between paragraphs.
4.  **Open-Ended Exploration:** If the raw message includes a question, ensure it's gentle, open-ended, and invites sharing rather than demanding specific answers. Frame it as an invitation (e.g., "Would you be open to sharing more about...?", "I wonder if you could tell me about..."). If no question is present, consider adding one if it feels natural to encourage dialogue.
5.  **Focus on User's Pace:** Convey patience and that there's no pressure to talk about anything they're not ready for.
6.  **Subtle Guidance:** While the raw message might hint at a direction, the styled response should feel like a natural continuation of the user's train of thought, not a forced turn.

## Key Elements to Include (CRITICAL)
1.  **Specific Acknowledgment:** Explicitly reference 1-2 key details, feelings, or phrases the user mentioned in their *latest* message.
2.  **Validation:** Communicate that their feelings/experiences are understandable.
3.  **Empathy:** Convey genuine care and understanding of their struggle.
4.  **Gentle Invitation:** Include a soft, open-ended question to encourage them to share more if they wish.
5.  **Appropriate Pacing:** Don't rush towards solutions; focus on understanding the current experience.

## Task
Restyle the 'Raw Message to Restyle' into a therapeutic response that:
1.  **PRIORITIZES** acknowledging the user's specific situation and feelings from their *latest message*.
2.  Delivers the core intent of the raw message (e.g., exploring a certain feeling) in an empathetic, conversational way.
3.  Encourages continued conversation and makes the user feel safe and heard.

**DO NOT** add new therapeutic techniques or change the fundamental topic hinted at in the raw message. Your task is **ONLY** to improve the delivery, warmth, empathy, and conversational flow, ensuring it directly connects with the user's last statement. Keep it concise.
"""
        return prompt

    def process_response(self, response_text: str, prompt_type: str) -> Dict[str, Any]:
        """Process the model's response to extract structured information."""
        result = {
            "response_text": response_text,
            "disorder": None,
            "confidence": 0.0,
            "confidence_scores": {},
            "node_id": None,
             "navigation_confidence": 0.0  # New field for navigation confidence       
             }
        
        # Clean up response text by removing any markers
        cleaned_response = response_text
        
        if prompt_type == "first":
            # Extract JSON for confidence scores
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    assessment = json.loads(json_str)
                    
                    # Check if we have confidence scores
                    if "confidence_scores" in assessment:
                        result["confidence_scores"] = assessment["confidence_scores"]
                        
                        # Find the disorder with highest confidence
                        if result["confidence_scores"]:
                            top_disorder = max(result["confidence_scores"].items(), key=lambda x: x[1])
                            result["disorder"] = top_disorder[0]
                            result["confidence"] = top_disorder[1]
                    
                    # Remove the JSON from the response text
                    cleaned_response = response_text.replace(json_str, "").strip()
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {json_match.group(1) if json_match else 'No match'}")
        
        elif prompt_type == "node_identification":
            # Extract node ID
            try:
                node_data = json.loads(response_text)
                result["node_id"] = node_data.get("node")
                result["confidence"] = node_data.get("confidence", 0.0)
                # For node identification, we don't modify the response_text as it's not shown to the user
            except json.JSONDecodeError:
                print(f"Failed to parse node identification JSON: {response_text}")
        
        elif prompt_type == "navigation":
            # Extract confidence and node ID from the format [CONF:XX%;NODE_ID]
            conf_node_match = re.search(r'\[CONF:(\d+)%;([^\]]+)\]', response_text)
            if conf_node_match:
                confidence_str = conf_node_match.group(1)
                node_id = conf_node_match.group(2)
                
                try:
                    # Convert confidence percentage to decimal
                    result["navigation_confidence"] = float(confidence_str) / 100.0
                    result["node_id"] = node_id
                    
                    # Remove the marker from the response
                    full_match = f"[CONF:{confidence_str}%;{node_id}]"
                    cleaned_response = response_text.replace(full_match, "").strip()
                except ValueError:
                    print(f"Failed to parse confidence value: {confidence_str}")
        
        result["response_text"] = cleaned_response
        return result
    def send_prompt(self, prompt: str, model: str = "openai/gpt-4.1-mini") -> str:
        """Send a prompt to the OpenAI API and get the response."""
        try:
            # Set temperature based on prompt type
            temperature = 0.3  # Default for most prompts
            
            # For therapeutic styling, use slightly higher temperature for more natural variety
            if "Therapeutic Communication Styling" in prompt:
                temperature = 0.4
                
            # For navigation, use lower temperature for more consistent node selection
            if "Decision Tree Navigation Task" in prompt:
                temperature = 0.2
            
            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error sending prompt to OpenAI: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Could you please try again?"

class MedicalChatbot:
    def __init__(self):
        """Initialize the medical chatbot."""
        self.session = ChatbotSession()
        self.decision_tree = DecisionTreeHandler()
        self.vector_db = VectorDBHandler()
        self.prompt_handler = PromptHandler()
    
    def process_message(self, user_message: str) -> str:
        """Process a user message and generate a response."""
        self.session.add_message("user", user_message)

        response = "" # Initialize response

        # Phase 1: Disorder identification (if needed)
        if not self.session.identified_disorder or self.session.disorder_confidence < CONFIDENCE_THRESHOLD:
            print("Phase: Disorder Identification")
            response = self._handle_disorder_identification()

            # If disorder identified, find starting node (silently) and return initial response
            if self.session.identified_disorder and not self.session.current_node_id:
                node_id = self._identify_starting_node() # This is internal setup
                print(f"Internal: Identified starting node: {node_id}")
                self.session.current_node_id = node_id
                # The response from _handle_disorder_identification is already therapeutic
                self.session.add_message("assistant", response)
                return response
            # If disorder still not identified, return the exploratory response
            elif not self.session.identified_disorder:
                 self.session.add_message("assistant", response)
                 return response

        # Phase 2: Therapeutic Exploration / Tree Navigation (Implicit)
        if self.session.identified_disorder and self.session.current_node_id:
            print(f"Phase: Therapeutic Exploration (Current Node: {self.session.current_node_id})")
            response = self._handle_tree_navigation(user_message)
        # Fallback if state is inconsistent
        elif not response:
            print("Warning: Inconsistent state, generating fallback response.")
            response = "I'm sorry, I seem to have gotten a bit lost in our conversation. Could you perhaps share again what's been on your mind?"
            # Reset state partially? Or just rely on next user message. For now, just respond.

        # Add assistant response to history
        self.session.add_message("assistant", response)
        return response

    def _handle_disorder_identification(self) -> str:
        """Handle the disorder identification phase using therapeutic conversation."""
        # Generate and send first prompt (already designed to be therapeutic)
        prompt = self.prompt_handler.generate_first_prompt(self.session)
        raw_response = self.prompt_handler.send_prompt(prompt)
        print(f"Raw disorder identification response: {raw_response}")

        # Process the response to extract hidden JSON and get clean text
        processed = self.prompt_handler.process_response(raw_response, prompt_type="first")
        therapeutic_text = processed["response_text"] # This is the user-facing text

        # Check confidence scores internally
        if processed["confidence_scores"]:
            print(f"Internal: Confidence scores: {json.dumps(processed['confidence_scores'])}")
            for disorder, confidence in processed["confidence_scores"].items():
                if confidence >= CONFIDENCE_THRESHOLD:
                    if not self.session.identified_disorder: # Only update if not already set
                         print(f"Internal: Identified disorder {disorder} with confidence {confidence}")
                         self.session.identified_disorder = disorder
                         self.session.disorder_confidence = confidence
                         # Load tree internally
                         self.session.decision_tree = self.decision_tree.load_tree(disorder)
                         if not self.session.decision_tree:
                              print(f"Error: Failed to load decision tree for {disorder}")
                              # Modify response slightly to indicate internal issue without breaking character
                              therapeutic_text = f"Thank you for sharing that. I'm understanding more about the challenges you're facing, particularly concerning {disorder}. I need a moment to gather my thoughts on how best to continue our conversation. Could you perhaps tell me a little more about how this has been affecting your day-to-day?"
                    break # Stop after finding the first one above threshold

        # Return the therapeutic text generated by the first prompt
        return therapeutic_text

    def _identify_starting_node(self) -> str:
        """Identify the appropriate starting node in the decision tree. (Now purely internal)."""
        print("Internal: Defaulting start node to NODE_A")
        # Ensure rootNode exists if we were to use it:
        # root_node_id = self.session.decision_tree.get("rootNode", "NODE_A") # Default to NODE_A if no root
        # return root_node_id
        return "NODE_A"

    def _handle_tree_navigation(self, user_message: str) -> str:
        """Handle the therapeutic exploration and implicit navigation phase."""
        # Check if we are ready to deliver the script from the previous turn
        final_response = None
        if self.session.at_leaf_node and self.session.therapeutic_script:
            script_content = self.session.therapeutic_script
            self.session.therapeutic_script = None
            self.session.at_leaf_node = False
            print("Delivering prepared therapeutic script.")
            # The script introduction should have happened last turn. Now just deliver.
            # Maybe add a small lead-in?
            # return f"Okay, here is the '{script_name_placeholder}' exercise we discussed:\n\nSCRIPT:::{script_content}"
            # For now, keep the simple marker for the frontend:
            return f"SCRIPT:::{script_content}"

        current_node = self.session.get_current_node()
        if not current_node:
            print("Error: Current node is None during navigation.")
            return "I seem to have lost my place in our conversation. Could you remind me what we were discussing?"

        # Get relevant RAG results (internal context)
        rag_results = self.vector_db.search(
            query=user_message,
            disorder=self.session.identified_disorder,
            limit=MAX_RAG_RESULTS
        )

        # STEP 1: Generate Therapeutic Exploration & Internal Navigation Assessment
        nav_prompt = self.prompt_handler.generate_navigation_prompt(
            session=self.session,
            rag_results=rag_results
        )
        raw_nav_response = self.prompt_handler.send_prompt(nav_prompt)
        print(f"DEBUG: Raw exploration/navigation response: {raw_nav_response}")

        # Process the response to separate therapeutic text and navigation marker
        processed = self.prompt_handler.process_response(raw_nav_response, prompt_type="navigation")
        exploratory_text = processed["response_text"] # This is the text part generated by the nav prompt
        nav_confidence = processed["navigation_confidence"]
        next_node_id = processed["node_id"]
        print(f"Internal: Navigation assessment - Node: {next_node_id}, Conf: {nav_confidence}")

        # Define confidence thresholds (can be adjusted)
        NAVIGATION_THRESHOLD = 0.80 # Single threshold for moving nodes

        identified_leaf_this_turn = False
        leaf_introduction_response = None # Store the intro message if generated

        # Update state based on navigation assessment (internal)
        if next_node_id:
            # Use the single NAVIGATION_THRESHOLD
            if nav_confidence >= NAVIGATION_THRESHOLD:
                old_node_id = self.session.current_node_id
                if old_node_id != next_node_id:
                     print(f"Internal: Confidence {nav_confidence} >= {NAVIGATION_THRESHOLD}. Moving from {old_node_id} to {next_node_id}")
                     self.session.current_node_id = next_node_id
                     # Now check if the *new* node is a leaf
                     new_node = self.session.get_current_node()
                     is_leaf_by_name = next_node_id.startswith("LEAF_")
                     is_leaf_by_data = new_node and new_node.get("isLeaf", False)

                     if is_leaf_by_name or is_leaf_by_data:
                         identified_leaf_this_turn = True
                         print(f"Internal: Leaf node {next_node_id} reached.")

                         script_content = self.decision_tree.get_therapeutic_script(
                             self.session.current_node_id,
                             self.session.identified_disorder,
                             self.session.decision_tree
                         )

                         if script_content:
                             self.session.therapeutic_script = script_content
                             self.session.at_leaf_node = True
                             print(f"Internal: Prepared script for node {self.session.current_node_id}.")

                             # Generate the script introduction message NOW
                             script_name_for_prompt = "this exercise"
                             script_tags_for_prompt = new_node.get("tags", [])
                             script_info = new_node.get("recommendation", {}).get("script", "")
                             if ":" in script_info:
                                 script_name_for_prompt = script_info.split(":")[1].strip().strip('"')
                             elif "SCRIPT_" in script_info:
                                 script_name_for_prompt = script_info.replace("SCRIPT_","").replace("_", " ")

                             intro_prompt = self.prompt_handler.generate_script_introduction_prompt(
                                 session=self.session,
                                 script_name=script_name_for_prompt,
                                 script_tags=script_tags_for_prompt
                             )
                             # This intro is the final response for *this* turn
                             leaf_introduction_response = self.prompt_handler.send_prompt(intro_prompt)
                             print(f"DEBUG: Generated script introduction response: {leaf_introduction_response}")
                         else:
                             print(f"Warning: Leaf node {self.session.current_node_id} reached, but no script found.")
                             self.session.at_leaf_node = False
                     else:
                         print(f"Internal: Confidence {nav_confidence}. Staying at {old_node_id}")
                         # No state change needed if staying, proceed to styling

            else:
                 # Low confidence - Generate a clarification request
                 print(f"Internal: Confidence {nav_confidence} < {NAVIGATION_THRESHOLD}. Staying at {self.session.current_node_id} and asking for clarification.")

                 # Use the clarification prompt handler method
                 clarification_prompt = self.prompt_handler.generate_clarification_prompt(
                     session=self.session,
                     original_exploratory_text=exploratory_text # Pass the LLM's initial thought
                 )
                 # Send this prompt to get the actual response for the user
                 final_response = self.prompt_handler.send_prompt(clarification_prompt)
                 print(f"DEBUG: Generated clarification response: {final_response}")
                 # Skip the default styling step below, as we've generated a specific response
                 leaf_introduction_response = None # Ensure we don't overwrite

        else:
            # No node ID provided by LLM (shouldn't happen with the prompt)
            print("Warning: No node ID provided in navigation assessment. Staying at current node.")
            # Ask for clarification as a fallback
            clarification_prompt = self.prompt_handler.generate_clarification_prompt(
                 session=self.session,
                 original_exploratory_text="I'm trying to understand the best way forward." # Generic text
             )
            final_response = self.prompt_handler.send_prompt(clarification_prompt)
            print(f"DEBUG: Generated clarification response (no node ID fallback): {final_response}")
            leaf_introduction_response = None # Ensure no overwrite


        # Handle redirection nodes if we haven't just identified a leaf *or asked for clarification*
        if not identified_leaf_this_turn and final_response is None: # Check if final_response was set by clarification
             current_node = self.session.get_current_node() # Re-get node in case it changed
             if current_node and "redirectTo" in current_node:
                 redirect_node_id = current_node["redirectTo"]
                 print(f"Internal: Redirecting from {self.session.current_node_id} to {redirect_node_id}")
                 self.session.current_node_id = redirect_node_id
                 # Note: Redirection might ideally trigger another internal processing loop
                 # immediately, but for now, we'll let the next user message handle it.
                 # We still need a response for *this* turn. Let's use the styling.

        # STEP 2: Generate Final Therapeutic Response for the User (if not already generated)

        # If we generated a specific script introduction, use that
        if leaf_introduction_response:
            final_response = leaf_introduction_response
        # If we generated a clarification response, final_response is already set
        elif final_response:
             pass # Already handled by the low-confidence or no-node-ID block
        else:
            # Otherwise (high confidence move, not a leaf, or stayed confidently),
            # style the exploratory text generated in Step 1
            style_prompt = self.prompt_handler.generate_therapeutic_styling_prompt(
                raw_message=exploratory_text, # Use the text part from the nav response
                conversation_history=self.session.get_conversation_history_formatted(max_messages=4),
                disorder=self.session.identified_disorder,
                latest_user_msg=user_message
            )
            therapeutic_response = self.prompt_handler.send_prompt(style_prompt)
            print(f"DEBUG: Therapeutically styled response: {therapeutic_response}")
            final_response = therapeutic_response

        # Add the final response to history before returning
        if final_response:
             self.session.add_message("assistant", final_response)

        return final_response

    def reset_session(self) -> None:
        """Reset the chatbot session."""
        self.session.reset()

# Example usage
if __name__ == "__main__":
    chatbot = MedicalChatbot()
    
    print("Medical Chatbot initialized. Type 'exit' to quit.")
    print("Bot: Hello! I'm here to help with your concerns. Could you tell me what's been troubling you?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
            
        response = chatbot.process_message(user_input)
        print(f"Bot: {response}")