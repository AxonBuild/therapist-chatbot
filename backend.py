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
AVAILABLE_DISORDERS = ['sleep', 'eating']

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
    You are a skilled and compassionate therapist having a conversation with someone who may be experiencing difficulties with their mental health. Your goal is to understand their experience and create a safe, supportive space.

    ## Available Areas of Focus
    {', '.join(AVAILABLE_DISORDERS)}

    ## Conversation History
    {conversation_history}

    ## Therapeutic Approach
    1. Create a safe emotional space by using warm, accepting language
    2. Practice active listening through reflection and validation
    3. Focus on the person's experience rather than diagnostic labels
    4. Use open-ended questions that invite deeper sharing
    5. Demonstrate genuine empathy by acknowledging emotions
    6. Avoid clinical terminology, judgments, or premature advice
    7. Respond to the emotional content as well as the factual content
    8. Use person-centered language that respects autonomy and agency

    ## Technical Requirements (Hidden from User)
    1. Based on the conversation, assess the likelihood each area needs attention:
    {{"confidence_scores": {{"sleep": 0.0-1.0, "eating": 0.0-1.0}}}}
    2. Include this assessment in JSON format at the end of EVERY response
    3. Never mention JSON or assessments to the user

    ## Response Guidelines
    - Begin with validation of the person's experience or feelings
    - Use reflection to show you understand ("It sounds like...", "I'm hearing that...")
    - Ask therapeutic questions that help explore their experience
    - Use a warm, conversational tone that conveys genuine care
    - Focus on building trust before problem-solving
    - Always include the hidden technical assessment
    """
        
        return prompt

    
#     def generate_node_identification_prompt(self, session: ChatbotSession) -> str:
#         """Generate a prompt to identify the appropriate starting node in the decision tree."""
#         conversation_history = session.get_conversation_history_formatted()
#         tree_data = session.decision_tree
        
#         # Get root node and first-level nodes
#         root_node_id = tree_data.get("rootNode")
#         root_node = tree_data["nodes"].get(root_node_id, {})
#         root_children = root_node.get("children", [])
        
#         # Format the nodes as options
#         options = []
#         for node_id in root_children:
#             node = tree_data["nodes"].get(node_id, {})
#             if "question" in node:
#                 options.append(f"{node_id}: {node['question']}")
#             elif "description" in node:
#                 options.append(f"{node_id}: {node['description']}")
                
#         options_text = "\n".join(options)
        
#         prompt = f"""# Decision Tree Node Identification Task

# ## Context
# The user has been identified as having concerns related to {session.identified_disorder.upper()}.
# I need you to identify the most appropriate starting node in the decision tree based on their conversation history.

# ## Decision Tree Title
# {tree_data.get("title", f"{session.identified_disorder.capitalize()} Decision Tree")}

# ## Available First-Level Nodes
# {options_text}

# ## Chat History
# {conversation_history}

# ## Instructions
# 1. Analyze the user's messages to understand their specific symptoms and concerns
# 2. Review the available first-level nodes and select the most appropriate one
# 3. Return your assessment as a JSON object with the node ID and your confidence:
#    {{"node": "NODE_XYZ", "confidence": 0.0-1.0}}


# ## Response
# Provide ONLY the JSON object with your assessment, nothing else.
# """
        
#         return prompt
    
    
    def generate_navigation_prompt(self, session: ChatbotSession, rag_results: List[Dict[str, Any]]) -> str:
        """Generate a prompt for decision tree navigation focused on decision-making."""
        conversation_history = session.get_conversation_history_formatted()
        current_node = session.get_current_node()
        
        if not current_node:
            return f"Error: No current node found in decision tree."
        
        # Determine if current node is a leaf
        is_leaf = current_node.get("isLeaf", False)
        print(f"Current node ID: {session.current_node_id}, is leaf: {is_leaf}")
        
        # Get node question or description
        node_prompt = current_node.get("question", current_node.get("description", "No description available"))
        
        # Get children or options
        options = []
        if not is_leaf:
            child_ids = current_node.get("children", [])
            for child_id in child_ids:
                child_node = session.decision_tree["nodes"].get(child_id, {})
                if "question" in child_node:
                    options.append(f"{child_id}: {child_node['question']}")
                elif "description" in child_node:
                    options.append(f"{child_id}: {child_node['description']}")
        
        options_text = "\n".join(options) if options else "This is a leaf node with no further options."
        
        # Format RAG results
        clinical_context = ""
        for i, result in enumerate(rag_results):
            clinical_context += f"CLINICAL CONTEXT {i+1} (Section: {result['section']}):\n{result['content']}\n\n"
        
        # Format recommendations if this is a leaf node
        recommendation_text = ""
        if is_leaf or "recommendation" in current_node:
            recommendation = current_node.get("recommendation", {})
            # Store the script info but don't put it directly in the main recommendation text
            script_info = recommendation.get("script", "No script available")
            # Change this line to be less direct about the script itself
            recommendation_text = "\\n## Recommendation\\nA therapeutic exercise or technique is recommended based on this path."

            # Get tags if available (can still be useful context)
            if "tags" in current_node:
                tags = current_node.get("tags", [])
                if tags: # Add check if tags list is not empty
                    recommendation_text += f"\\nTags: {', '.join(tags)}"
        
        # Simplified navigation prompt focused on decision making
        prompt = f"""# Decision Tree Navigation Task

## Current Focus Node
{node_prompt}
{recommendation_text}

## Possible Paths
{options_text}

## Conversation History
{conversation_history}

## Clinical Context 
{clinical_context}

## Task: Navigation Decision
1. Analyze the user's messages to determine which path in the decision tree best matches their situation
2. Infer answers to questions even when not directly stated by looking for implicit information
3. Consider the full context of the conversation history
4. If the user's message is vague or unclear, assign a lower confidence score
5. Do not commit to high confidence navigation unless the evidence is clear
6. Consider which child node (or current node) best matches the user's situation

## Response Format Instructions
1. Generate a clear, concise response that represents your navigation decision
2. Your response should identify which path to take but DOES NOT need to sound therapeutic yet
3. Clearly indicate with [CONF:XX%;NODE_ID] your confidence level and the node ID
4. For example: [CONF:75%;NODE_A2A] or [CONF:90%;LEAF_A2A1] 
5. For vague/unclear messages, use a low confidence score: [CONF:40%;NODE_A1]
6. Include a brief question to gather more information if needed
7. Keep your response straightforward and focused on the decision

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
        """Generate a prompt to style a response in a therapeutic manner."""
        
        # Sample therapist conversations showing ideal therapeutic responses
        sample_conversations = """
EXAMPLE 1:
User: I can't sleep at night. My mind keeps racing with worries about work and my upcoming performance review next week.
Raw Response: You appear to have difficulty with sleep onset insomnia related to work stress. Would meditation before bed help?
Therapeutic Response: 
Those racing thoughts about work and your upcoming performance review must make it really hard to settle in at night.

When our minds fixate on something important like a review next week, it can feel impossible to quiet down.

Would you like to explore some ways to ease that work-related worry before bedtime, or would you prefer we take a different approach?

EXAMPLE 2:
User: I don't know what's wrong with me. I just feel sad all the time since my mother's visit last month.
Raw Response: Persistent sadness could indicate depression. Can you identify specific triggers for your sad feelings?
Therapeutic Response: 
That persistent sadness you've been carrying since your mother's visit last month sounds really heavy.

Sometimes our relationships with family can stir up complex feelings that linger with us.

Would it help to talk a bit about what happened during that visit, or would you rather focus on finding ways to care for yourself right now?

EXAMPLE 3:
User: Sometimes I just don't feel hungry for days, especially when my workload increases at the office.
Raw Response: Loss of appetite lasting several days could indicate depression or an eating disorder. When did this pattern start?
Therapeutic Response:
Those periods without hunger, particularly when your workload gets heavy at the office, sound challenging.

Our bodies often respond to stress in ways that affect our basic needs like appetite.

What feels most important to you right now—understanding how your work stress might be affecting your eating patterns, or finding ways to nourish yourself during these busy times?
"""

        prompt = f"""# Therapeutic Communication Styling

## Context
You are a skilled, empathetic therapist crafting warm responses to someone experiencing difficulties related to {disorder if disorder else "mental wellbeing"}.

## User's Latest Message (IMPORTANT - ACKNOWLEDGE THESE DETAILS)
"{latest_user_msg}"

## Raw Message to Restyle
```
{raw_message}
```

## Recent Conversation History
{conversation_history}

## Sample Therapeutic Styling
{sample_conversations}

## Style Requirements
1. Keep responses brief and concise (3-5 short paragraphs maximum)
2. Use generous white space between paragraphs for readability
3. Maintain a warm, conversational tone that feels human, not clinical
4. Offer options rather than pushing in one direction
5. Use simple, everyday language instead of psychological terminology
6. Convey gentle support without pressure
7. Focus on being present with their experience rather than rushing to solutions
8. When asking questions, offer them as gentle invitations, not requirements

## Key Elements to Include (CRITICAL)
1. SPECIFIC acknowledgment of details the user mentioned (job, experiences, feelings, timeline)
2. Direct references to at least 2-3 specific words/phrases the user used
3. Some indication of choice or agency for the person
4. A gentle invitation to explore further if appropriate
5. Reassurance that there's no pressure or rush

## Task
Restyle the raw message into a therapeutic response that:
1. MUST explicitly reference specific details from the user's latest message
2. Delivers content in an empathetic, conversational way with appropriate spacing between thoughts
3. Acknowledges the user's actual experience before moving to questions or suggestions

DO NOT add new content, change the fundamental direction, or introduce new therapeutic suggestions not present in the raw message. Your task is ONLY to improve the delivery style WHILE ensuring specific user details are acknowledged.

Keep your response appropriately brief (similar length to the examples).
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
        # Add user message to conversation history
        self.session.add_message("user", user_message)
        
        # Determine which phase we're in
        if not self.session.identified_disorder or self.session.disorder_confidence < CONFIDENCE_THRESHOLD:
            # Phase 1: Disorder identification
            response = self._handle_disorder_identification()
            
            # If we just identified a disorder, identify starting node
            if self.session.identified_disorder and not self.session.current_node_id:
                node_id = self._identify_starting_node()
                print(f"Identified starting node: {node_id}")
                self.session.current_node_id = node_id
                
                # After identifying node, we should return to the user but not navigate yet
                self.session.add_message("assistant", response)
                return response
        
        # If we have a disorder and node, navigate the tree
        if self.session.identified_disorder and self.session.current_node_id:
            response = self._handle_tree_navigation(user_message)
        else:
            # This should only happen if something went wrong with identification
            response = "I'm sorry, I'm having trouble understanding your situation. Could you provide more details about what you're experiencing?"
        
        # Add assistant response to conversation history
        self.session.add_message("assistant", response)
        return response
    
    def _handle_disorder_identification(self) -> str:
        """Handle the disorder identification phase."""
        # Generate and send first prompt
        prompt = self.prompt_handler.generate_first_prompt(self.session)
        raw_response = self.prompt_handler.send_prompt(prompt)
        print(f"Raw disorder identification response: {raw_response}")
        
        # Process the response
        processed = self.prompt_handler.process_response(raw_response, prompt_type="first")
        
        # Check all confidence scores
        if processed["confidence_scores"]:
            print(f"Confidence scores: {json.dumps(processed['confidence_scores'])}")
            
            # Check if any disorder has confidence above threshold
            for disorder, confidence in processed["confidence_scores"].items():
                if confidence >= CONFIDENCE_THRESHOLD:
                    self.session.identified_disorder = disorder
                    self.session.disorder_confidence = confidence
                    
                    # Load decision tree for the identified disorder
                    self.session.decision_tree = self.decision_tree.load_tree(disorder)
                    if not self.session.decision_tree:
                        return f"I've identified that you may be experiencing issues related to {disorder}, but I'm having trouble accessing the specific guidance for this area. Could you try describing your situation again, or perhaps focus on a different aspect of your concerns?"
                    
                    return processed["response_text"]
        
        return processed["response_text"]
    
    def _identify_starting_node(self) -> str:
        """Identify the appropriate starting node in the decision tree."""
        # Generate and send node identification prompt
         #prompt = self.prompt_handler.generate_node_identification_prompt(self.session)
        # raw_response = self.prompt_handler.send_prompt(prompt)
        # print(f"Node identification response: {raw_response}")
        
        # # Process the response
        # processed = self.prompt_handler.process_response(raw_response, prompt_type="node_identification")
        
        # if processed["node_id"] and processed["confidence"] >= 0.6:
        #     self.session.current_node_id = processed["node_id"]
        #     processed["node_id"]="NODE_A"  # Placeholder for node ID
        #     return processed["node_id"]
        
        # Default to root node if identification fails
        # root_node_id = self.session.decision_tree.get("rootNode")
        # if root_node_id:
        #     self.session.current_node_id = root_node_id
        #     return root_node_id
        
        # Current implementation: Default to a fixed starting node.
        # If dynamic start node identification is needed later, the above commented code can be revisited.
        return "NODE_A"
    
    def _handle_tree_navigation(self, user_message: str) -> str:
        """Handle the decision tree navigation phase."""
        # Check if we are ready to deliver the script from the previous turn
        if self.session.at_leaf_node and self.session.therapeutic_script:
            script_content = self.session.therapeutic_script
            # Reset flags after delivering script
            self.session.therapeutic_script = None 
            self.session.at_leaf_node = False 
            print("Delivering prepared therapeutic script.")
            # Add prefix marker when returning the script
            return f"SCRIPT:::{script_content}"

        # Navigation logic starts now.
        current_node = self.session.get_current_node()
        # If somehow we lost the current node, handle gracefully
        if not current_node:
            print("Error: Current node is None during navigation.")
            return "I seem to have lost my place in our conversation. Could you remind me what we were discussing?"
            
        # Get relevant RAG results
        rag_results = self.vector_db.search(
            query=user_message,
            disorder=self.session.identified_disorder,
            limit=MAX_RAG_RESULTS
        )
        
        # STEP 1: PURE NAVIGATION DECISION
        # Generate and send navigation prompt
        nav_prompt = self.prompt_handler.generate_navigation_prompt(
            session=self.session,
            rag_results=rag_results
        )
        nav_response = self.prompt_handler.send_prompt(nav_prompt)
        print(f"DEBUG: Raw navigation response: {nav_response}")

        # Process the navigation response
        processed = self.prompt_handler.process_response(nav_response, prompt_type="navigation")
        print(f"Navigation processed: {processed}")
        
        # Define confidence thresholds for different navigation decisions
        HIGH_CONFIDENCE = 0.85  # Strong confidence to navigate or confirm leaf
        MEDIUM_CONFIDENCE = 0.60  # Reasonable but not certain
        LOW_CONFIDENCE = 0.50  # Very uncertain, likely needs clarification
        
        # Flag to check if we identified a leaf node in this turn
        identified_leaf_this_turn = False
        leaf_introduction = None
        
        # Process navigation decision
        if processed["node_id"]:
            nav_confidence = processed["navigation_confidence"]
            print(f"Navigation confidence: {nav_confidence}")
            
            # HIGH CONFIDENCE: Change nodes or confirm current path
            if nav_confidence >= HIGH_CONFIDENCE:
                # Update the node ID
                old_node_id = self.session.current_node_id
                self.session.current_node_id = processed["node_id"]
                
                if old_node_id != processed["node_id"]:
                    print(f"HIGH CONFIDENCE: Moving from {old_node_id} to {processed['node_id']} with confidence {nav_confidence}")
                else:
                    print(f"HIGH CONFIDENCE: Staying at {self.session.current_node_id} with confidence {nav_confidence}")
                
                # Check if this is a leaf node
                new_node = self.session.get_current_node()
                is_leaf_by_name = processed["node_id"].startswith("LEAF_")
                is_leaf_by_data = new_node and new_node.get("isLeaf", False)
                
                if is_leaf_by_name or is_leaf_by_data:
                    # Handle leaf node identification
                    identified_leaf_this_turn = True
                    print(f"Identified leaf node {processed['node_id']} with confidence {nav_confidence}")
                    
                    # Get script content for next turn
                    script_content = self.decision_tree.get_therapeutic_script(
                        self.session.current_node_id,
                        self.session.identified_disorder,
                        self.session.decision_tree
                    )
                    
                    if script_content:
                        # Save script content for next turn
                        self.session.therapeutic_script = script_content
                        self.session.at_leaf_node = True 
                        print(f"Prepared script for node {self.session.current_node_id}.")

                        # Generate script introduction
                        script_name_for_prompt = "this exercise" # Default
                        script_tags_for_prompt = new_node.get("tags", []) or []
                        # Extract better name if available
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
                        leaf_introduction = self.prompt_handler.send_prompt(intro_prompt)
                        print(f"DEBUG: Generated script introduction: {leaf_introduction}")
                    else:
                        print(f"Warning: Leaf node {self.session.current_node_id} identified, but no script found.")
                        self.session.at_leaf_node = False
            
            # MEDIUM CONFIDENCE: Still navigate but note uncertainty            
            elif nav_confidence >= MEDIUM_CONFIDENCE:
                old_node_id = self.session.current_node_id
                self.session.current_node_id = processed["node_id"]
                print(f"MEDIUM CONFIDENCE: Moving from {old_node_id} to {processed['node_id']} with confidence {nav_confidence}")
            
            # LOW CONFIDENCE: Stay at current node, ask clarifying question
            else:
                print(f"LOW CONFIDENCE: Staying at {self.session.current_node_id}, confidence {nav_confidence} is too low")
        else:
            print("No node ID provided. Staying at current node.")
            
        # Handle redirection nodes if needed
        if not identified_leaf_this_turn:
            current_node = self.session.get_current_node()
            if current_node and "redirectTo" in current_node:
                redirect_node_id = current_node["redirectTo"]
                print(f"Redirecting from {self.session.current_node_id} to {redirect_node_id}")
                self.session.current_node_id = redirect_node_id

        # STEP 2: THERAPEUTIC STYLING
        # If we have a leaf node introduction, use that directly
        if leaf_introduction:
            final_response = leaf_introduction
        else:
            # For normal navigation, apply therapeutic styling to the raw response
            raw_response = processed["response_text"]
            
            # Generate therapeutic styling prompt
            style_prompt = self.prompt_handler.generate_therapeutic_styling_prompt(
                raw_message=raw_response,
                conversation_history=self.session.get_conversation_history_formatted(max_messages=3),
                disorder=self.session.identified_disorder,
                latest_user_msg=user_message
            )
            
            # Send styling prompt to LLM
            therapeutic_response = self.prompt_handler.send_prompt(style_prompt)
            print(f"DEBUG: Therapeutically styled response: {therapeutic_response}")
            
            final_response = therapeutic_response
            
        # Return the final therapeutic response
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