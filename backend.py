import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
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
        """Generate a prompt for decision tree navigation with enhanced therapeutic approach."""
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
        script_info = ""
        if is_leaf or "recommendation" in current_node:
            recommendation = current_node.get("recommendation", {})
            script = recommendation.get("script", "No script available")
            recommendation_text = f"\n## Recommendation\nScript: {script}"
            script_info = script
            
            # Get tags if available
            if "tags" in current_node:
                tags = current_node.get("tags", [])
                recommendation_text += f"\nTags: {', '.join(tags)}"
        
        # Script introduction instructions
        script_intro_text = ""
        if is_leaf:
            # Extract script name for introduction
            script_name = "this exercise"
            if ":" in script_info:
                script_name = f'"{script_info.split(":")[1].strip().strip("\"")}"'
            
            # Get tag information
            tags_text = ""
            if "tags" in current_node:
                tags = current_node.get("tags", [])
                if tags:
                    tags_text = f" focusing on {', '.join(tags[:3])}"
            
            script_intro_text = f"""
    ## Script Introduction Guidelines
    If confirmed as an appropriate leaf node, provide a thoughtful introduction to the therapeutic script.
    - Mention that you're going to share {script_name}{tags_text}
    - Explain how this exercise relates to the user's specific experiences
    - Create a gentle transition that prepares them for the script
    - Make it clear that this is designed to help with their particular situation
    - Keep the introduction warm, conversational, and encouraging
    """
        exploration_guidance = """
## Path Exploration Instructions
1. Consider ALL possible paths that might match the user's symptoms, not just the most obvious one
2. Carefully weigh the evidence for different paths before selecting one
3. Look for symptoms the user mentions that might indicate alternative paths
4. Be willing to suggest a node that isn't the first obvious choice if it better matches their overall description
5. Pay special attention to recent messages for new information that might suggest a different direction
6. Don't commit too early to a path before gathering sufficient information
"""
        # Add varied therapeutic response patterns
        therapeutic_phrases = """
    ## Therapeutic Response Variety
    Use a variety of therapeutic phrasing to avoid repetition. Below are examples to vary your tone and approach:

    ### Warm Welcome & Emotional Validation (examples)
    - "I'm really sorry you're feeling this way. You don't have to go through it alone."
    - "What you're going through matters. I'm with you, and you're allowed to feel exactly how you feel."
    - "It's okay to feel overwhelmed — we all do sometimes. Let's take a moment together."
    - "You're not alone in this. I'm here, and I care."
    - "I can see this is difficult, and I appreciate your willingness to explore it."

    ### Gentle Opening to Dialogue (examples)
    - "Would you like to share a bit about what's on your mind?"
    - "If you want, we can gently talk about what's going on."
    - "Is there anything you'd like to explore right now about these feelings?"
    - "No need to dive in unless you want to — we can start wherever you feel safe."
    - "Let's explore this together at a pace that feels comfortable for you."

    ### Safe Proposals & Transitions (examples)
    - "We could look at what's coming up for you — thoughts, sensations, emotions."
    - "If you'd like, we can explore what's beneath the surface."
    - "Whether it's your thoughts racing or your body feeling tense, we can find a gentle path forward."
    - "Let's go at your pace. We can unpack a little, or just move toward something calming."
    - "I'm wondering if it might be helpful to explore this exercise that focuses specifically on what you're describing."

    ### Script Introductions (examples)
    - "Based on what you've shared, I'd like to introduce an exercise that might bring some relief."
    - "I think there's an approach that could be particularly helpful for what you're experiencing."
    - "I'd like to share a practice that was designed specifically for situations like the one you're describing."
    - "Given what you've told me, I believe this exercise might offer some meaningful support."
    - "Would it be alright if I shared a therapeutic exercise that addresses exactly these kinds of feelings?"
    """
        
        prompt = f"""# Therapeutic Conversation Guide
        ## Current Focus Area
        {node_prompt}
        {recommendation_text}

        ## Possible Directions
        {options_text}

        ## Conversation History
        {conversation_history}

        ## Clinical Context 
        {clinical_context}
        {therapeutic_phrases}
        {exploration_guidance}

        ## Therapeutic Approach
        1. Use person-centered therapeutic techniques that emphasize empathy, unconditional positive regard, and authenticity
        2. Practice validation by acknowledging the person's feelings and experiences as understandable
        3. Employ reflective listening by paraphrasing and summarizing to show understanding
        4. Use therapeutic silence by giving space for the person to process their thoughts
        5. Follow the person's lead while gently guiding toward beneficial areas of exploration
        6. Focus on strengths and resilience, not just challenges
        7. Use collaborative language ("we", "let's explore", "together we might")
        8. Balance emotional support with practical guidance
        9. Vary your therapeutic language patterns to avoid repetition and create a more natural conversation

        ## Technical Navigation Requirements (Hidden from User)
        1. Prioritize a thorough and accurate understanding of the user's situation
        2. At the end of your response, include your confidence level and your assessment of the most appropriate node in this format: [CONF:XX%;NODE_ID]
        For example: [CONF:75%;NODE_A2A] or [CONF:90%;LEAF_A2A1]
        3. Rate your confidence percentage (0-100%) based on:
        - How well you understand the patient's situation
        - How clearly their symptoms match a particular path
        - How much information you've gathered so far
        4. Select the node that best matches the user's symptoms and concerns based on all available information
        5. Keep all technical details and reasoning hidden from the user

        ## Effective Questioning Guidelines
        1. Ask ONE clear, specific question in each response that directly relates to understanding the user's experience
        2. Frame questions to elicit specific symptoms or experiences rather than general feelings
        3. Avoid vague questions like "How does that feel?" or "Does that make sense?"
        4. Instead ask targeted questions like "Do you notice this happens more in the evening or morning?" or "When you experience X, do you also notice Y happening?"
        5. Make questions concrete and behaviorally focused rather than abstract
        6. Your question should help clarify aspects of the user's experience that are still unclear
        7. Ensure your question feels natural within the conversation, not clinical or interrogative
        8. Make the question the clear focus of your response - don't bury it in other text

        ## Response Guidelines
        - Respond with genuine warmth and care
        - Validate the person's feelings and experiences
        - Use therapeutic reflection to demonstrate understanding
        - Offer gentle guidance while respecting autonomy
        - Frame suggestions as collaborative explorations
        - Use language that conveys hope and possibility
        - Keep your response conversational and natural
        - If exploring a potential leaf node, provide a meaningful introduction to the possible therapeutic approach
        - Vary your phrasing to sound more human and less repetitive
        - IMPORTANT: Do NOT end your response with generic questions like "How does that feel?" or "Does that make sense?"
        - Include ONLY the confidence percentage and node ID in the specified format at the very end

        {script_intro_text}

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
            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
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
        
        # Last resort fallback
        return "NODE_A"
    
    def _handle_tree_navigation(self, user_message: str) -> str:
        """Handle the decision tree navigation phase."""
        current_node = self.session.get_current_node()
        
        # Case 1: We're at a confirmed leaf node and have a script
        if self.session.at_leaf_node and self.session.therapeutic_script:
            # Just return the script without wrapping it in a chat message
            return self.session.therapeutic_script
            
        # Case 2: We're at a confirmed leaf node but need to get the script
        if current_node and current_node.get("isLeaf", False) and self.session.at_leaf_node:
            script = self.decision_tree.get_therapeutic_script(
                self.session.current_node_id,
                self.session.identified_disorder,
                self.session.decision_tree
            )
            
            if script:
                self.session.therapeutic_script = script
                # Just return the script without wrapping it in a chat message
                return script
        
          # Case 3: Still navigating the tree - use RAG and LLM
        # Get relevant RAG results
        rag_results = self.vector_db.search(
            query=user_message,
            disorder=self.session.identified_disorder,
            limit=MAX_RAG_RESULTS
        )
        
        # Generate and send navigation prompt
        prompt = self.prompt_handler.generate_navigation_prompt(
            session=self.session,
            rag_results=rag_results  # Includes RAG results for context
        )
        raw_response = self.prompt_handler.send_prompt(prompt)
        
        # Process the response
        processed = self.prompt_handler.process_response(raw_response, prompt_type="navigation")
        print(f"Navigation response processed: {processed}")
        
        # Use confidence threshold to determine whether to navigate
        CONFIDENCE_THRESHOLD = 0.85  
        
        if processed["node_id"]:
            # Only change nodes if confidence is above threshold or it's the same node
            if processed["navigation_confidence"] >= CONFIDENCE_THRESHOLD or processed["node_id"] == self.session.current_node_id:
                # If confidence is high enough or we're staying at the same node, update the node ID
                old_node_id = self.session.current_node_id
                self.session.current_node_id = processed["node_id"]
                
                if old_node_id != processed["node_id"]:
                    print(f"Moving from {old_node_id} to {processed['node_id']} with confidence {processed['navigation_confidence']}")
                
                # Check if this is a leaf node
                new_node = self.session.get_current_node()
                is_leaf_by_name = processed["node_id"].startswith("LEAF_")
                is_leaf_by_data = new_node and new_node.get("isLeaf", False)
                
                if is_leaf_by_name or is_leaf_by_data:
                    self.session.at_leaf_node = True
                    
                    # If it's a leaf node, check for a therapeutic script
                    script = self.decision_tree.get_therapeutic_script(
                        self.session.current_node_id,
                        self.session.identified_disorder,
                        self.session.decision_tree
                    )
                    
                    if script:
                        # Save script to session
                        self.session.therapeutic_script = script
                        # Return the processed response (the model will have created a suitable introduction)
                        return processed["response_text"]
            else:
                # If confidence is too low, don't change nodes but log this
                print(f"Staying at {self.session.current_node_id} because confidence {processed['navigation_confidence']} is below threshold {CONFIDENCE_THRESHOLD}")
        
        # Handle redirection nodes
        current_node = self.session.get_current_node()
        if current_node and "redirectTo" in current_node:
            redirect_node_id = current_node["redirectTo"]
            print(f"Redirecting from {self.session.current_node_id} to {redirect_node_id}")
            self.session.current_node_id = redirect_node_id
        
        return processed["response_text"]
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