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
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "EkoMindAI"
CONFIDENCE_THRESHOLD = 0.7
MAX_RAG_RESULTS = 3

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
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

    
    def generate_node_identification_prompt(self, session: ChatbotSession) -> str:
        """Generate a prompt to identify the appropriate starting node in the decision tree."""
        conversation_history = session.get_conversation_history_formatted()
        tree_data = session.decision_tree
        
        # Get root node and first-level nodes
        root_node_id = tree_data.get("rootNode")
        root_node = tree_data["nodes"].get(root_node_id, {})
        root_children = root_node.get("children", [])
        
        # Format the nodes as options
        options = []
        for node_id in root_children:
            node = tree_data["nodes"].get(node_id, {})
            if "question" in node:
                options.append(f"{node_id}: {node['question']}")
            elif "description" in node:
                options.append(f"{node_id}: {node['description']}")
                
        options_text = "\n".join(options)
        
        prompt = f"""# Decision Tree Node Identification Task

## Context
The user has been identified as having concerns related to {session.identified_disorder.upper()}.
I need you to identify the most appropriate starting node in the decision tree based on their conversation history.

## Decision Tree Title
{tree_data.get("title", f"{session.identified_disorder.capitalize()} Decision Tree")}

## Available First-Level Nodes
{options_text}

## Chat History
{conversation_history}

## Instructions
1. Analyze the user's messages to understand their specific symptoms and concerns
2. Review the available first-level nodes and select the most appropriate one
3. Return your assessment as a JSON object with the node ID and your confidence:
   {{"node": "NODE_XYZ", "confidence": 0.0-1.0}}

## Response
Provide ONLY the JSON object with your assessment, nothing else.
"""
        
        return prompt
    
    
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
        
        # Get parent node information for potential backtracking
        parent_node_id = None
        for node_id, node in session.decision_tree["nodes"].items():
            if "children" in node and session.current_node_id in node["children"]:
                parent_node_id = node_id
                break
        
        backtrack_text = ""
        if parent_node_id:
            parent_node = session.decision_tree["nodes"].get(parent_node_id)
            if parent_node:
                backtrack_text = f"\n## Parent Node (For Backtracking)\nID: {parent_node_id}\nDescription: {parent_node.get('question', parent_node.get('description', 'No description'))}"
        
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
            level = recommendation.get("level", 1)
            recommendation_text = f"\n## Recommendation\nScript: {script}\nLevel: {level}"
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
        
        prompt = f"""# Therapeutic Conversation Guide

    ## Current Focus Area
    {node_prompt}
    {recommendation_text}

    ## Possible Directions
    {options_text}
    {backtrack_text}

    ## Conversation History
    {conversation_history}

    ## Clinical Context 
    {clinical_context}
    {script_intro_text}

    ## Therapeutic Approach
    1. Use person-centered therapeutic techniques that emphasize empathy, unconditional positive regard, and authenticity
    2. Practice validation by acknowledging the person's feelings and experiences as understandable
    3. Employ reflective listening by paraphrasing and summarizing to show understanding
    4. Use therapeutic silence by giving space for the person to process their thoughts
    5. Follow the person's lead while gently guiding toward beneficial areas of exploration
    6. Focus on strengths and resilience, not just challenges
    7. Use collaborative language ("we", "let's explore", "together we might")
    8. Balance emotional support with practical guidance

    ## Technical Navigation Requirements (Hidden from User)
    1. Based on the conversation, determine the appropriate path forward
    2. For non-leaf nodes, include [NODE: selected_node_id] at the end of your response
    3. For appropriate leaf nodes, include [LEAF_CONFIRMED] at the end of your response
    4. If the user's response suggests the current path isn't appropriate, use [BACKTRACK: parent_node_id] to return to the parent node
    5. Keep all technical details and reasoning hidden from the user
    6. Always prioritize what the user is actually saying over the decision tree path

    ## Response Guidelines
    - Respond with genuine warmth and care
    - Validate the person's feelings and experiences
    - Use therapeutic reflection to demonstrate understanding
    - Offer gentle guidance while respecting autonomy
    - Frame suggestions as collaborative explorations
    - Use language that conveys hope and possibility
    - Keep your response conversational and natural
    - If at a leaf node, provide a meaningful introduction to the therapeutic script
    - IMPORTANT: Do NOT end your response with generic questions like "How does that feel?" or "Does that make sense?"
    - IMPORTANT: Do NOT add any follow-up questions beyond what's strictly needed for navigation
    - Include the appropriate technical marker at the very end
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
            "leaf_confirmed": False,
            "backtrack": False,
            "backtrack_node_id": None
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
            # Check for leaf confirmation - now check for both formats
            if "[LEAF_CONFIRMED]" in response_text:
                result["leaf_confirmed"] = True
                cleaned_response = response_text.replace("[LEAF_CONFIRMED]", "").strip()
            
            # Check for backtracking instruction
            backtrack_match = re.search(r'\[BACKTRACK:\s*([^\]]+)\]', response_text)
            if backtrack_match:
                result["backtrack"] = True
                result["backtrack_node_id"] = backtrack_match.group(1).strip()
                cleaned_response = re.sub(r'\[BACKTRACK:\s*([^\]]+)\]', '', response_text).strip()
            
            # Look for leaf node IDs in brackets, which might be included directly
            leaf_match = re.search(r'\[LEAF_([^\]]+)\]', response_text)
            if leaf_match:
                result["leaf_confirmed"] = True
                # The full match with brackets
                full_match = f"[LEAF_{leaf_match.group(1)}]"
                cleaned_response = response_text.replace(full_match, "").strip()
                
                # If we have a leaf ID but no node_id set yet, use the leaf ID
                if not result["node_id"]:
                    result["node_id"] = f"LEAF_{leaf_match.group(1)}"
            
            # Check for regular node IDs in brackets (like [NODE_A2A])
            node_id_match = re.search(r'\[(NODE_[^\]]+)\]', response_text)
            if node_id_match:
                # The full match with brackets
                full_match = f"[{node_id_match.group(1)}]"
                cleaned_response = response_text.replace(full_match, "").strip()
                
                # Set the node_id
                result["node_id"] = node_id_match.group(1)
            
            # Check for node selection with the [NODE: xyz] format
            node_match = re.search(r'\[NODE:\s*([^\]]+)\]', response_text)
            if node_match:
                result["node_id"] = node_match.group(1).strip()
                cleaned_response = re.sub(r'\[NODE:\s*([^\]]+)\]', '', response_text).strip()
        
        result["response_text"] = cleaned_response
        return result
    def send_prompt(self, prompt: str, model: str = "gpt-4.1-mini") -> str:
        """Send a prompt to the OpenAI API and get the response."""
        try:
            response = self.openai_client.chat.completions.create(
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
        prompt = self.prompt_handler.generate_node_identification_prompt(self.session)
        raw_response = self.prompt_handler.send_prompt(prompt)
        print(f"Node identification response: {raw_response}")
        
        # Process the response
        processed = self.prompt_handler.process_response(raw_response, prompt_type="node_identification")
        
        if processed["node_id"] and processed["confidence"] >= 0.6:
            self.session.current_node_id = processed["node_id"]
            return processed["node_id"]
        
        # Default to root node if identification fails
        root_node_id = self.session.decision_tree.get("rootNode")
        if root_node_id:
            self.session.current_node_id = root_node_id
            return root_node_id
        
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
        
        # Handle backtracking if needed
        if processed.get("backtrack") and processed.get("backtrack_node_id"):
            backtrack_node_id = processed["backtrack_node_id"]
            print(f"Backtracking from {self.session.current_node_id} to {backtrack_node_id}")
            self.session.current_node_id = backtrack_node_id
            self.session.at_leaf_node = False
        # Update session based on response
        elif processed["node_id"]:
            # User is being directed to a different node
            self.session.current_node_id = processed["node_id"]
            new_node = self.session.get_current_node()
            
            # Check if the new node is a leaf
            if new_node and new_node.get("isLeaf", False):
                self.session.at_leaf_node = True
        
        # If we've confirmed we're at a leaf node
        elif processed["leaf_confirmed"]:
            self.session.at_leaf_node = True
            
            # Check for therapeutic script
            script = self.decision_tree.get_therapeutic_script(
                self.session.current_node_id,
                self.session.identified_disorder,
                self.session.decision_tree
            )
            
            if script:
                # Save script to session
                self.session.therapeutic_script = script
                
                # Return response as is - the model will have already created a 
                # suitable introduction to the script based on our prompt
                return f"{processed['response_text']}\n\n{script}"
        
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