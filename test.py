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
        self.current_node = None
        self.tree_content = None
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

    def reset(self) -> None:
        """Reset the session state."""
        self.conversation_history = []
        self.identified_disorder = None
        self.disorder_confidence = 0.0
        self.current_node = None
        self.tree_content = None
        self.at_leaf_node = False
        self.therapeutic_script = None

class DecisionTreeHandler:
    def __init__(self, tree_dir: str = "./trees"):
        self.tree_dir = Path(tree_dir)
    
    def load_tree(self, disorder: str) -> Optional[str]:
        """Load the decision tree for a specific disorder."""
        # Look for tree document with the disorder name
        print(f"Loading tree for disorder: {disorder}")
        for ext in ['.txt', '.docx']:
            potential_file = self.tree_dir /f"{disorder}"/ f"{disorder}_tree{ext}"
            print(f"Checking for file: {potential_file}")
            if potential_file.exists():
                return self._read_file_content(potential_file)
                
            # Try alternative naming patterns
            potential_file = self.tree_dir / f"tree_{disorder}{ext}"
            if potential_file.exists():
                return self._read_file_content(potential_file)
        
        print(f"Warning: No tree document found for disorder: {disorder}")
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
    
    def get_therapeutic_script(self, leaf_node: str, disorder: str) -> Optional[str]:
        """Get the therapeutic script associated with a leaf node."""
        # Parse leaf node to identify script
        script_match = re.search(r'SCRIPT_(\w+)', leaf_node)
        if not script_match:
            return None
            
        script_id = script_match.group(1)
        
        # Look for script files with matching ID
        script_dir = self.tree_dir / disorder / "scripts"
        if not script_dir.exists():
            script_dir = self.tree_dir  # Fall back to main directory
            
        for file in script_dir.glob(f"*SCRIPT_{script_id}*"):
            return self._read_file_content(file)
            
        # If not found by ID, try to find by leaf node name
        for file in script_dir.glob(f"*{leaf_node}*"):
            return self._read_file_content(file)
            
        return None

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
    def generate_node_identification_prompt(self, session: ChatbotSession, tree_content: str) -> str:
        """Generate a prompt to identify the appropriate starting node in the decision tree."""
        conversation_history = session.get_conversation_history_formatted()
        
        prompt = f"""# Decision Tree Node Identification Task

    ## Context
    The user has been identified as having concerns related to {session.identified_disorder.upper()}.
    I need you to identify the most appropriate starting node in the decision tree based on their conversation history.

    ## Decision Tree Content
    {tree_content}

    ## Chat History
    {conversation_history}

    ## Instructions
    1. Analyze the user's messages to understand their specific symptoms and concerns
    2. Review the decision tree structure to find the most appropriate NODE to begin navigation
    3. Choose the most specific node possible based on the information available
    4. Return your assessment as a JSON object with the node ID and your confidence:
    {{"node": "NODE_XYZ", "confidence": 0.0-1.0}}

    ## Response
    Provide ONLY the JSON object with your assessment, nothing else.
    """
        
        return prompt
    def generate_first_prompt(self, session: ChatbotSession) -> str:
        """Generate the first prompt for disorder identification."""
        last_user_message = session.get_last_user_message()
        conversation_history = session.get_conversation_history_formatted()
        
        prompt = f"""# Disorder Identification Task

    ## Context
    You are a specialized medical chatbot tasked with identifying the primary psychological or mental health disorder from a user's messages.

    ## Available Disorders
    {', '.join(AVAILABLE_DISORDERS)}

    ## User Messages
    {conversation_history}

    ## Instructions
    1. Based on the user's message(s), assess the likelihood of each disorder category
    2. You may need to ask follow-up questions to gather more information
    3. Respond in a warm, empathetic manner while gathering information
    - Behave like a human therapist, not a machine
    - Don't repeat the user's message back to them
    - Avoid using technical jargon or clinical terms
    4. In EVERY response, include a JSON assessment at the end with confidence scores for ALL disorders:
    {{"confidence_scores": {{"sleep": 0.0-1.0, "eating": 0.0-1.0}}}}
    5. Make sure the JSON is valid and includes all available disorders
    6. Do not mention JSON or confidence scores to the user in your response text

    ## Response Guidelines
    - Start with an empathetic response to the user
    - Ask relevant follow-up questions about symptoms, timing, triggers, etc.
    - ALWAYS include confidence scores for ALL disorders at the end of your message
    - Higher confidence (>0.7) indicates strong evidence for that disorder
    """
        
        return prompt
    
    def generate_second_prompt(self, 
                              session: ChatbotSession, 
                              tree_content: str, 
                              rag_results: List[Dict[str, Any]]) -> str:
        """Generate the second prompt for tree navigation."""
        conversation_history = session.get_conversation_history_formatted()
        print(f"decision tree content: {tree_content}") 
        # Format RAG results
        clinical_context = ""
        for i, result in enumerate(rag_results):
            clinical_context += f"CLINICAL CONTEXT {i+1} (Section: {result['section']}):\n{result['content']}\n\n"
        print(f"current node: {session.current_node}")
        # Check if we're at a starting point or continuing navigation
        navigation_task = f"""
2. CONTINUE navigation from current node: {session.current_node}
3. Ask follow-up questions based on the decision tree structure to reach a leaf node
"""


        prompt = f"""# Decision Tree Navigation Task

## Disorder Identified
{session.identified_disorder.upper()}

## Decision Tree Content
{tree_content}

## Chat History
{conversation_history}

## Clinical Context from Database
{clinical_context}

## Instructions
1. Analyze the user's messages in relation to the decision tree
{navigation_task}
4. Use the clinical context to inform your navigation and responses
5. If you reach a leaf node, indicate this in your response with: [LEAF_NODE: node_id]
6. Provide warm, empathetic responses while guiding the user through the tree

## Response Format
- Start with an empathetic response to the user
- Ask focused questions that help navigate to the appropriate tree node
- If a leaf node is reached, include [LEAF_NODE: node_id] at the end of your message
- Do not expose the technical details of the tree to the user
- Make your questions conversational and natural, not clinical or mechanical
"""
        
        return prompt

    def process_response(self, response_text: str, is_first_prompt: bool) -> Dict[str, Any]:
        """Process the model's response to extract structured information."""
        result = {
            "response_text": response_text,
            "disorder": None,
            "confidence": 0.0,
            "confidence_scores": {},
            "leaf_node": None
        }
        
        # Clean up response text by removing any JSON
        cleaned_response = response_text
        
        if is_first_prompt:
            # Look for JSON assessment in first prompt response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    assessment = json.loads(json_str)
                    
                    # Check if we have confidence scores for all disorders
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
        else:
            # Look for leaf node in second prompt response
            leaf_match = re.search(r'\[LEAF_NODE: (.*?)\]', response_text)
            if leaf_match:
                result["leaf_node"] = leaf_match.group(1)
                
                # Remove the leaf node marker from the response text
                cleaned_response = re.sub(r'\[LEAF_NODE: (.*?)\]', '', response_text).strip()
        
        result["response_text"] = cleaned_response
        return result

    def send_prompt(self, prompt: str, model: str = "gpt-4o-mini") -> str:
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
            
            # If we just identified a disorder, also identify starting node
            if self.session.identified_disorder and not self.session.current_node:
                node_id = self._identify_starting_node()
                print(f"Identified starting node: {node_id}")
        else:
            # Phase 3: Decision tree navigation from current node
            response = self._handle_tree_navigation(user_message)
        
        # Add assistant response to conversation history
        self.session.add_message("assistant", response)
        return response
    
    def _identify_starting_node(self) -> str:
        """Identify the appropriate starting node in the decision tree."""
        # Generate and send node identification prompt
        prompt = self.prompt_handler.generate_node_identification_prompt(
            session=self.session,
            tree_content=self.session.tree_content
        )
        raw_response = self.prompt_handler.send_prompt(prompt)
        print(f"Node identification response: {raw_response}")
        
        # Extract the node from JSON response
        try:
            node_data = json.loads(raw_response)
            node_id = node_data.get("node")
            confidence = node_data.get("confidence", 0.0)
            
            if node_id and confidence >= 0.6:  # Lower threshold for node identification
                self.session.current_node = node_id
                return node_id
            else:
                # Default to the root node if identification fails
                return "NODE_A"
        except:
            print("Failed to parse node identification response")
            return "NODE_A"  # Default to root node
    def _handle_disorder_identification(self) -> str:
        """Handle the disorder identification phase."""
        # Generate and send first prompt
        prompt = self.prompt_handler.generate_first_prompt(self.session)
        raw_response = self.prompt_handler.send_prompt(prompt)
        print(f"Raw response: {raw_response}")
        
        # Process the response
        processed = self.prompt_handler.process_response(raw_response, is_first_prompt=True)
        #print(f"Processed response: {processed}")
        
        # Check all confidence scores
        if processed["confidence_scores"]:
            # Print confidence scores for debugging
            print(f"Confidence scores: {json.dumps(processed['confidence_scores'])}")
            
            # Check if any disorder has confidence above threshold
            for disorder, confidence in processed["confidence_scores"].items():
                if confidence >= CONFIDENCE_THRESHOLD:
                    self.session.identified_disorder = disorder
                    self.session.disorder_confidence = confidence
                    
                    # Load decision tree for the identified disorder
                    self.session.tree_content = self.decision_tree.load_tree(disorder)
                    if not self.session.tree_content:
                        return f"I've identified that you may be experiencing issues related to {disorder}, but I'm having trouble accessing the specific guidance for this area. Could you try describing your situation again, or perhaps focus on a different aspect of your concerns?"
                    
                    # Add transition to response if we're moving to phase 2
                    return f"{processed['response_text']}\n\nI think I understand what you're experiencing. Let me ask you a few more specific questions to better help you."
        
        return processed["response_text"]
    
    def _handle_tree_navigation(self, user_message: str) -> str:
        """Handle the decision tree navigation phase."""
        # If we're at a leaf node, check if we need to present a therapeutic script
        if self.session.at_leaf_node and not self.session.therapeutic_script:
            script = self.decision_tree.get_therapeutic_script(
                self.session.current_node, 
                self.session.identified_disorder
            )
            if script:
                self.session.therapeutic_script = script
                return f"Based on what you've shared, I think this might be helpful:\n\n{script}"
        
        # Get relevant RAG results to enhance the prompt
        rag_results = self.vector_db.search(
            query=user_message,
            disorder=self.session.identified_disorder,
            limit=MAX_RAG_RESULTS
        )
        
        # Generate and send second prompt
        prompt = self.prompt_handler.generate_second_prompt(
            session=self.session,
            tree_content=self.session.tree_content,
            rag_results=rag_results
        )
        raw_response = self.prompt_handler.send_prompt(prompt)
        
        # Process the response
        processed = self.prompt_handler.process_response(raw_response, is_first_prompt=False)
        print(f"processed 2nd response: {processed}")
        # Update session if a leaf node was reached
        if processed["leaf_node"]:
            self.session.current_node = processed["leaf_node"]
            self.session.at_leaf_node = True
            
            # Check for therapeutic script
            script = self.decision_tree.get_therapeutic_script(
                processed["leaf_node"], 
                self.session.identified_disorder
            )
            if script:
                self.session.therapeutic_script = script
                return f"{processed['response_text']}\n\nBased on what you've shared, I think this might be helpful:\n\n{script}"
        elif "NODE_" in raw_response:
            # Extract current node if mentioned in the response
            node_match = re.search(r'NODE_\w+', raw_response)
            if node_match:
                self.session.current_node = node_match.group(0)
        
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