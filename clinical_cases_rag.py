import json
import os
from typing import List, Dict, Any, Set, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "ClinicalCases"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536
KEYWORD_THRESHOLD = 0.5
LIMITING_BELIEF_THRESHOLD = 0.7
CASE_DIR = os.path.join("Data", "cases")

# Configuration
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

qdrant_url = os.environ.get("QDRANT_URL")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

def embed_text(text: str) -> List[float]:
    """Generate embedding for text using OpenAI API."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def search_keyword_matches(symptoms: List[str]) -> Dict[str, float]:
    """Search for keyword matches (tags and emotions) in clinical cases."""
    case_match_scores = {}
    print("Searching for keyword matches...")
    print(f"Symptoms: {symptoms}")
    # For each symptom, search for matching keywords
    for symptom in symptoms:
        try:
            symptom_vector = embed_text(symptom)
            
            results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=symptom_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_type",
                            match=models.MatchValue(value="keyword")
                        )
                    ]
                ),
                limit=5,
                score_threshold=KEYWORD_THRESHOLD
            )
            
            # Process results
            for result in results:
                case_id = result.payload.get("case_id")
                score = result.score
                keyword = result.payload.get("original_text")
                
                print(f"Keyword match: '{keyword}' from {case_id} (score: {score:.4f})")
                
                # Add to case scores
                if case_id not in case_match_scores:
                    case_match_scores[case_id] = 0
                case_match_scores[case_id] += score
                
        except Exception as e:
            print(f"Error searching for keyword matches for '{symptom}': {e}")
    
    return case_match_scores

def search_limiting_belief_matches(user_message: str, session) -> Dict[str, Dict[str, Any]]:
    """Search for limiting belief matches in clinical cases."""
    belief_matches = {}
    
    # Get message embedding
    try:
        message_vector = embed_text(user_message)
        
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=message_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_type",
                        match=models.MatchValue(value="limiting_belief")
                    )
                ]
            ),
            limit=3,
            score_threshold=LIMITING_BELIEF_THRESHOLD
        )
        
        # Process results
        for result in results:
            case_id = result.payload.get("case_id")
            score = result.score
            belief = result.payload.get("original_text")
            
            print(f"Limiting belief match: '{belief}' from {case_id} (score: {score:.4f})")
            
            # Track if this is a new match
            is_new_match = True
            if hasattr(session, "matched_limiting_beliefs"):
                if belief in session.matched_limiting_beliefs:
                    is_new_match = False
            
            # Store match information
            belief_matches[belief] = {
                "case_id": case_id,
                "score": score,
                "is_new": is_new_match
            }
            
    except Exception as e:
        print(f"Error searching for limiting belief matches: {e}")
    
    return belief_matches

def load_case_file(case_id: str) -> Dict[str, Any]:
    """Load a clinical case file by ID."""
    try:
        file_path = os.path.join(CASE_DIR, f"{case_id}.json")
        if not os.path.exists(file_path):
            print(f"Case file not found: {file_path}")
            return {}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            case_data = json.load(f)
        return case_data
    except Exception as e:
        print(f"Error loading case file for {case_id}: {e}")
        return {}

def get_next_solutions(case_id: str, session) -> Dict[str, List[str]]:
    """Get the next solutions to present based on delivery status."""
    solutions = {
        "immediate": [],
        "intermediate": [],
        "long_term": []
    }
    
    case_data = load_case_file(case_id)
    if not case_data or "solutions" not in case_data:
        return solutions
    
    # Determine what's already been delivered
    delivered_solutions = {}
    if hasattr(session, "delivered_solutions") and case_id in session.delivered_solutions:
        delivered_solutions = session.delivered_solutions.get(case_id, {})
    
    # Determine which solution types to include
    if not delivered_solutions.get("immediate", False):
        solutions["immediate"] = case_data["solutions"].get("immediate", [])
    elif not delivered_solutions.get("intermediate", False):
        solutions["intermediate"] = case_data["solutions"].get("intermediate", [])
    elif not delivered_solutions.get("long_term", False):
        solutions["long_term"] = case_data["solutions"].get("long_term", [])
    
    return solutions

def fetch_clinical_cases(symptoms: List[str], user_message: str, session) -> Tuple[str, Set[str]]:
    """
    Fetch clinical cases based on symptoms and user message with simple historical belief tracking.
    
    Args:
        symptoms: List of identified symptoms/keywords
        user_message: The user's latest message
        session: The current chat session for tracking delivery status
        
    Returns:
        Tuple of (guidance text for system prompt, set of matched case IDs)
    """
    # Step 1: Search for keyword matches
    keyword_matches = search_keyword_matches(symptoms)
    
    # Step 2: Search for limiting belief matches in current message
    current_belief_matches = search_limiting_belief_matches(user_message, session)
    print("Current limiting belief matches:", current_belief_matches)
    # Step 3: Update session with current beliefs
    message_index = len(session.conversation_history) // 2
    if hasattr(session, "update_limiting_belief_matches"):
        session.update_limiting_belief_matches(current_belief_matches, message_index)
    
    # Step 4: Get all historical beliefs for scoring
    all_beliefs = getattr(session, "matched_limiting_beliefs", {})
    
    # Step 5: Simple combined scoring
    case_scores = keyword_matches.copy()
    
    # Add scores from all limiting beliefs (current gets 2x, historical gets 1x)
    for belief, info in all_beliefs.items():
        case_id = info["case_id"]
        score = info["score"]
        
        # Simple weighting: current beliefs = 2x, historical = 1x
        weight = 2.0 if belief in current_belief_matches else 1.0
        weighted_score = score * weight
        
        if case_id not in case_scores:
            case_scores[case_id] = 0
        case_scores[case_id] += weighted_score
    
    # Step 6: Select top cases
    matched_cases = []
    for case_id, score in sorted(case_scores.items(), key=lambda x: x[1], reverse=True):
        if score >= 1.0 and len(matched_cases) < 3:
            matched_cases.append(case_id)
    
    if not matched_cases:
        return "", set()
    
    # Step 7: Build context
    context = ""
    matched_case_ids = set()
    
    for case_id in matched_cases:
        case_data = load_case_file(case_id)
        if not case_data:
            continue
            
        matched_case_ids.add(case_id)
        
        # Get solutions for this case
        next_solutions = get_next_solutions(case_id, session)
        
        # Get all beliefs that match this case
        case_beliefs = [belief for belief, info in all_beliefs.items() if info["case_id"] == case_id]
        
        # Build context section
        context += f"\n## Clinical Pattern: {case_id}\n"
        
        if case_beliefs:
            context += "User's thought patterns:\n"
            for belief in case_beliefs:
                context += f"- \"{belief}\"\n"
            context += "\n"
        
        # Add appropriate solutions
        if next_solutions["immediate"]:
            context += "Immediate solutions to consider:\n"
            for solution in next_solutions["immediate"]:
                if solution.strip() and solution != "**":
                    context += f"- {solution}\n"
        elif next_solutions["intermediate"]:
            context += "Intermediate solutions to consider:\n"
            for solution in next_solutions["intermediate"]:
                if solution.strip() and solution != "**":
                    context += f"- {solution}\n"
        elif next_solutions["long_term"]:
            context += "Long-term solutions to consider:\n"
            for solution in next_solutions["long_term"]:
                if solution.strip() and solution != "**":
                    context += f"- {solution}\n"
        
        # Add therapeutic insight
        if case_data.get("motivational_closure"):
            context += f"\nTherapeutic insight: {case_data['motivational_closure']}\n"
            
        context += "\n"
    
    return context, matched_case_ids

def analyze_solution_delivery(response_text: str, case_ids: Set[str], llm_client) -> Dict[str, Dict[str, bool]]:
    """
    Analyze which solutions were delivered in the response.
    
    Args:
        response_text: The assistant's response text
        case_ids: Set of case IDs that were used for guidance
        llm_client: The LLM client for analysis
        
    Returns:
        Dictionary mapping case IDs to solution delivery status
    """
    if not case_ids:
        return {}
    
    # Build solutions context for analysis
    solutions_context = ""
    for case_id in case_ids:
        case_data = load_case_file(case_id)
        if not case_data or "solutions" not in case_data:
            continue
            
        solutions_context += f"\nCase {case_id} Solutions:\n"
        
        if "immediate" in case_data["solutions"]:
            solutions_context += "Immediate solutions:\n"
            for solution in case_data["solutions"]["immediate"]:
                solutions_context += f"- {solution}\n"
                
        if "intermediate" in case_data["solutions"]:
            solutions_context += "Intermediate solutions:\n"
            for solution in case_data["solutions"]["intermediate"]:
                solutions_context += f"- {solution}\n"
                
        if "long_term" in case_data["solutions"]:
            solutions_context += "Long-term solutions:\n"
            for solution in case_data["solutions"]["long_term"]:
                solutions_context += f"- {solution}\n"
    
    # Build analysis prompt
    analysis_prompt = f"""
    Analyze if the following therapeutic response addressed any of the clinical solutions.
    
    Response:
    {response_text}
    
    Clinical Solutions:
    {solutions_context}
    
    For each case, determine if immediate, intermediate, or long-term solutions were delivered.
    Return your analysis as a JSON object with the following format:
    
    {{
        "case_id1": {{
            "immediate": true/false,
            "intermediate": true/false,
            "long_term": true/false
        }},
        "case_id2": {{
            "immediate": true/false,
            "intermediate": true/false,
            "long_term": true/false
        }}
    }}
    """
    
    try:
        delivery_analysis = llm_client.send_prompt(analysis_prompt, extract_json=True)
        
        # Handle case where LLM returns boolean instead of dict
        if isinstance(delivery_analysis, bool):
            print(f"Warning: LLM returned boolean instead of dict: {delivery_analysis}")
            return {}
        
        # Handle case where LLM returns None
        if delivery_analysis is None:
            print("Warning: LLM returned None for solution delivery analysis")
            return {}
        
        # Handle case where LLM returns string instead of dict
        if isinstance(delivery_analysis, str):
            print(f"Warning: LLM returned string instead of dict: {delivery_analysis}")
            return {}
        
        # Ensure we have a dictionary
        if not isinstance(delivery_analysis, dict):
            print(f"Warning: LLM returned unexpected type: {type(delivery_analysis)}")
            return {}
        
        # Validate the structure of returned dictionary
        validated_analysis = {}
        for case_id in case_ids:
            if case_id in delivery_analysis:
                case_data = delivery_analysis[case_id]
                if isinstance(case_data, dict):
                    validated_analysis[case_id] = {
                        "immediate": case_data.get("immediate", False),
                        "intermediate": case_data.get("intermediate", False), 
                        "long_term": case_data.get("long_term", False)
                    }
                else:
                    print(f"Warning: Invalid case data for {case_id}: {case_data}")
                    validated_analysis[case_id] = {
                        "immediate": False,
                        "intermediate": False,
                        "long_term": False
                    }
        
        return validated_analysis
        
    except Exception as e:
        print(f"Error analyzing solution delivery: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    symptoms = ["anxiety", "panic attacks"]
    user_message = "I feel like I'm never going to get better. Everything I try fails."
    
    class MockSession:
        def __init__(self):
            self.matched_limiting_beliefs = {}
            self.delivered_solutions = {}
            
        def update_limiting_belief_matches(self, belief_matches):
            for belief, info in belief_matches.items():
                self.matched_limiting_beliefs[belief] = info
    
    session = MockSession()
    
    context, matched_cases = fetch_clinical_cases(symptoms, user_message, session)
    print("\nSystem Context:")
    print(context)
    print("\nMatched Cases:", matched_cases)