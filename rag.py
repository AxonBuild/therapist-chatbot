import json
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "EkoMindAI"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

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
        
        # Verify embedding dimension
        if len(embedding) != VECTOR_SIZE:
            print(f"Warning: Expected embedding size {VECTOR_SIZE}, got {len(embedding)}")
        
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise  # Re-raise to halt processing if embeddings fail

 # Try a basic search to verify functionality

def fetch_guidance_notes(symptoms: List[str], session=None):
    """Fetch guidance notes based on symptoms and include delivery status if session provided."""
    requests = [
        SearchRequest(
            vector=embed_text(symptom),
            limit=2,
            score_threshold=0.6,
            with_payload=True
        )    
        for symptom in symptoms
    ]
    
    response = qdrant_client.search_batch(
        collection_name=COLLECTION_NAME,
        requests=requests
    )
    
    flattened_response = []
    for result in response:
        flattened_response.extend(result)
    
    print("Matched keywords:")
    
    note_files = set()
    for result in flattened_response:
        print("-", result.payload.get("keyword"), result.score)
        note_files.add(os.path.join("Data", "Guidance Notes", result.payload.get("filename")))
    
    print("Note Files to load:", note_files)
    
    context = ""
    for note_file in note_files:
        try:
            data = json.load(open(note_file, "r", encoding="utf-8"))
            themes = '\n'.join(data["themes"])
            
            # Get primary and secondary content from updated JSON structure
            primary_content = data.get("primary_content", "")
            secondary_content = data.get("secondary_content", "")
            
            # Check delivery status if session provided
            primary_status = "Pending"
            secondary_status = "Pending"
            
            if session and hasattr(session, "delivered_guidance"):
                filename = os.path.basename(note_file)
                if filename in session.delivered_guidance:
                    primary_status = "Delivered" if session.delivered_guidance[filename]["primary"] else "Pending"
                    secondary_status = "Delivered" if session.delivered_guidance[filename]["secondary"] else "Pending"
            
            # Format with status
            context += f"""
- For symptoms like:
{themes}

Primary content: {primary_content}
Status: {primary_status}

Secondary content: {secondary_content}
Status: {secondary_status}

"""
        except Exception as e:
            print(f"Error processing note file {note_file}: {e}")
    
    return context

if __name__ == "__main__":
    # Example usage
    symptoms = ["anxiety", "depression"]
    context = fetch_guidance_notes(symptoms)
    print("Context:", context)