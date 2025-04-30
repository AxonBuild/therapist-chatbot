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

def fetch_guidance_notes(symptoms: List[str]):
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
        note_files.add(os.path.join("Data", "Guidance Notes", result.payload.get("filename") ))
    
    print("Note Files to load:", note_files)
    
    context = ""
    for note_file in note_files:
        data = json.load(open(note_file, "r", encoding="utf-8"))
        themes = '\n'.join(data["themes"])
        context += f"""
- For symptoms like:
{themes}

{data["context"]}

      
"""
    
    return context
