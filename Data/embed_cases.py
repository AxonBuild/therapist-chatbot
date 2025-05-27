import os
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import glob
import re

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "ClinicalCases"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536
BATCH_SIZE = 10
DEBUG = True

# Configuration
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

qdrant_url = os.environ.get("QDRANT_URL")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")

print(f"Connecting to Qdrant at: {qdrant_url}")

# Initialize clients
openai_client = OpenAI(api_key=openai_api_key)
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

def validate_json_files(directory: str) -> bool:
    """Validate all JSON files in the directory to ensure they have the required structure."""
    file_paths = glob.glob(os.path.join(directory, "*.json"))
    print(f"Validating {len(file_paths)} JSON files in {directory}")
    
    all_valid = True
    required_fields = [
        "case_id", "emotions", "limiting_beliefs", "tags", "solutions"
    ]
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = os.path.basename(file_path)
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"ERROR: {filename} is missing required fields: {missing_fields}")
                all_valid = False
                continue
                
            # Validate solutions structure
            if not isinstance(data["solutions"], dict) or not all(k in data["solutions"] for k in ["immediate", "intermediate", "long_term"]):
                print(f"ERROR: {filename} has invalid solutions structure. Expected dict with immediate, intermediate, long_term keys")
                all_valid = False
                continue
                
            # Validate emotions and tags are lists
            if not isinstance(data["emotions"], list) or not isinstance(data["tags"], list):
                print(f"ERROR: {filename} has invalid emotions or tags. Both should be lists")
                all_valid = False
                continue
                
            # Validate limiting beliefs is a list
            if not isinstance(data["limiting_beliefs"], list):
                print(f"ERROR: {filename} has invalid limiting_beliefs. Should be a list")
                all_valid = False
                continue
                
            print(f"✓ {filename} is valid")
                
        except json.JSONDecodeError:
            print(f"ERROR: {file_path} contains invalid JSON")
            all_valid = False
        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            all_valid = False
    
    return all_valid

def load_clinical_cases(directory: str) -> List[Dict[str, Any]]:
    """Load all JSON files from the Clinical Cases directory."""
    all_cases = []
    file_paths = glob.glob(os.path.join(directory, "*.json"))
    
    print(f"Found {len(file_paths)} JSON files in {directory}")
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Store the case data
            all_cases.append(data)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(all_cases)} clinical cases")
    return all_cases

def prepare_embeddings(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract all items to embed from clinical cases."""
    embeddings_to_create = []
    
    for case in cases:
        case_id = case["case_id"]
        
        # Process emotions (as keywords)
        for emotion in case.get("emotions", []):
            embeddings_to_create.append({
                "case_id": case_id,
                "content_type": "keyword",
                "keyword_type": "emotion",
                "original_text": emotion.strip()
            })
        
        # Process tags (as keywords)
        for tag in case.get("tags", []):
            embeddings_to_create.append({
                "case_id": case_id,
                "content_type": "keyword",
                "keyword_type": "tag",
                "original_text": tag.strip()
            })
        
        # Process limiting beliefs
        for belief in case.get("limiting_beliefs", []):
            # Clean up the belief text (remove quotes, etc.)
            belief_text = belief.strip()
            belief_text = re.sub(r'^[\"\'\-\s]+|[\"\'\s]+$', '', belief_text)
            
            embeddings_to_create.append({
                "case_id": case_id,
                "content_type": "limiting_belief",
                "original_text": belief_text
            })
    
    print(f"Prepared {len(embeddings_to_create)} items to embed")
    return embeddings_to_create

def delete_collection_if_exists():
    """Delete the collection if it exists for a fresh start."""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME in collection_names:
            print(f"Deleting existing collection '{COLLECTION_NAME}'")
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' deleted")
            # Small delay to ensure deletion is processed
            time.sleep(2)
    except Exception as e:
        print(f"Error checking/deleting collection: {e}")

def create_collection():
    """Create Qdrant collection."""
    try:
        print(f"Creating collection '{COLLECTION_NAME}'")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        
        # Try to create payload indexes
        try:
            print("Creating payload indexes...")
            from qdrant_client.http.models import PayloadSchemaType
            
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="case_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="content_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="keyword_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            print("Created payload indexes")
        except Exception as e:
            print(f"Warning: Failed to create indexes: {e}")
        
        # Verify collection was created
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME in collection_names:
            print(f"Collection '{COLLECTION_NAME}' created successfully")
            return True
        else:
            print(f"Failed to create collection '{COLLECTION_NAME}'")
            return False
            
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False

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

def process_and_upload_entry(entry: Dict[str, Any], idx: int) -> bool:
    """Process and upload a single entry with detailed error reporting."""
    try:
        # Generate embedding
        original_text = entry["original_text"]
        content_type = entry["content_type"]
        
        print(f"  Generating embedding for {content_type}: '{original_text}' from {entry['case_id']}")
        
        embedding = embed_text(original_text)
        
        # Create payload
        payload = {
            "case_id": entry["case_id"],
            "content_type": content_type,
            "original_text": original_text
        }
        
        # Add keyword_type if applicable
        if "keyword_type" in entry:
            payload["keyword_type"] = entry["keyword_type"]
        
        # Create point for Qdrant
        point = models.PointStruct(
            id=idx,
            vector=embedding,
            payload=payload
        )
        
        # Upload to Qdrant
        print(f"  Uploading point {idx} to Qdrant")
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        print(f"  Successfully uploaded {content_type}: '{original_text}'")
        return True
        
    except Exception as e:
        print(f"Error processing entry {idx}: {e}")
        return False

def main():
    # Directory configuration
    cases_dir = os.path.join("cases")
    
    # Make sure the directory exists
    if not os.path.exists(cases_dir):
        print(f"Directory '{cases_dir}' does not exist. Exiting.")
        return
    
    # Validate JSON files first
    if not validate_json_files(cases_dir):
        print("Some JSON files are invalid. Please fix them before proceeding.")
        return
    
    # Load cases
    cases = load_clinical_cases(cases_dir)
    if not cases:
        print("No cases to process. Exiting.")
        return
    
    # Prepare embeddings
    embeddings = prepare_embeddings(cases)
    if not embeddings:
        print("No embeddings to create. Exiting.")
        return
    
    # Delete existing collection for a fresh start
    #delete_collection_if_exists()
    
    # Create new collection
    if not create_collection():
        print("Failed to create collection. Exiting.")
        return
    
    # Process a small subset for testing if in debug mode
    if DEBUG:
        print("Debug mode: Processing only first few entries")
        embeddings = embeddings[:min(100, len(embeddings))]
    
    # Process and upload entries
    print(f"Processing {len(embeddings)} embeddings...")
    success_count = 0
    
    for i, entry in enumerate(embeddings):
        print(f"\nProcessing entry {i+1}/{len(embeddings)}")
        if process_and_upload_entry(entry, i):
            success_count += 1
        
        # Small delay between entries to avoid rate limits
        if i < len(embeddings) - 1:
            time.sleep(1)
    
    # Verify final counts
    try:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        vectors_count = collection_info.vectors_count
        print(f"\nCollection '{COLLECTION_NAME}' now contains {vectors_count} vectors")
        
        if vectors_count != success_count:
            print(f"Warning: Expected {success_count} vectors, but found {vectors_count}")
    except Exception as e:
        print(f"Error getting collection info: {e}")
    
    # Test search functionality
    try:
        print("\nTesting search functionality...")
        
        # Test keyword search
        test_keyword = "Inadequacy"
        print(f"Testing keyword search with: '{test_keyword}'")
        keyword_vector = embed_text(test_keyword)
        
        keyword_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=keyword_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_type",
                        match=models.MatchValue(value="keyword")
                    )
                ]
            ),
            limit=3
        )
        
        print(f"Keyword search returned {len(keyword_results)} results:")
        for i, result in enumerate(keyword_results):
            print(f"  Result {i+1}: '{result.payload.get('original_text')}' from {result.payload.get('case_id')} (Score: {result.score:.4f})")
        
        # Test limiting belief search
        test_belief = "I am not good enough"
        print(f"\nTesting limiting belief search with: '{test_belief}'")
        belief_vector = embed_text(test_belief)
        
        belief_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=belief_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_type",
                        match=models.MatchValue(value="limiting_belief")
                    )
                ]
            ),
            limit=3
        )
        
        print(f"Limiting belief search returned {len(belief_results)} results:")
        for i, result in enumerate(belief_results):
            print(f"  Result {i+1}: '{result.payload.get('original_text')}' from {result.payload.get('case_id')} (Score: {result.score:.4f})")
            
    except Exception as e:
        print(f"Error testing search: {e}")
    
    print(f"\nProcess complete. Successfully uploaded {success_count}/{len(embeddings)} embeddings to Qdrant.")

if __name__ == "__main__":
    main()