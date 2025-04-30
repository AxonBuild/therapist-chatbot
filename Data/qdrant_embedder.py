import os
import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import glob

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "EkoMindAI"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536
BATCH_SIZE = 10  # Reduced batch size for better diagnostics
DEBUG = False  # Set to True for more verbose output

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

def load_guidance_notes(directory: str) -> List[Dict[str, Any]]:
    """Load all JSON files from the Guidance Notes directory."""
    all_entries = []
    file_paths = glob.glob(os.path.join(directory, "*.json"))
    
    print(f"Found {len(file_paths)} JSON files in {directory}")
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = os.path.basename(file_path)
            print(f"Processing {filename}: found {len(data.get('keywords', []))} keywords")
            
            # Create an entry for each keyword
            for keyword in data.get("keywords", []):
                entry = {
                    "keyword": keyword,
                    "filename": filename,
                }
                all_entries.append(entry)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Created {len(all_entries)} total entries from all files")
    return all_entries

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
        
        # Try to create payload indexes - version agnostic approach
        try:
            print("Creating payload indexes...")
            
            # Try with string syntax (older versions)
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="keyword",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="filename",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
                print("Created indexes for 'keyword' and 'filename' fields")
            except Exception as e1:
                print(f"Couldn't create index with string syntax: {e1}")
                
                # Try with enum syntax (newer versions)
                try:
                    from qdrant_client.http.models import PayloadSchemaType
                    
                    qdrant_client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name="keyword",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    
                    qdrant_client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name="filename",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    
                    print("Created indexes using PayloadSchemaType")
                except Exception as e2:
                    print(f"Couldn't create index with enum syntax either: {e2}")
        
        except Exception as e:
            print(f"Warning: Failed to create any indexes: {e}")
        
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
        
        # Verify embedding dimension
        if len(embedding) != VECTOR_SIZE:
            print(f"Warning: Expected embedding size {VECTOR_SIZE}, got {len(embedding)}")
        
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise  # Re-raise to halt processing if embeddings fail

def process_and_upload_entry(entry: Dict[str, Any], idx: int) -> bool:
    """Process and upload a single keyword entry with detailed error reporting."""
    try:
        # Generate embedding for the keyword
        keyword = entry["keyword"]
        print(f"  Generating embedding for keyword: '{keyword}' from {entry['filename']}")
        
        embedding = embed_text(keyword)
        
        # Print first few values of embedding for diagnostic purposes
        if DEBUG:
            print(f"  Embedding first 5 values: {embedding[:5]}")
            print(f"  Embedding is list: {isinstance(embedding, list)}")
            print(f"  Embedding length: {len(embedding)}")
        
        # Create point for Qdrant
        point = models.PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "keyword": keyword,
                "filename": entry["filename"]
            }
        )
        
        # Upload to Qdrant
        print(f"  Uploading point {idx} to Qdrant")
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        # Verify the point was added with the correct vector
        retrieved_point = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[idx],
            with_vectors=True
        )
        
        if not retrieved_point:
            print(f"  Warning: Point {idx} was not found after upload")
            return False
            
        if DEBUG:
            print(f"  Retrieved point: {retrieved_point[0].id}")
            if hasattr(retrieved_point[0], 'vector') and retrieved_point[0].vector:
                print(f"  Vector exists and first 5 values: {retrieved_point[0].vector[:5]}")
            else:
                print(f"  Warning: Retrieved point has no vector")
        
        print(f"  Successfully uploaded keyword '{keyword}' from {entry['filename']}")
        return True
        
    except Exception as e:
        print(f"Error processing entry {idx}: {e}")
        return False

def main():
    # Directory configuration
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    guidance_notes_dir = os.path.join("Data", "Guidance Notes")
    
    # Make sure the directory exists
    if not os.path.exists(guidance_notes_dir):
        print(f"Directory '{guidance_notes_dir}' does not exist. Creating it.")
        os.makedirs(guidance_notes_dir)
    
    # Load entries from all JSON files
    entries = load_guidance_notes(guidance_notes_dir)
    if not entries:
        print("No entries to process. Exiting.")
        return
    
    # Delete existing collection for a fresh start
    delete_collection_if_exists()
    
    # Create new collection
    if not create_collection():
        print("Failed to create collection. Exiting.")
        return
    
    # Process a small subset for testing if in debug mode
    if DEBUG:
        print("Debug mode: Processing only first few entries")
        entries = entries[:min(5, len(entries))]
    
    # Process and upload entries one by one for better diagnostics
    print(f"Processing {len(entries)} keyword entries...")
    success_count = 0
    
    for i, entry in enumerate(entries):
        print(f"\nProcessing entry {i+1}/{len(entries)}")
        if process_and_upload_entry(entry, i):
            success_count += 1
        
        # Small delay between entries
        if i < len(entries) - 1:
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
    
    # Try a basic search to verify functionality
    try:
        print("\nTesting search functionality...")
        test_query = "anxiety"
        print(f"Testing search with query: '{test_query}'")
        
        # First embed the query
        query_vector = embed_text(test_query)
        
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        print(f"Search test returned {len(results)} results")
        
        if results:
            for i, result in enumerate(results):
                print(f"Result {i+1}: '{result.payload.get('keyword')}' from {result.payload.get('filename')} (Score: {result.score:.4f})")
    except Exception as e:
        print(f"Error testing search: {e}")
    
    print(f"\nProcess complete. Successfully uploaded {success_count}/{len(entries)} keyword entries.")

if __name__ == "__main__":
    main()