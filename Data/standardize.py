import os
import re
import json
import glob
import docx
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("Please set OPENROUTER_API_KEY in your .env file")

# Initialize OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# Constants
MODEL = "openai/gpt-4o-mini"  # Using GPT-4o mini via OpenRouter
MAX_TOKENS = 4096
TEMPERATURE = 0.3  # Lower temperature for more consistent formatting

# Standardized section headers we want in all clinical cases
STANDARD_SECTIONS = [
    "Demographics",
    "Summary of Issues",
    "Dominant Emotions",
    "Triggers and Mechanisms",
    "Limiting Beliefs",
    "Proposed Solutions",
    "Progress Indicators"
]

def extract_text_from_docx(file_path: str) -> str:
    """Extract text content from a .docx file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def read_text_file(file_path: str) -> str:
    """Read content from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_content(file_path: str) -> str:
    """Extract content from various file formats."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return read_text_file(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return ""

def is_clinical_case(content: str, filename: str) -> bool:
    """Determine if a file is likely a clinical case."""
    # Check filename first
    if "clinical" in filename.lower() or "case" in filename.lower():
        return True
    
    # Check for common clinical case indicators in content
    indicators = [
        r"age and gender", 
        r"summary of (?:identified )?issues",
        r"dominant emotions",
        r"diagnosis",
        r"symptoms",
        r"treatment",
        r"patient",
        r"\b(?:male|female),? \d+(?:\s+years?)?"
    ]
    
    for indicator in indicators:
        if re.search(indicator, content, re.IGNORECASE):
            return True
    
    return False

def standardize_clinical_case(content: str, disorder_type: str = None) -> str:
    """
    Use GPT to standardize the clinical case format.
    """
    prompt = f"""
    # Task: Format Clinical Case Document

    ## Context
    I have a clinical case document that needs to be formatted in a standardized way for a RAG (Retrieval Augmented Generation) system. The document contains clinical information about a patient with a psychological or mental health condition.

    ## Input Format
    The document may have inconsistent formatting, section titles, and organization.

    ## Required Output Format
    Please reformat the document using these EXACT section headers in this order:

    1. Demographics - Include age, gender, and any other demographic details provided
    2. Summary of Issues - Main problems, symptoms, or conditions the patient is experiencing
    3. Dominant Emotions - Key emotional states described or implied
    4. Triggers and Mechanisms - What causes or worsens the condition and underlying psychological mechanisms
    5. Limiting Beliefs - Identified negative beliefs or thought patterns
    6. Proposed Solutions - Therapeutic approaches, treatments, or recommendations
    7. Progress Indicators - How progress would be measured or tracked

    ## Requirements:
    - PRESERVE ALL original clinical information - do not fabricate or add new clinical details
    - Use the EXACT section headers specified above
    - If information for a specific section isn't present in the original, include the section header and write "No information provided" below it
    - Maintain professional, clinical language
    - Format as plain text with clear section separation
    - If the disorder type is evident, mention it in the Summary of Issues section

    ## Original Document:
    {content}
    """
    
    if disorder_type:
        prompt += f"\n\nNote: This case appears to be related to {disorder_type}. Please ensure this is reflected in the Summary of Issues section if appropriate."
    
    try:
        # Using OpenRouter client to access GPT-4o-mini
        completion = openrouter_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        standardized_content = completion.choices[0].message.content
        return standardized_content
    
    except Exception as e:
        print(f"Error in API call: {e}")
        return None

def save_standardized_case(content: str, output_path: str) -> None:
    """Save standardized content to output file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Saved standardized case to {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")

def main():
    # Directory configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "standardized_cases")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all disorder folders
    processed_count = 0
    
    # Get all subdirectories in the current directory (disorder types)
    disorder_dirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
    
    for disorder_dir in disorder_dirs:
        # Skip the output directory if it exists
        if disorder_dir == "standardized_cases":
            continue
            
        disorder_type = disorder_dir.lower()
        disorder_path = os.path.join(current_dir, disorder_dir)
        print(f"Processing disorder: {disorder_type}")
        
        # Create corresponding output subdirectory
        disorder_output_dir = os.path.join(output_dir, disorder_type)
        os.makedirs(disorder_output_dir, exist_ok=True)
        
        # Find all potential clinical case files in this disorder directory
        file_patterns = ['*.txt', '*.docx']
        disorder_files = []
        for pattern in file_patterns:
            disorder_files.extend(glob.glob(os.path.join(disorder_path, pattern)))
        
        # Process each file in this disorder directory
        for file_path in disorder_files:
            filename = os.path.basename(file_path)
            print(f"  Processing {filename}...")
            
            # Extract content
            content = extract_content(file_path)
            if not content:
                print(f"    Skipping {filename} - could not extract content")
                continue
            
            # Check if it's a clinical case
            if not is_clinical_case(content, filename):
                print(f"    Skipping {filename} - not identified as a clinical case")
                continue
            
            print(f"    Using folder name for disorder type: {disorder_type}")
            
            # Standardize content
            standardized_content = standardize_clinical_case(content, disorder_type)
            if not standardized_content:
                print(f"    Error standardizing {filename}")
                continue
            
            # Generate output filename
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_standardized.txt"
            output_path = os.path.join(disorder_output_dir, output_filename)
            
            # Save standardized content
            save_standardized_case(standardized_content, output_path)
            processed_count += 1
            
            # Rate limiting for API calls
            if len(disorder_files) > 1:
                print("    Waiting for API rate limit...")
                time.sleep(2)  # Adjust based on API rate limits
    
    print(f"Processed {processed_count} clinical case files across {len(disorder_dirs) - (1 if 'standardized_cases' in disorder_dirs else 0)} disorder directories")

if __name__ == "__main__":
    main()