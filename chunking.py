import os
import re
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

# Constants
STANDARD_SECTIONS = [
    "Demographics",
    "Summary of Issues",
    "Dominant Emotions",
    "Triggers and Mechanisms",
    "Limiting Beliefs",
    "Proposed Solutions",
    "Progress Indicators"
]

def read_text_file(file_path: str) -> str:
    """Read content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text content from a .docx file."""
    try:
        import docx
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
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

def normalize_section_headings(content: str) -> str:
    """Normalize section headings to a consistent format."""
    normalized = content
    
    # Find and normalize each standard section heading
    for section in STANDARD_SECTIONS:
        # Different heading patterns that might appear in standardized files
        patterns = [
            # "Demographics:" format
            rf"(?:^|\n)({section}):\s*\n",
            # "Demographics" format (standalone)
            rf"(?:^|\n)({section})\s*\n",
            # "1. Demographics" format
            rf"(?:^|\n)\d+\.\s*({section})[\s:]*\n",
            # "Demographics -" format
            rf"(?:^|\n)({section})\s*-\s*\n",
            # "## Demographics" format (markdown)
            rf"(?:^|\n)##\s*({section})[\s:]*\n"
        ]
        
        # Replace all variations with a standardized format
        for pattern in patterns:
            normalized = re.sub(pattern, f"\n\n## {section}\n\n", normalized, flags=re.IGNORECASE)
    
    return normalized

def parse_standardized_case(file_path: str) -> Dict[str, Any]:
    """Parse a standardized clinical case file and extract sections."""
    content = read_text_file(file_path)
    if not content:
        return None
    
    # Normalize section headings to a consistent format
    normalized_content = normalize_section_headings(content)
    
    # Extract document-level information
    file_name = os.path.basename(file_path)
    document_id = os.path.splitext(file_name)[0].replace('_standardized', '')
    
    # Extract sections using normalized headings
    sections = {}
    
    # First find all section headers and their positions
    header_positions = []
    for section_name in STANDARD_SECTIONS:
        pattern = rf"\n\n## {section_name}\n\n"
        for match in re.finditer(pattern, normalized_content, re.IGNORECASE):
            header_positions.append((match.end(), section_name))
    
    # Sort positions to establish section boundaries
    header_positions.sort()
    
    # Extract content between headers
    for i, (start_pos, section_name) in enumerate(header_positions):
        if i < len(header_positions) - 1:
            end_pos = header_positions[i+1][0] - 4  # Adjust to remove the "\n\n##" of next section
            section_content = normalized_content[start_pos:end_pos].strip()
        else:
            section_content = normalized_content[start_pos:].strip()
        
        sections[section_name] = section_content
    
    # Check for missing sections
    for section_name in STANDARD_SECTIONS:
        if section_name not in sections:
            sections[section_name] = "No information provided"
    
    return {
        "document_id": document_id,
        "file_name": file_name,
        "sections": sections
    }

def create_chunks_from_case(case_data: Dict[str, Any], disorder: str) -> List[Dict[str, Any]]:
    """Create chunks from a parsed clinical case."""
    chunks = []
    document_id = case_data["document_id"]
    
    for section_name, section_content in case_data["sections"].items():
        if section_content == "No information provided":
            continue
            
        # Create a unique chunk ID
        chunk_id = hashlib.md5(f"{document_id}_{section_name}".encode()).hexdigest()
        
        # Format the content with the section header
        formatted_content = f"# {section_name}\n\n{section_content}"
        
        # Create the chunk
        chunk = {
            "id": chunk_id,
            "document_id": document_id,
            "disorder": disorder,
            "section": section_name,
            "content": formatted_content,
            "metadata": {}  # Empty metadata since tags are removed
        }
        
        chunks.append(chunk)
    
    return chunks

# def create_overview_chunk(case_data: Dict[str, Any], disorder: str) -> Dict[str, Any]:
#     """Create an overview chunk that summarizes the case."""
#     document_id = case_data["document_id"]
    
#     # Extract demographics and summary
#     demographics = case_data["sections"].get("Demographics", "No information provided")
#     summary = case_data["sections"].get("Summary of Issues", "No information provided")
    
#     # Create content
#     content = f"# Case Overview\n\n"
#     content += f"## Demographics\n{demographics}\n\n"
#     content += f"## Summary\n{summary}\n\n"
    
#     # Create a unique chunk ID
#     chunk_id = hashlib.md5(f"{document_id}_overview".encode()).hexdigest()
    
#     # Create the chunk
#     chunk = {
#         "id": chunk_id,
#         "document_id": document_id,
#         "disorder": disorder,
#         "section": "Overview",
#         "content": content,
#         "metadata": {}  # Empty metadata since tags are removed
#     }
    
#     return chunk

def save_debug_info(case_data: Dict[str, Any], output_dir: str, filename: str) -> None:
    """Save debugging information about parsed sections for troubleshooting."""
    debug_path = os.path.join(output_dir, f"{filename}_debug.json")
    
    debug_info = {
        "document_id": case_data["document_id"],
        "section_count": len([s for s, c in case_data["sections"].items() if c != "No information provided"]),
        "sections": {k: (v[:100] + "..." if len(v) > 100 else v) for k, v in case_data["sections"].items()}
    }
    
    with open(debug_path, 'w', encoding='utf-8') as f:
        json.dump(debug_info, f, indent=2, ensure_ascii=False)

def main():
    # Directory configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    standardized_dir = os.path.join(current_dir, "standardized_cases")
    output_dir = os.path.join(current_dir, "chunks")
    debug_dir = os.path.join(output_dir, "debug")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Process all standardized cases
    all_chunks = []
    
    # Get all subdirectories (disorder types)
    disorder_dirs = [d for d in os.listdir(standardized_dir) if os.path.isdir(os.path.join(standardized_dir, d))]
    
    # Process each disorder folder
    for disorder_dir in disorder_dirs:
        disorder_type = disorder_dir.lower()
        disorder_path = os.path.join(standardized_dir, disorder_dir)
        print(f"Processing disorder: {disorder_type}")
        
        # Find all standardized case files in this disorder directory
        case_files = glob.glob(os.path.join(disorder_path, "*_standardized.txt"))
        
        for file_path in case_files:
            filename = os.path.basename(file_path)
            print(f"  Chunking {filename}...")
            
            # Parse the case
            case_data = parse_standardized_case(file_path)
            if not case_data:
                print(f"    Skipping {filename} - could not parse")
                continue
            
            # Save debug information
            save_debug_info(case_data, debug_dir, os.path.splitext(filename)[0])
            
            # Check if sections were correctly parsed
            valid_sections = sum(1 for s, c in case_data["sections"].items() if c != "No information provided")
            if valid_sections < 2:  # At least Demographics and Summary should be present
                print(f"    Warning: Only {valid_sections} sections found in {filename}")
            
            # Create chunks from the case
            case_chunks = create_chunks_from_case(case_data, disorder_type)
            
            # # Create an overview chunk
            # overview_chunk = create_overview_chunk(case_data, disorder_type)
            
            # Add all chunks to the list
            #all_chunks.append(overview_chunk)
            all_chunks.extend(case_chunks)
            
            print(f"    Created {len(case_chunks) + 1} chunks")
    
    # Save all chunks to JSON
    output_path = os.path.join(output_dir, "all_chunks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Chunks saved to: {output_path}")
    
    # Save chunk stats
    stats = {
        "total_chunks": len(all_chunks),
        "disorders": {},
        "sections": {}
    }
    
    for chunk in all_chunks:
        # Count by disorder
        disorder = chunk["disorder"]
        if disorder not in stats["disorders"]:
            stats["disorders"][disorder] = 0
        stats["disorders"][disorder] += 1
        
        # Count by section
        section = chunk["section"]
        if section not in stats["sections"]:
            stats["sections"][section] = 0
        stats["sections"][section] += 1
    
    stats_path = os.path.join(output_dir, "chunk_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Chunk statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()