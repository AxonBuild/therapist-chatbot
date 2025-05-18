import os
import json
import re

def convert_json_files(directory):
    """Convert guidance note JSON files from combined context to separate keys."""
    
    # Pattern to identify sections in the context
    patterns = {
        "primary_content": r"Preferred First Response Strategy:(.*?)(?=Secondary Exploration Strategy:|Relevant Therapeutic Interventions:|Special Cautions or Flags:|$)",
        "secondary_content": r"Secondary Exploration Strategy:(.*?)(?=Preferred First Response Strategy:|Relevant Therapeutic Interventions:|Special Cautions or Flags:|$)",
        "interventions": r"Relevant Therapeutic Interventions:(.*?)(?=Preferred First Response Strategy:|Secondary Exploration Strategy:|Special Cautions or Flags:|$)",
        "cautions": r"Special Cautions or Flags:(.*?)(?=Preferred First Response Strategy:|Secondary Exploration Strategy:|Relevant Therapeutic Interventions:|$)"
    }
    
    # Process each JSON file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            
            # Read existing JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip if already converted
            if 'context' not in data:
                print(f"Skipping {filename} - already converted")
                continue
                
            # Extract content from context field
            context = data.get('context', '')
            
            # Create new structure
            new_data = {
                "themes": data.get("themes", []),
                "keywords": data.get("keywords", [])
            }
            
            # Extract each section using regex
            for key, pattern in patterns.items():
                match = re.search(pattern, context, re.DOTALL)
                content = match.group(1).strip() if match else ""
                new_data[key] = content
            
            # Remove original context field
            if 'context' in new_data:
                del new_data['context']
            
            # Save updated JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)
            
            print(f"Converted {filename}")

# Run the conversion
guidance_notes_dir = os.curdir
convert_json_files(guidance_notes_dir)