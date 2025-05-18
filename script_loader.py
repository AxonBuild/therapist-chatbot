import json
import os
from typing import Dict, List, Any, Optional, Tuple

class ScriptLoader:
    """Handles script selection based on symptoms and user profile"""
    
    def __init__(self, json_path=None):
        """Initialize with path to JSON file containing script data"""
        if not json_path:
            # Default path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "Scripts_Summary_Table_V1.json")
        
        self.json_path = json_path
        self.scripts_data = self._load_scripts()
    
    def _load_scripts(self) -> List[Dict[str, Any]]:
        """Load and parse the scripts JSON file"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                scripts = json.load(f)
            print(f"Loaded {len(scripts)} scripts from {self.json_path}")
            return scripts
        except Exception as e:
            print(f"Error loading scripts: {e}")
            return []
    
    def find_matching_script(self, symptoms: List[str], user_profile: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Find the best matching script based on symptoms and user profile
        
        Args:
            symptoms: List of identified symptoms/keywords
            user_profile: Dictionary with 'age_group' and 'emotional_intensity'
            
        Returns:
            Tuple of (script dictionary or None, match score)
        """
        if not self.scripts_data:
            return None, 0
            
        # Extract profile information
        age_group = user_profile.get("age_group", "").lower()
        emotional_intensity = user_profile.get("emotional_intensity", "").lower()
        
        print(f"Finding script for age: {age_group}, intensity: {emotional_intensity}")
        print(f"Symptoms: {symptoms}")
        
        # Map emotional intensity to level
        level_mapping = {
            "mild": "Mild",
            "moderate": "Moderate", 
            "intense": "Intense"
        }
        target_level = level_mapping.get(emotional_intensity, None)
        
        # Map age groups to target population
        population_mapping = {
            "child": "Child",
            "teen": "Teen",
            "adult": "Adult",
            "senior": "Adult"  # Default seniors to adult content
        }
        target_population = population_mapping.get(age_group, None)
        
        # Calculate match scores for each script
        best_match = None
        best_score = -1
        
        for script in self.scripts_data:
            score = 0
            
            # Check keyword matches
            trigger_keywords = []
            if "Trigger Keywords" in script:
                trigger_keywords = [kw.strip().lower() for kw in script["Trigger Keywords"].split(",")]
                
            for symptom in symptoms:
                symptom_lower = symptom.lower()
                # Exact match or partial match in trigger keywords
                for keyword in trigger_keywords:
                    if keyword in symptom_lower or symptom_lower in keyword:
                        score += 3
                        break
            
            # Check tag matches
            tags = []
            if "Tags" in script:
                # Split tags by '#' and remove empty entries
                tags_str = script["Tags"].replace("#", " ")
                tags = [tag.strip().lower() for tag in tags_str.split() if tag.strip()]
                
            for symptom in symptoms:
                symptom_lower = symptom.lower()
                for tag in tags:
                    if tag in symptom_lower or symptom_lower in tag:
                        score += 2
                        break
            
            # Level match (high priority)
            if target_level and script.get("Level") == target_level:
                score += 5
            
            # Population match (high priority)
            if target_population and script.get("Target Population") == target_population or "Adult" in script.get("Target Population", ""):
                score += 5
                
            # Update best match if we found a better score
            if score > best_score:
                best_score = score
                best_match = script
        
        if best_match:
            print(f"Selected script: {best_match.get('New Title')} (Score: {best_score})")
        else:
            print("No matching script found")
            
        return best_match, best_score
    
    def get_script_content(self, script_id: str) -> str:
        """Get script content based on ID or filename"""
        # Find the matching script
        script = None
        for s in self.scripts_data:
            if s.get("Filename") == script_id or s.get("Script ID") == script_id:
                script = s
                break
        
        if not script:
            return f"Script not found: {script_id}"
        
        # Generate content from metadata since we don't have the actual content files
        title = script.get("New Title", "Therapeutic Exercise")
        summary = script.get("Summary", "")
        level = script.get("Level", "")
        target = script.get("Target Population", "")
        technique = script.get("Primary Therapeutic Technique", "")
        duration = script.get("Estimated Duration (min)", "")
        
        content = f"""# {title}

## Summary
{summary}

## Details
- Level: {level}
- Target: {target}
- Technique: {technique}
- Duration: {duration} minutes

*Note: This is a placeholder for the actual script content. In the production version, the complete therapeutic script will be displayed here.*
"""
        return content