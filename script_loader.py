import os
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

class ScriptLoader:
    """Handles script selection based on symptoms and user profile using pandas"""
    
    def __init__(self, csv_path=None):
        """Initialize with path to CSV file containing script data"""
        if not csv_path:
            # Use the default CSV path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, "Scripts_Summary_Table_V1.csv")
        
        self.csv_path = csv_path
        self.df = self._load_scripts()
    
    def _load_scripts(self) -> pd.DataFrame:
        """Load script data from CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(df)} scripts from {self.csv_path}")
            return df
        except Exception as e:
            print(f"Error loading scripts: {e}")
            return pd.DataFrame()
    
    def find_matching_script(self, symptoms: List[str], user_profile: Dict[str, str], exclude_scripts: set[str] = None) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Find the best matching script based on symptoms and user profile
        
        Args:
            symptoms: List of identified symptoms/keywords
            user_profile: Dictionary with 'age_group' and 'emotional_intensity'
            exclude_scripts: Set of script IDs/filenames to exclude from selection
            
        Returns:
            Tuple of (script dictionary or None, match score)
        """
        if self.df.empty:
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
        
        # Create a copy of the dataframe to add scoring column
        scored_df = self.df.copy()
        scored_df['match_score'] = 0
        
        # Score based on symptoms matching keywords
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            # Check for keyword matches
            scored_df['match_score'] += scored_df['Trigger Keywords'].fillna('').str.lower().apply(
                lambda keywords: 3 if any(kw.strip() in symptom_lower or symptom_lower in kw.strip() 
                                      for kw in keywords.split(',') if kw.strip()) else 0
            )
            
            # Check for tag matches
            scored_df['match_score'] += scored_df['Tags'].fillna('').str.lower().apply(
                lambda tags: 2 if any(tag.replace('#', '').strip() in symptom_lower or 
                                   symptom_lower in tag.replace('#', '').strip() 
                                   for tag in tags.split() if tag.strip()) else 0
            )
        
        # Level match (high priority)
        if target_level:
            scored_df.loc[scored_df['Level'] == target_level, 'match_score'] += 5
        
        # Population match (high priority)
        if target_population:
            scored_df.loc[scored_df['Target Population'] == target_population, 'match_score'] += 5
        
        # Filter out already offered scripts
        if exclude_scripts:
            print(f"Excluding {len(exclude_scripts)} already offered scripts: {exclude_scripts}")
            # Filter by both Filename and Script ID columns if they exist
            if 'Filename' in scored_df.columns:
                scored_df = scored_df[~scored_df['Filename'].isin(exclude_scripts)]
            if 'Script ID' in scored_df.columns:
                scored_df = scored_df[~scored_df['Script ID'].isin(exclude_scripts)]
            print(f"Remaining scripts after filtering: {len(scored_df)}")
        
        # Sort by score and get the best match
        scored_df = scored_df.sort_values('match_score', ascending=False)
        
        if scored_df.empty or scored_df.iloc[0]['match_score'] <= 0:
            print("No matching script found")
            return None, 0
        
        # Get the best match
        best_match = scored_df.iloc[0].to_dict()
        best_score = int(best_match.get('match_score', 0))
        
        print(f"Selected script: {best_match.get('New Title')} (Score: {best_score})")
        
        return best_match, best_score
    
    def get_script_content(self, script_id: str) -> str:
        """Get script content based on ID or filename"""
        if self.df.empty:
            return f"Script not found: {script_id}"
            
        # Find the matching script
        script = None
        if 'Filename' in self.df.columns:
            matches = self.df[self.df['Filename'] == script_id]
            if not matches.empty:
                script = matches.iloc[0].to_dict()
                
        if script is None and 'Script ID' in self.df.columns:
            matches = self.df[self.df['Script ID'] == script_id]
            if not matches.empty:
                script = matches.iloc[0].to_dict()
        
        if script is None:
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