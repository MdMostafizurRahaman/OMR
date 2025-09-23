import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

class AnswerKeyManager:
    def __init__(self):
        self.answer_keys_dir = "data/answer_keys"
        os.makedirs(self.answer_keys_dir, exist_ok=True)
    
    def save_answer_key(self, answer_key_data: Dict) -> str:
        """Save answer key and return unique ID"""
        # Generate unique ID
        answer_key_id = str(uuid.uuid4())
        answer_key_data["answer_key_id"] = answer_key_id
        
        # Save to JSON file
        filename = f"{answer_key_id}.json"
        filepath = os.path.join(self.answer_keys_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(answer_key_data, f, indent=2)
        
        return answer_key_id
    
    def get_answer_key(self, answer_key_id: str) -> Optional[Dict]:
        """Get answer key by ID"""
        filepath = os.path.join(self.answer_keys_dir, f"{answer_key_id}.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        return None
    
    def get_all_answer_keys(self) -> List[Dict]:
        """Get all answer keys"""
        answer_keys = []
        
        for filename in os.listdir(self.answer_keys_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.answer_keys_dir, filename)
                with open(filepath, 'r') as f:
                    answer_key = json.load(f)
                    answer_keys.append(answer_key)
        
        # Sort by creation time (newest first)
        answer_keys.sort(key=lambda x: x.get('created_time', ''), reverse=True)
        
        return answer_keys
    
    def find_matching_answer_key(self, branch: str, class_name: str, subject: str, set_code: str = None) -> Optional[Dict]:
        """Find answer key matching the given criteria"""
        answer_keys = self.get_all_answer_keys()
        
        for answer_key in answer_keys:
            if (answer_key.get('branch') == branch and 
                answer_key.get('class') == class_name and 
                answer_key.get('subject') == subject):
                
                # If set_code is provided, match it too
                if set_code is None or answer_key.get('set_code') == set_code:
                    return answer_key
        
        return None
    
    def delete_answer_key(self, answer_key_id: str) -> bool:
        """Delete answer key by ID"""
        filepath = os.path.join(self.answer_keys_dir, f"{answer_key_id}.json")
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        
        return False