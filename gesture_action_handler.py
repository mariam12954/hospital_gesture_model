import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

class GestureActionHandler:
    """Handles gesture-to-action mapping for hospital system"""

    def __init__(self, mapping_file='config/medical_gesture_mapping.json'):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            self.gesture_map = json.load(f)
        self.action_history = []
        self.last_action_time = None
        self.debounce_time = 1.0

    def get_action(self, gesture: str, confidence: float) -> Optional[Dict[str, Any]]:
        thresholds = {
            'critical': 0.85,
            'high': 0.75,
            'medium': 0.65,
            'low': 0.55
        }

        if gesture not in self.gesture_map:
            return {'status': 'unknown', 'reason': f'Gesture "{gesture}" not in mapping'}

        gesture_info = self.gesture_map[gesture]
        required_confidence = thresholds.get(gesture_info['priority'], 0.70)

        if confidence < required_confidence:
            return {'status': 'rejected', 'reason': 'Low confidence'}

        current_time = time.time()
        if self.last_action_time and (current_time - self.last_action_time < self.debounce_time):
            return {'status': 'debounced', 'reason': 'Too soon'}

        self.last_action_time = current_time
        self.action_history.append({
            'timestamp': datetime.now().isoformat(),
            'gesture': gesture,
            'confidence': confidence,
            'action': gesture_info['action'],
            'priority': gesture_info['priority']
        })

        return {
            'status': 'success',
            'action': gesture_info['action'],
            'description': gesture_info['description'],
            'medical_use': gesture_info['medical_use'],
            'icon': gesture_info['icon'],
            'priority': gesture_info['priority'],
            'confidence': confidence,
            'alert_level': gesture_info.get('alert_level', None)
        }

    def get_action_history(self, limit: int = 10):
        return self.action_history[-limit:]

    def clear_history(self):
        self.action_history = []
