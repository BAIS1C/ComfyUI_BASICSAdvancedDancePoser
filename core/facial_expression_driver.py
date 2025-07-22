# core/facial_expression_driver.py
# Maps music tonality to facial expressions using COCO face keypoints

import numpy as np
from typing import Dict

class FacialExpressionDriver:
    """Maps music tonality to facial expressions using COCO face keypoints"""
    
    def __init__(self):
        self.face_start_idx = 23  # COCO-WholeBody face starts at index 23
        self.expression_templates = self._create_expression_templates()
    
    def _create_expression_templates(self) -> Dict:
        """Create facial expression templates"""
        return {
            'neutral': np.zeros(68),
            'happy': self._create_happy_expression(),
            'sad': self._create_sad_expression(),  
            'surprised': self._create_surprised_expression(),
            'concentrated': self._create_concentrated_expression()
        }
    
    def _create_happy_expression(self) -> np.ndarray:
        """Create happy expression offsets"""
        offsets = np.zeros((68, 2))
        # Mouth corners up, eyes slightly squinted
        offsets[48:54, 1] -= 0.01  # Mouth up
        offsets[36:42, 1] += 0.005  # Eyes squint
        return offsets
    
    def _create_sad_expression(self) -> np.ndarray:
        """Create sad expression offsets"""  
        offsets = np.zeros((68, 2))
        # Mouth corners down, eyebrows down
        offsets[48:54, 1] += 0.01  # Mouth down
        offsets[17:22, 1] += 0.005  # Eyebrows down
        return offsets
    
    def _create_surprised_expression(self) -> np.ndarray:
        """Create surprised expression offsets"""
        offsets = np.zeros((68, 2))
        # Eyes wide, mouth open
        offsets[36:48, 1] -= 0.01  # Eyes wide
        offsets[60:68, 1] += 0.01  # Mouth open
        return offsets
        
    def _create_concentrated_expression(self) -> np.ndarray:
        """Create concentrated expression offsets"""
        offsets = np.zeros((68, 2))
        # Slight frown, focused eyes
        offsets[48:54, 1] += 0.005  # Slight frown
        offsets[36:42, 0] += 0.002  # Eyes focused
        return offsets
    
    def apply_expression(self, pose: np.ndarray, tonality_data: Dict, frame_idx: int, intensity: float) -> np.ndarray:
        """Apply facial expression based on music tonality"""
        if frame_idx >= len(tonality_data['major_strength']):
            return pose
            
        major = tonality_data['major_strength'][frame_idx]
        minor = tonality_data['minor_strength'][frame_idx]
        brightness = tonality_data['brightness'][frame_idx] 
        darkness = tonality_data['darkness'][frame_idx]
        
        # Determine expression based on tonality
        if major > minor and brightness > 0.5:
            expression = 'happy'
            strength = major * brightness
        elif minor > major and darkness > 0.3:
            expression = 'sad'
            strength = minor * darkness
        elif brightness > 0.7:
            expression = 'surprised'
            strength = brightness
        else:
            expression = 'concentrated'
            strength = 0.3
        
        # Apply expression
        if expression in self.expression_templates:
            offsets = self.expression_templates[expression]
            blend_strength = intensity * strength * 0.5
            
            for i in range(68):
                face_idx = self.face_start_idx + i
                if face_idx < len(pose):
                    pose[face_idx][0] += offsets[i][0] * blend_strength
                    pose[face_idx][1] += offsets[i][1] * blend_strength
        
        return pose