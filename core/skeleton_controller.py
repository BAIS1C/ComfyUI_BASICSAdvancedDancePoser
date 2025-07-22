# core/skeleton_controller.py
# Manages COCO-WholeBody 133-point skeleton structure

import numpy as np
import math
from typing import Dict, List, Tuple
from .constants import COCOWholeBodyConstants

class SkeletonController:
    """Manages COCO-WholeBody 133-point skeleton structure"""
    
    def __init__(self):
        self.constants = COCOWholeBodyConstants()
        self.base_pose = self._create_base_pose()
        
    def _create_base_pose(self) -> List[List[float]]:
        """Create anatomically correct 133-point base pose"""
        pose = [[0.0, 0.0] for _ in range(133)]
        
        # Body keypoints (0-16) - COCO standard
        body_positions = [
            [0.50, 0.10],  # 0: nose
            [0.48, 0.09],  # 1: left_eye
            [0.52, 0.09],  # 2: right_eye
            [0.46, 0.09],  # 3: left_ear
            [0.54, 0.09],  # 4: right_ear
            [0.42, 0.25],  # 5: left_shoulder
            [0.58, 0.25],  # 6: right_shoulder
            [0.35, 0.40],  # 7: left_elbow
            [0.65, 0.40],  # 8: right_elbow
            [0.28, 0.55],  # 9: left_wrist
            [0.72, 0.55],  # 10: right_wrist
            [0.44, 0.55],  # 11: left_hip
            [0.56, 0.55],  # 12: right_hip
            [0.42, 0.75],  # 13: left_knee
            [0.58, 0.75],  # 14: right_knee
            [0.40, 0.95],  # 15: left_ankle
            [0.60, 0.95],  # 16: right_ankle
        ]
        
        # Set body positions
        for i, pos in enumerate(body_positions):
            pose[i] = pos.copy()
            
        # Feet keypoints (17-22)
        foot_positions = [
            [0.38, 0.98], [0.40, 1.00], [0.42, 0.98],  # Left foot
            [0.58, 0.98], [0.60, 1.00], [0.62, 0.98]   # Right foot
        ]
        
        for i, pos in enumerate(foot_positions):
            pose[17 + i] = pos.copy()
            
        # Face keypoints (23-90) - Simplified face outline
        self._create_face_keypoints(pose, 23, center_x=0.50, center_y=0.10)
        
        # Left hand keypoints (91-111)
        self._create_hand_keypoints(pose, 91, center_x=0.28, center_y=0.55)
        
        # Right hand keypoints (112-132)
        self._create_hand_keypoints(pose, 112, center_x=0.72, center_y=0.55)
        
        return pose
    
    def _create_face_keypoints(self, pose: List, start_idx: int, center_x: float, center_y: float):
        """Create 68 face keypoints in anatomically correct positions"""
        # Simplified face layout - you can expand this for more detail
        face_radius = 0.08
        
        # Face outline (17 points)
        for i in range(17):
            angle = -math.pi + (2 * math.pi * i / 16)
            x = center_x + face_radius * 0.8 * math.cos(angle)
            y = center_y + face_radius * 1.2 * math.sin(angle) + 0.02
            pose[start_idx + i] = [x, y]
        
        # Eyebrows, eyes, nose, mouth (51 points) - simplified positions
        for i in range(17, 68):
            # Distribute remaining facial features
            angle = 2 * math.pi * (i - 17) / 51
            radius = face_radius * 0.3
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            pose[start_idx + i] = [x, y]
    
    def _create_hand_keypoints(self, pose: List, start_idx: int, center_x: float, center_y: float):
        """Create 21 hand keypoints"""
        hand_size = 0.08
        
        # Simplified hand layout - 5 fingers, 4 joints each + wrist
        for i in range(21):
            if i == 0:  # Wrist
                pose[start_idx + i] = [center_x, center_y]
            else:
                # Distribute finger joints
                finger = (i - 1) // 4
                joint = (i - 1) % 4
                
                finger_angle = -math.pi/2 + (math.pi * finger / 4)
                extension = hand_size * (joint + 1) / 4
                
                x = center_x + extension * math.cos(finger_angle)
                y = center_y + extension * math.sin(finger_angle)
                pose[start_idx + i] = [x, y]
    
    def get_all_connections(self) -> List[Tuple[int, int]]:
        """Get all skeleton connections"""
        connections = []
        connections.extend(self.constants.BODY_CONNECTIONS)
        connections.extend(self.constants.FOOT_CONNECTIONS)
        connections.extend(self.constants.FACE_OUTLINE)
        connections.extend(self.constants.LEFT_HAND_CONNECTIONS)
        connections.extend(self.constants.RIGHT_HAND_CONNECTIONS)
        return connections
    
    def get_keypoint_groups(self) -> Dict[str, List[int]]:
        """Get keypoint indices grouped by body part"""
        return {
            'body': self.constants.BODY_KEYPOINTS,
            'feet': self.constants.FOOT_KEYPOINTS,
            'face': self.constants.FACE_KEYPOINTS,
            'left_hand': self.constants.LEFT_HAND_KEYPOINTS,
            'right_hand': self.constants.RIGHT_HAND_KEYPOINTS
        }