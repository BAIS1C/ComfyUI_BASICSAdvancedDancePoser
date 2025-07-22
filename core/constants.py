# core/constants.py
# COCO-WholeBody 133 keypoint structure and connections

from typing import List, Tuple

class COCOWholeBodyConstants:
    """COCO-WholeBody 133 keypoint structure and connections"""
    
    # Keypoint indices by body part
    BODY_KEYPOINTS = list(range(0, 17))        # 0-16: Standard COCO body
    FOOT_KEYPOINTS = list(range(17, 23))       # 17-22: Feet (6 points)
    FACE_KEYPOINTS = list(range(23, 91))       # 23-90: Face (68 points)
    LEFT_HAND_KEYPOINTS = list(range(91, 112)) # 91-111: Left hand (21 points)
    RIGHT_HAND_KEYPOINTS = list(range(112, 133)) # 112-132: Right hand (21 points)
    
    # Body connections (COCO standard)
    BODY_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),      # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        (5, 11), (6, 12)  # Torso
    ]
    
    # Foot connections (simplified)
    FOOT_CONNECTIONS = [
        (15, 17), (15, 18), (15, 19),  # Left foot
        (16, 20), (16, 21), (16, 22)   # Right foot
    ]
    
    # Face outline connections (simplified for performance)
    FACE_OUTLINE = [(i, i+1) for i in range(23, 39)]  # Face border
    
    # Hand connections (simplified)
    LEFT_HAND_CONNECTIONS = [(91 + i, 91 + i + 1) for i in range(0, 20, 4)]
    RIGHT_HAND_CONNECTIONS = [(112 + i, 112 + i + 1) for i in range(0, 20, 4)]
    
    # Body part names for easy reference
    BODY_PART_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]