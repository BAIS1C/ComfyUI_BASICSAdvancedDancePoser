# renderers/pose_renderer.py
# Enhanced Pose Renderer - Professional COCO-WholeBody visualization

import torch
import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Optional

class EnhancedPoseRenderer:
    """
    Professional renderer for COCO-WholeBody 133 keypoints with visual hierarchy,
    colors, effects, and proper anatomical representation
    """
    
    def __init__(self):
        self.setup_visual_config()
        self.setup_connection_groups()
        
    def setup_visual_config(self):
        """Setup colors, sizes, and visual hierarchy"""
        
        # Color schemes for different backgrounds
        self.color_schemes = {
            'black': {
                'body_skeleton': (255, 255, 255),      # White
                'body_joints': (255, 200, 200),        # Light pink
                'face_skeleton': (0, 255, 255),        # Cyan
                'face_joints': (100, 255, 255),        # Light cyan
                'hand_skeleton': (255, 255, 0),        # Yellow
                'hand_joints': (255, 255, 150),        # Light yellow
                'foot_skeleton': (255, 100, 255),      # Magenta
                'foot_joints': (255, 150, 255),        # Light magenta
                'emphasis': (0, 255, 0),               # Green for special highlights
                'beat_pulse': (255, 0, 0),             # Red for beat emphasis
                'text': (255, 255, 255)               # White text
            },
            'white': {
                'body_skeleton': (0, 0, 0),            # Black
                'body_joints': (100, 50, 50),          # Dark red
                'face_skeleton': (0, 100, 150),        # Dark cyan
                'face_joints': (0, 50, 100),           # Darker cyan
                'hand_skeleton': (150, 150, 0),        # Dark yellow
                'hand_joints': (100, 100, 0),          # Darker yellow
                'foot_skeleton': (150, 0, 150),        # Dark magenta
                'foot_joints': (100, 0, 100),          # Darker magenta
                'emphasis': (0, 150, 0),               # Dark green
                'beat_pulse': (200, 0, 0),             # Dark red
                'text': (0, 0, 0)                     # Black text
            },
            'gray': {
                'body_skeleton': (255, 255, 255),      # White
                'body_joints': (220, 180, 180),        # Light pink
                'face_skeleton': (100, 255, 255),      # Bright cyan
                'face_joints': (150, 255, 255),        # Light cyan
                'hand_skeleton': (255, 255, 100),      # Bright yellow
                'hand_joints': (255, 255, 180),        # Light yellow
                'foot_skeleton': (255, 100, 255),      # Bright magenta
                'foot_joints': (255, 180, 255),        # Light magenta
                'emphasis': (100, 255, 100),           # Bright green
                'beat_pulse': (255, 100, 100),         # Bright red
                'text': (255, 255, 255)               # White text
            }
        }
        
        # Line thicknesses for different body parts
        self.line_thickness = {
            'body_major': 4,      # Torso, major limbs
            'body_minor': 3,      # Arms, legs
            'face_outline': 2,    # Face perimeter
            'face_details': 1,    # Facial features
            'hand_major': 2,      # Hand structure
            'hand_details': 1,    # Finger details
            'foot_structure': 3,  # Foot outline
            'foot_details': 2     # Toe details
        }
        
        # Joint sizes for different body parts
        self.joint_sizes = {
            'head': 6,
            'body_major': 5,      # Shoulders, hips
            'body_minor': 4,      # Elbows, knees, ankles
            'face': 2,
            'hand_wrist': 4,
            'hand_finger': 2,
            'foot_main': 4,
            'foot_toe': 3
        }
        
        # Animation effects
        self.effects = {
            'beat_pulse_scale': 1.5,        # How much joints pulse on beat
            'movement_trail_length': 3,     # Frames to show movement trail
            'glow_intensity': 0.3,          # Glow effect strength
            'dance_style_modifier': 1.2     # Style-based visual emphasis
        }
    
    def setup_connection_groups(self):
        """Define connection groups with visual hierarchy"""
        
        # Body connections (COCO standard 0-16)
        self.body_connections = {
            'torso_major': [
                (5, 6),   # Shoulders
                (11, 12), # Hips
                (5, 11),  # Left torso
                (6, 12)   # Right torso
            ],
            'head_neck': [
                (0, 1), (0, 2),  # Nose to eyes
                (1, 3), (2, 4),  # Eyes to ears
                (0, 5), (0, 6)   # Head to shoulders (simplified neck)
            ],
            'arms': [
                (5, 7), (7, 9),   # Left arm
                (6, 8), (8, 10)   # Right arm
            ],
            'legs': [
                (11, 13), (13, 15),  # Left leg
                (12, 14), (14, 16)   # Right leg
            ]
        }
        
        # Foot connections (17-22)
        self.foot_connections = [
            (15, 17), (15, 18), (15, 19),  # Left ankle to foot parts
            (16, 20), (16, 21), (16, 22),  # Right ankle to foot parts
            (17, 18), (18, 19), (19, 17),  # Left foot triangle
            (20, 21), (21, 22), (22, 20)   # Right foot triangle
        ]
        
        # Face connections (23-90) - Simplified for performance
        self.face_connections = [
            # Face outline (jaw line)
            *[(23 + i, 23 + i + 1) for i in range(16)],
            # Simplified facial features
            (27, 30), (30, 33), (33, 35),  # Nose bridge and tip
            (36, 39), (42, 45),            # Eye outlines (simplified)
            (48, 54), (54, 48)             # Mouth outline (simplified)
        ]
        
        # Hand connections (91-111, 112-132) - Simplified finger structure
        self.hand_connections = {
            'left_hand': [
                # Simplified hand structure - main finger lines
                (91, 95), (95, 99), (99, 103),     # Index finger
                (91, 96), (96, 100), (100, 104),   # Middle finger  
                (91, 97), (97, 101), (101, 105),   # Ring finger
                (91, 98), (98, 102), (102, 106),   # Pinky
                (91, 107), (107, 111)              # Thumb
            ],
            'right_hand': [
                # Mirror for right hand
                (112, 116), (116, 120), (120, 124),  # Index finger
                (112, 117), (117, 121), (121, 125),  # Middle finger
                (112, 118), (118, 122), (122, 126),  # Ring finger
                (112, 119), (119, 123), (123, 127),  # Pinky
                (112, 128), (128, 132)               # Thumb
            ]
        }
    
    def render_pose_sequence(self, pose_sequence: List, width: int, height: int,
                           background_color: str = "black", show_hands: bool = True,
                           show_face_details: bool = True, show_effects: bool = True,
                           beat_data: Optional[List] = None, dance_style: str = "energetic",
                           debug: bool = False) -> torch.Tensor:
        """
        Render complete pose sequence with professional visualization
        """
        
        bg_colors = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128)}
        bg_color = bg_colors.get(background_color, (0, 0, 0))
        colors = self.color_schemes.get(background_color, self.color_schemes["black"])
        
        frames = []
        
        for frame_idx, pose in enumerate(pose_sequence):
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            
            # Convert pose to pixel coordinates
            keypoints = self._pose_to_pixels(pose, width, height)
            
            # Get beat strength for this frame
            beat_strength = 0.0
            if beat_data and frame_idx < len(beat_data):
                beat_strength = beat_data[frame_idx]
            
            # Apply dance style visual modifier
            style_intensity = self._get_style_intensity(dance_style, beat_strength)
            
            # Render skeleton connections
            self._render_skeleton_connections(frame, keypoints, colors, beat_strength, style_intensity)
            
            # Render detailed body parts
            if show_hands:
                self._render_hands(frame, keypoints, colors, beat_strength, style_intensity)
            
            if show_face_details:
                self._render_face_details(frame, keypoints, colors, beat_strength, style_intensity)
            
            self._render_feet(frame, keypoints, colors, beat_strength, style_intensity)
            
            # Render joints
            self._render_joints(frame, keypoints, colors, beat_strength, style_intensity)
            
            # Apply visual effects
            if show_effects:
                self._apply_visual_effects(frame, keypoints, colors, beat_strength, style_intensity)
            
            # Debug information
            if debug:
                self._render_debug_info(frame, keypoints, colors, frame_idx, beat_strength)
            
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
        
        return torch.stack(frames)
    
    def _pose_to_pixels(self, pose: List, width: int, height: int) -> List[Tuple[int, int]]:
        """Convert normalized pose coordinates to pixel coordinates with bounds checking"""
        keypoints = []
        for x, y in pose:
            pixel_x = int(np.clip(x * width, 0, width - 1))
            pixel_y = int(np.clip(y * height, 0, height - 1))
            keypoints.append((pixel_x, pixel_y))
        return keypoints
    
    def _get_style_intensity(self, dance_style: str, beat_strength: float) -> float:
        """Get visual intensity modifier based on dance style"""
        style_modifiers = {
            'energetic': 1.3,
            'smooth': 0.8,
            'dramatic': 1.5,
            'robot': 0.9,
            'bounce': 1.2
        }
        base_intensity = style_modifiers.get(dance_style, 1.0)
        return base_intensity * (1.0 + beat_strength * 0.5)
    
    def _render_skeleton_connections(self, frame: np.ndarray, keypoints: List, 
                                   colors: Dict, beat_strength: float, style_intensity: float):
        """Render main skeleton connections with hierarchy"""
        
        # Render torso (highest priority)
        for connection in self.body_connections['torso_major']:
            if self._valid_connection(connection, keypoints):
                thickness = int(self.line_thickness['body_major'] * style_intensity)
                color = self._get_beat_color(colors['body_skeleton'], colors['beat_pulse'], beat_strength)
                cv2.line(frame, keypoints[connection[0]], keypoints[connection[1]], color, thickness)
        
        # Render head/neck
        for connection in self.body_connections['head_neck']:
            if self._valid_connection(connection, keypoints):
                thickness = int(self.line_thickness['body_major'] * style_intensity)
                color = self._get_beat_color(colors['emphasis'], colors['beat_pulse'], beat_strength * 0.7)
                cv2.line(frame, keypoints[connection[0]], keypoints[connection[1]], color, thickness)
        
        # Render arms
        for connection in self.body_connections['arms']:
            if self._valid_connection(connection, keypoints):
                thickness = int(self.line_thickness['body_minor'] * style_intensity)
                color = colors['body_skeleton']
                cv2.line(frame, keypoints[connection[0]], keypoints[connection[1]], color, thickness)
        
        # Render legs
        for connection in self.body_connections['legs']:
            if self._valid_connection(connection, keypoints):
                thickness = int(self.line_thickness['body_minor'] * style_intensity)
                color = colors['body_skeleton']
                cv2.line(frame, keypoints[connection[0]], keypoints[connection[1]], color, thickness)
    
    def _render_hands(self, frame: np.ndarray, keypoints: List, colors: Dict, 
                     beat_strength: float, style_intensity: float):
        """Render detailed hand structures"""
        
        hand_color = self._get_beat_color(colors['hand_skeleton'], colors['beat_pulse'], beat_strength * 0.3)
        thickness = int(self.line_thickness['hand_major'] * style_intensity)
        
        # Render left hand
        for connection in self.hand_connections['left_hand']:
            if self._valid_connection(connection, keypoints):
                cv2.line(frame, keypoints[connection[0]], keypoints[connection[1]], hand_color, thickness)
        
        # Render right hand  
        for connection in self.hand_connections['right_hand']:
            if self._valid_connection(connection, keypoints):
                cv2.line(frame, keypoints[connection[0]], keypoints[connection[1]], hand_color, thickness)
    
    def _render_face_details(self, frame: np.ndarray, keypoints: List, colors: Dict,
                           beat_strength: float, style_intensity: float):
        """Render facial features"""
        
        face_color = self._get_beat_color(colors['face_skeleton'], colors['emphasis'], beat_strength * 0.2)
        thickness = max(1, int(self.line_thickness['face_outline'] * style_intensity))
        
        # Render face connections
        for connection in self.face_connections:
            if self._valid_connection(connection, keypoints):
                cv2.line(frame, keypoints[connection[0]], keypoints[connection[1]], face_color, thickness)
    
    def _render_feet(self, frame: np.ndarray, keypoints: List, colors: Dict,
                    beat_strength: float, style_intensity: float):
        """Render foot structures with emphasis on beat"""
        
        foot_color = self._get_beat_color(colors['foot_skeleton'], colors['beat_pulse'], beat_strength * 0.8)
        thickness = int(self.line_thickness['foot_structure'] * style_intensity)
        
        # Render foot connections
        for connection in self.foot_connections:
            if self._valid_connection(connection, keypoints):
                cv2.line(frame, keypoints[connection[0]], keypoints[connection[1]], foot_color, thickness)
    
    def _render_joints(self, frame: np.ndarray, keypoints: List, colors: Dict,
                      beat_strength: float, style_intensity: float):
        """Render joints with size hierarchy and beat emphasis"""
        
        beat_scale = 1.0 + beat_strength * (self.effects['beat_pulse_scale'] - 1.0)
        
        # Head (0-4)
        for i in range(5):
            if i < len(keypoints):
                size = int(self.joint_sizes['head'] * style_intensity * beat_scale)
                color = self._get_beat_color(colors['body_joints'], colors['beat_pulse'], beat_strength)
                cv2.circle(frame, keypoints[i], size, color, -1)
        
        # Major body joints (shoulders, hips)
        for i in [5, 6, 11, 12]:
            if i < len(keypoints):
                size = int(self.joint_sizes['body_major'] * style_intensity * beat_scale)
                color = self._get_beat_color(colors['body_joints'], colors['emphasis'], beat_strength)
                cv2.circle(frame, keypoints[i], size, color, -1)
        
        # Minor body joints (elbows, knees, ankles)
        for i in [7, 8, 9, 10, 13, 14, 15, 16]:
            if i < len(keypoints):
                size = int(self.joint_sizes['body_minor'] * style_intensity)
                cv2.circle(frame, keypoints[i], size, colors['body_joints'], -1)
        
        # Foot joints (17-22)
        for i in range(17, 23):
            if i < len(keypoints):
                size = int(self.joint_sizes['foot_main'] * style_intensity * beat_scale)
                color = self._get_beat_color(colors['foot_joints'], colors['beat_pulse'], beat_strength)
                cv2.circle(frame, keypoints[i], size, color, -1)
        
        # Face joints (23-90)
        for i in range(23, min(91, len(keypoints))):
            size = max(1, int(self.joint_sizes['face'] * style_intensity))
            cv2.circle(frame, keypoints[i], size, colors['face_joints'], -1)
        
        # Hand joints (91-132)
        for i in range(91, min(133, len(keypoints))):
            if i in [91, 112]:  # Wrists
                size = int(self.joint_sizes['hand_wrist'] * style_intensity)
                color = colors['hand_joints']
            else:  # Fingers
                size = int(self.joint_sizes['hand_finger'] * style_intensity)
                color = colors['hand_joints']
            cv2.circle(frame, keypoints[i], size, color, -1)
    
    def _apply_visual_effects(self, frame: np.ndarray, keypoints: List, colors: Dict,
                            beat_strength: float, style_intensity: float):
        """Apply special visual effects"""
        
        # Beat pulse glow effect
        if beat_strength > 0.6:
            self._apply_glow_effect(frame, keypoints, colors['beat_pulse'], beat_strength)
        
        # Dance style specific effects
        if style_intensity > 1.2:
            self._apply_energy_effect(frame, keypoints, colors['emphasis'], style_intensity)
    
    def _apply_glow_effect(self, frame: np.ndarray, keypoints: List, glow_color: Tuple, intensity: float):
        """Apply glow effect around major joints"""
        glow_size = int(15 * intensity)
        
        # Apply glow to major joints
        for i in [0, 5, 6, 9, 10, 11, 12, 15, 16]:  # Head, shoulders, wrists, hips, ankles
            if i < len(keypoints):
                # Create glow circle with alpha blending simulation
                overlay = frame.copy()
                cv2.circle(overlay, keypoints[i], glow_size, glow_color, -1)
                alpha = self.effects['glow_intensity'] * intensity
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def _apply_energy_effect(self, frame: np.ndarray, keypoints: List, energy_color: Tuple, intensity: float):
        """Apply energy lines between major joints"""
        energy_thickness = max(1, int(intensity))
        
        # Energy connections
        energy_connections = [(5, 6), (11, 12), (9, 10), (15, 16)]  # Shoulders, hips, wrists, ankles
        
        for connection in energy_connections:
            if self._valid_connection(connection, keypoints):
                # Draw energy line with slight transparency effect
                overlay = frame.copy()
                cv2.line(overlay, keypoints[connection[0]], keypoints[connection[1]], 
                        energy_color, energy_thickness)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    def _render_debug_info(self, frame: np.ndarray, keypoints: List, colors: Dict,
                          frame_idx: int, beat_strength: float):
        """Render debug information"""
        
        text_color = colors['text']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25), font, font_scale, text_color, thickness)
        cv2.putText(frame, f"Beat: {beat_strength:.2f}", (10, 45), font, font_scale, text_color, thickness)
        cv2.putText(frame, f"Keypoints: {len(keypoints)}", (10, 65), font, font_scale, text_color, thickness)
        
        # Keypoint indices (for major joints only)
        major_joints = [0, 5, 6, 9, 10, 11, 12, 15, 16]
        for i in major_joints:
            if i < len(keypoints):
                cv2.putText(frame, str(i), (keypoints[i][0] + 5, keypoints[i][1] - 5),
                           font, 0.3, text_color, 1)
    
    def _valid_connection(self, connection: Tuple[int, int], keypoints: List) -> bool:
        """Check if connection indices are valid"""
        return (connection[0] < len(keypoints) and connection[1] < len(keypoints) and
                connection[0] >= 0 and connection[1] >= 0)
    
    def _get_beat_color(self, base_color: Tuple, beat_color: Tuple, beat_strength: float) -> Tuple:
        """Blend base color with beat color based on beat strength"""
        if beat_strength <= 0:
            return base_color
        
        # Blend colors
        blend_factor = min(beat_strength, 1.0)
        blended = tuple(
            int(base_color[i] * (1 - blend_factor) + beat_color[i] * blend_factor)
            for i in range(3)
        )
        return blended
    
    def create_visualization_config(self, show_body: bool = True, show_hands: bool = True,
                                  show_face: bool = True, show_effects: bool = True,
                                  custom_colors: Optional[Dict] = None) -> Dict:
        """Create customizable visualization configuration"""
        
        config = {
            'show_body': show_body,
            'show_hands': show_hands,
            'show_face': show_face,
            'show_effects': show_effects,
            'colors': custom_colors or self.color_schemes['black'],
            'line_thickness': self.line_thickness.copy(),
            'joint_sizes': self.joint_sizes.copy(),
            'effects': self.effects.copy()
        }
        
        return config