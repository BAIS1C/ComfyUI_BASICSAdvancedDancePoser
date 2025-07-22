# patterns/foot_movement_patterns.py
# FootMovementPattern - Coordinated foot and leg movement system

import numpy as np
import math
from typing import Dict, Tuple, List

class FootMovementPattern:
    """
    Manages foot movements (6 keypoints) coordinated with leg movements (knees/ankles)
    COCO-WholeBody indices:
    - Legs: left_knee(13), right_knee(14), left_ankle(15), right_ankle(16) 
    - Feet: left_big_toe(17), left_small_toe(18), left_heel(19), right_big_toe(20), right_small_toe(21), right_heel(22)
    """
    
    def __init__(self):
        # COCO-WholeBody keypoint indices
        self.left_knee = 13
        self.right_knee = 14
        self.left_ankle = 15
        self.right_ankle = 16
        self.left_big_toe = 17
        self.left_small_toe = 18
        self.left_heel = 19
        self.right_big_toe = 20
        self.right_small_toe = 21
        self.right_heel = 22
        
        self.step_patterns = self._create_step_patterns()
        self.tap_patterns = self._create_tap_patterns()
        self.stance_patterns = self._create_stance_patterns()
        
        # State tracking for coordinated movement
        self.current_step_phase = 0
        self.step_height = 0.0
        self.weight_distribution = 0.5  # 0.0=left foot, 0.5=centered, 1.0=right foot
        
    def _create_step_patterns(self) -> Dict:
        """Create stepping patterns with coordinated leg-foot movement"""
        return {
            'step_in_place': {
                'left_step': {
                    'knee_lift': 0.15,        # How much knee lifts
                    'ankle_lift': 0.12,       # Ankle follows knee
                    'foot_angle': -10,        # Foot angle (degrees)
                    'toe_lift': 0.08,         # Toes lift slightly
                    'heel_lift': 0.15,        # Heel lifts more
                    'weight_shift': -0.3      # Weight shifts away
                },
                'right_step': {
                    'knee_lift': 0.15,
                    'ankle_lift': 0.12, 
                    'foot_angle': -10,
                    'toe_lift': 0.08,
                    'heel_lift': 0.15,
                    'weight_shift': 0.3
                }
            },
            
            'march_step': {
                'left_step': {
                    'knee_lift': 0.25,        # Higher knee lift for marching
                    'ankle_lift': 0.20,
                    'foot_angle': -5,         # More horizontal foot
                    'toe_lift': 0.15,
                    'heel_lift': 0.25,
                    'weight_shift': -0.4
                },
                'right_step': {
                    'knee_lift': 0.25,
                    'ankle_lift': 0.20,
                    'foot_angle': -5,
                    'toe_lift': 0.15,
                    'heel_lift': 0.25,
                    'weight_shift': 0.4
                }
            },
            
            'shuffle_step': {
                'left_step': {
                    'knee_lift': 0.05,        # Minimal lift for shuffle
                    'ankle_lift': 0.03,
                    'foot_angle': 0,          # Foot stays flat
                    'toe_lift': 0.02,
                    'heel_lift': 0.02,
                    'weight_shift': -0.2,
                    'side_slide': -0.08       # Slide sideways
                },
                'right_step': {
                    'knee_lift': 0.05,
                    'ankle_lift': 0.03,
                    'foot_angle': 0,
                    'toe_lift': 0.02,
                    'heel_lift': 0.02,
                    'weight_shift': 0.2,
                    'side_slide': 0.08
                }
            },
            
            'bounce_step': {
                'both_feet': {
                    'knee_lift': 0.08,        # Both feet bounce together
                    'ankle_lift': 0.06,
                    'foot_angle': -15,        # Feet point down for bounce
                    'toe_lift': 0.04,
                    'heel_lift': 0.08,
                    'weight_shift': 0.0,      # Centered weight
                    'bounce_sync': True       # Both feet move together
                }
            }
        }
    
    def _create_tap_patterns(self) -> Dict:
        """Create tapping patterns for musical emphasis"""
        return {
            'toe_tap_left': {
                'knee_lift': 0.02,
                'ankle_flex': 0.05,           # Ankle flexes for toe tap
                'toe_emphasis': 0.08,         # Big toe taps down
                'heel_stay': 0.0,            # Heel stays planted
                'tap_intensity': 1.0
            },
            
            'toe_tap_right': {
                'knee_lift': 0.02,
                'ankle_flex': 0.05,
                'toe_emphasis': 0.08,
                'heel_stay': 0.0,
                'tap_intensity': 1.0
            },
            
            'heel_tap_left': {
                'knee_lift': 0.03,
                'ankle_point': -0.08,         # Ankle points for heel tap
                'toe_lift': 0.10,            # Toes lift up
                'heel_emphasis': 0.12,       # Heel taps down
                'tap_intensity': 1.2
            },
            
            'heel_tap_right': {
                'knee_lift': 0.03,
                'ankle_point': -0.08,
                'toe_lift': 0.10,
                'heel_emphasis': 0.12,
                'tap_intensity': 1.2
            },
            
            'stomp': {
                'knee_lift': 0.20,           # High lift for stomp
                'ankle_prep': 0.15,          # Ankle lifts before stomp
                'whole_foot_down': 0.25,     # Entire foot crashes down
                'impact_spread': 0.15,       # Foot spreads on impact
                'tap_intensity': 2.0
            }
        }
    
    def _create_stance_patterns(self) -> Dict:
        """Create stance patterns that affect overall posture"""
        return {
            'wide_stance': {
                'foot_spread': 0.15,         # Feet wider apart
                'knee_angle': 0.08,          # Knees slightly bent outward
                'weight_low': 0.05,          # Lower center of gravity
                'stability': 1.2
            },
            
            'narrow_stance': {
                'foot_spread': -0.05,        # Feet closer together
                'knee_angle': -0.03,         # Knees slightly inward
                'weight_high': -0.02,        # Higher center of gravity
                'stability': 0.8
            },
            
            'dancer_stance': {
                'foot_spread': 0.10,
                'knee_angle': 0.05,
                'toe_point': 0.08,           # Toes pointed outward
                'heel_lift': 0.03,           # Slight heel lift
                'grace_factor': 1.5
            },
            
            'power_stance': {
                'foot_spread': 0.20,
                'knee_angle': 0.12,
                'foot_plant': 0.10,          # Feet planted firmly
                'weight_centered': 0.0,
                'strength_factor': 1.8
            }
        }
    
    def apply_step_pattern(self, pose: np.ndarray, pattern_name: str, step_phase: int, 
                          intensity: float, beat_strength: float) -> np.ndarray:
        """Apply stepping pattern with coordinated leg movement"""
        
        if pattern_name not in self.step_patterns:
            return pose
            
        pattern = self.step_patterns[pattern_name]
        
        # Determine which foot is stepping based on phase
        is_left_step = (step_phase % 2) == 0
        step_strength = intensity * (0.5 + 0.5 * beat_strength)
        
        if pattern_name == 'bounce_step':
            # Both feet bounce together
            self._apply_bounce_movement(pose, pattern['both_feet'], step_strength)
        else:
            # Alternating step pattern
            if is_left_step:
                self._apply_single_step(pose, pattern['left_step'], 'left', step_strength)
                self._apply_support_leg(pose, 'right', step_strength * 0.3)
            else:
                self._apply_single_step(pose, pattern['right_step'], 'right', step_strength)
                self._apply_support_leg(pose, 'left', step_strength * 0.3)
        
        return pose
    
    def _apply_single_step(self, pose: np.ndarray, step_data: Dict, foot: str, strength: float):
        """Apply movement to a single foot and its leg"""
        
        # Get keypoint indices for this foot
        if foot == 'left':
            knee_idx, ankle_idx = self.left_knee, self.left_ankle
            big_toe_idx, small_toe_idx, heel_idx = self.left_big_toe, self.left_small_toe, self.left_heel
            side_multiplier = -1
        else:
            knee_idx, ankle_idx = self.right_knee, self.right_ankle
            big_toe_idx, small_toe_idx, heel_idx = self.right_big_toe, self.right_small_toe, self.right_heel
            side_multiplier = 1
        
        # Apply knee movement
        knee_lift = step_data.get('knee_lift', 0) * strength
        pose[knee_idx][1] -= knee_lift  # Lift knee up
        
        # Apply ankle movement (follows knee but less)
        ankle_lift = step_data.get('ankle_lift', 0) * strength
        pose[ankle_idx][1] -= ankle_lift
        
        # Apply foot angle and position
        foot_angle_deg = step_data.get('foot_angle', 0)
        foot_angle_offset = math.sin(math.radians(foot_angle_deg)) * 0.02 * strength
        
        # Apply toe movements
        toe_lift = step_data.get('toe_lift', 0) * strength
        pose[big_toe_idx][1] -= toe_lift
        pose[small_toe_idx][1] -= toe_lift
        pose[big_toe_idx][1] += foot_angle_offset
        pose[small_toe_idx][1] += foot_angle_offset
        
        # Apply heel movement
        heel_lift = step_data.get('heel_lift', 0) * strength
        pose[heel_idx][1] -= heel_lift
        pose[heel_idx][1] -= foot_angle_offset  # Opposite of toes for angle
        
        # Apply side slide if present (for shuffle)
        if 'side_slide' in step_data:
            side_slide = step_data['side_slide'] * strength
            pose[ankle_idx][0] += side_slide
            pose[big_toe_idx][0] += side_slide
            pose[small_toe_idx][0] += side_slide
            pose[heel_idx][0] += side_slide
        
        # Update weight distribution
        weight_shift = step_data.get('weight_shift', 0) * strength
        self.weight_distribution = np.clip(0.5 + weight_shift, 0.0, 1.0)
    
    def _apply_support_leg(self, pose: np.ndarray, foot: str, strength: float):
        """Apply subtle support leg adjustments"""
        
        if foot == 'left':
            knee_idx, ankle_idx = self.left_knee, self.left_ankle
        else:
            knee_idx, ankle_idx = self.right_knee, self.right_ankle
            
        # Support leg bends slightly to absorb weight
        support_bend = 0.03 * strength
        pose[knee_idx][1] += support_bend
        pose[ankle_idx][1] += support_bend * 0.5
    
    def _apply_bounce_movement(self, pose: np.ndarray, bounce_data: Dict, strength: float):
        """Apply synchronized bounce to both feet"""
        
        knee_lift = bounce_data.get('knee_lift', 0) * strength
        ankle_lift = bounce_data.get('ankle_lift', 0) * strength
        foot_angle_deg = bounce_data.get('foot_angle', 0)
        toe_lift = bounce_data.get('toe_lift', 0) * strength
        heel_lift = bounce_data.get('heel_lift', 0) * strength
        
        foot_angle_offset = math.sin(math.radians(foot_angle_deg)) * 0.02 * strength
        
        # Apply to both legs and feet
        for knee_idx, ankle_idx, big_toe_idx, small_toe_idx, heel_idx in [
            (self.left_knee, self.left_ankle, self.left_big_toe, self.left_small_toe, self.left_heel),
            (self.right_knee, self.right_ankle, self.right_big_toe, self.right_small_toe, self.right_heel)
        ]:
            # Legs
            pose[knee_idx][1] -= knee_lift
            pose[ankle_idx][1] -= ankle_lift
            
            # Feet
            pose[big_toe_idx][1] -= toe_lift + foot_angle_offset
            pose[small_toe_idx][1] -= toe_lift + foot_angle_offset
            pose[heel_idx][1] -= heel_lift - foot_angle_offset
    
    def apply_tap_pattern(self, pose: np.ndarray, tap_name: str, intensity: float, beat_strength: float) -> np.ndarray:
        """Apply tapping movement for musical accents"""
        
        if tap_name not in self.tap_patterns:
            return pose
            
        tap_data = self.tap_patterns[tap_name]
        tap_strength = intensity * beat_strength * tap_data.get('tap_intensity', 1.0)
        
        # Determine which foot to tap
        if 'left' in tap_name:
            foot_keypoints = [self.left_knee, self.left_ankle, self.left_big_toe, self.left_small_toe, self.left_heel]
        elif 'right' in tap_name:
            foot_keypoints = [self.right_knee, self.right_ankle, self.right_big_toe, self.right_small_toe, self.right_heel]
        else:  # Stomp - both feet
            foot_keypoints = [
                self.left_knee, self.left_ankle, self.left_big_toe, self.left_small_toe, self.left_heel,
                self.right_knee, self.right_ankle, self.right_big_toe, self.right_small_toe, self.right_heel
            ]
        
        # Apply tap-specific movements
        if tap_name.startswith('toe_tap'):
            knee_idx, ankle_idx, big_toe_idx, small_toe_idx, heel_idx = foot_keypoints[:5]
            
            # Subtle knee lift
            pose[knee_idx][1] -= tap_data['knee_lift'] * tap_strength
            
            # Ankle flexes
            ankle_flex = tap_data['ankle_flex'] * tap_strength
            pose[ankle_idx][1] -= ankle_flex
            
            # Toe emphasis
            toe_emphasis = tap_data['toe_emphasis'] * tap_strength
            pose[big_toe_idx][1] -= toe_emphasis
            pose[small_toe_idx][1] -= toe_emphasis
            
        elif tap_name.startswith('heel_tap'):
            knee_idx, ankle_idx, big_toe_idx, small_toe_idx, heel_idx = foot_keypoints[:5]
            
            # Knee lift
            pose[knee_idx][1] -= tap_data['knee_lift'] * tap_strength
            
            # Ankle points
            ankle_point = tap_data['ankle_point'] * tap_strength
            pose[ankle_idx][1] += ankle_point
            
            # Toes lift
            toe_lift = tap_data['toe_lift'] * tap_strength
            pose[big_toe_idx][1] -= toe_lift
            pose[small_toe_idx][1] -= toe_lift
            
            # Heel emphasis  
            heel_emphasis = tap_data['heel_emphasis'] * tap_strength
            pose[heel_idx][1] -= heel_emphasis
            
        elif tap_name == 'stomp':
            # Apply stomp to both feet
            knee_lift = tap_data['knee_lift'] * tap_strength
            ankle_prep = tap_data['ankle_prep'] * tap_strength
            whole_foot_down = tap_data['whole_foot_down'] * tap_strength
            
            # Both knees lift high
            pose[self.left_knee][1] -= knee_lift
            pose[self.right_knee][1] -= knee_lift
            
            # Ankles prepare
            pose[self.left_ankle][1] -= ankle_prep
            pose[self.right_ankle][1] -= ankle_prep
            
            # All foot parts crash down
            for idx in [self.left_big_toe, self.left_small_toe, self.left_heel,
                       self.right_big_toe, self.right_small_toe, self.right_heel]:
                pose[idx][1] -= whole_foot_down
        
        return pose
    
    def apply_stance_pattern(self, pose: np.ndarray, stance_name: str, intensity: float) -> np.ndarray:
        """Apply stance pattern that affects overall foot positioning"""
        
        if stance_name not in self.stance_patterns:
            return pose
            
        stance_data = self.stance_patterns[stance_name]
        stance_strength = intensity
        
        # Apply foot spread
        if 'foot_spread' in stance_data:
            foot_spread = stance_data['foot_spread'] * stance_strength
            
            # Move left foot left, right foot right
            left_foot_indices = [self.left_ankle, self.left_big_toe, self.left_small_toe, self.left_heel]
            right_foot_indices = [self.right_ankle, self.right_big_toe, self.right_small_toe, self.right_heel]
            
            for idx in left_foot_indices:
                pose[idx][0] -= foot_spread
            for idx in right_foot_indices:
                pose[idx][0] += foot_spread
        
        # Apply knee angle adjustments
        if 'knee_angle' in stance_data:
            knee_angle = stance_data['knee_angle'] * stance_strength
            pose[self.left_knee][0] -= knee_angle
            pose[self.right_knee][0] += knee_angle
        
        # Apply weight adjustments
        if 'weight_low' in stance_data:
            weight_adj = stance_data['weight_low'] * stance_strength
            pose[self.left_knee][1] += weight_adj
            pose[self.right_knee][1] += weight_adj
        elif 'weight_high' in stance_data:
            weight_adj = stance_data['weight_high'] * stance_strength
            pose[self.left_knee][1] += weight_adj
            pose[self.right_knee][1] += weight_adj
        
        # Special stance effects
        if 'toe_point' in stance_data:  # Dancer stance
            toe_point = stance_data['toe_point'] * stance_strength
            pose[self.left_big_toe][0] -= toe_point
            pose[self.left_small_toe][0] -= toe_point * 0.7
            pose[self.right_big_toe][0] += toe_point
            pose[self.right_small_toe][0] += toe_point * 0.7
            
        if 'heel_lift' in stance_data:  # Dancer stance
            heel_lift = stance_data['heel_lift'] * stance_strength
            pose[self.left_heel][1] -= heel_lift
            pose[self.right_heel][1] -= heel_lift
        
        return pose
    
    def get_coordinated_movement(self, dance_style: str, beat_strength: float, frame_idx: int) -> Tuple[str, str, str]:
        """Get coordinated foot pattern based on dance style and beat"""
        
        # Map dance styles to foot patterns
        style_patterns = {
            'energetic': {
                'step': 'step_in_place',
                'tap': 'toe_tap_left' if frame_idx % 2 == 0 else 'toe_tap_right',
                'stance': 'power_stance'
            },
            'smooth': {
                'step': 'shuffle_step',
                'tap': 'heel_tap_left' if frame_idx % 3 == 0 else 'heel_tap_right',
                'stance': 'dancer_stance'
            },
            'dramatic': {
                'step': 'march_step',
                'tap': 'stomp' if beat_strength > 0.8 else 'heel_tap_left',
                'stance': 'wide_stance'
            },
            'robot': {
                'step': 'step_in_place',
                'tap': 'toe_tap_left' if frame_idx % 4 < 2 else 'toe_tap_right',
                'stance': 'narrow_stance'
            },
            'bounce': {
                'step': 'bounce_step',
                'tap': 'toe_tap_left' if beat_strength > 0.6 else None,
                'stance': 'power_stance'
            }
        }
        
        patterns = style_patterns.get(dance_style, style_patterns['energetic'])
        
        return patterns['step'], patterns.get('tap'), patterns['stance']
    
    def update_step_phase(self, beat_strength: float, tempo_multiplier: float):
        """Update step phase based on beat"""
        if beat_strength > 0.5:
            self.current_step_phase += int(tempo_multiplier)
        
        # Keep phase in reasonable range
        self.current_step_phase = self.current_step_phase % 8