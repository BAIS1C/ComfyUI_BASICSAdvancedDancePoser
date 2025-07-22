# advanced_dance_poser.py
# Main node for ComfyUI_BASICSAdvancedDancePoser - Complete orchestration

import torch
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional

# Import all our modules
from .core.constants import COCOWholeBodyConstants
from .core.audio_analyzer import AudioAnalyzer
from .core.skeleton_controller import SkeletonController
from .core.facial_expression_driver import FacialExpressionDriver
from .core.body_animation_driver import BodyAnimationDriver, HandGestureController
from .patterns.foot_movement_patterns import FootMovementPattern
from .renderers.pose_renderer import EnhancedPoseRenderer

class ComfyUI_BASICSAdvancedDancePoser:
    """
    Advanced COCO-WholeBody 133-keypoint dance animation system with:
    - Musical facial expressions based on tonality
    - Professional gesture library (victory, clap, pump fists, etc.)
    - Coordinated foot and leg movement patterns
    - Beat-responsive visual effects
    - Multiple dance styles with unique characteristics
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # === CORE SETTINGS ===
                "audio": ("AUDIO",),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "max_duration": ("FLOAT", {"default": 180.0, "min": 3.0, "max": 180.0, "step": 0.1}),
                "width": ("INT", {"default": 480, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 832, "min": 64, "max": 2048}),
                "fps": ("INT", {"default": 24, "min": 12, "max": 60}),
                
                # === DANCE STYLE ===
                "dance_style": (["energetic", "smooth", "dramatic", "robot", "bounce"], {"default": "energetic"}),
                "movement_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smoothing": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "beat_sensitivity": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tempo_factor": (["1x", "0.5x", "0.25x", "2x"], {"default": "1x"}),
                
                # === BODY PARTS CONTROL ===
                "enable_facial_expressions": ("BOOLEAN", {"default": True}),
                "facial_expression_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "arm_movement_style": (["flowing", "sharp", "waves", "reaches", "gestures"], {"default": "gestures"}),
                "leg_movement_style": (["step", "shuffle", "march", "bounce"], {"default": "step"}),
                "hand_gesture_style": (["auto", "open_palm", "closed_fist", "pointing", "peace_sign", "thumbs_up", "rock_horns"], {"default": "auto"}),
                "foot_pattern_style": (["auto", "step_in_place", "march_step", "shuffle_step", "bounce_step"], {"default": "auto"}),
                
                # === ADVANCED FEATURES ===
                "enable_coordinated_movement": ("BOOLEAN", {"default": True}),
                "enable_beat_emphasis": ("BOOLEAN", {"default": True}),
                "enable_auto_gesture_change": ("BOOLEAN", {"default": True}),
                "gesture_change_interval": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5}),
            },
            "optional": {
                # === VISUAL SETTINGS ===
                "background_color": (["black", "white", "gray"], {"default": "black"}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "show_face_details": ("BOOLEAN", {"default": True}),
                "show_visual_effects": ("BOOLEAN", {"default": True}),
                "visual_style_intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                
                # === DEBUG & MISC ===
                "seed": ("INT", {"default": 42, "min": 0, "max": 99999}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("pose_video", "audio")
    FUNCTION = "generate_pose_video"
    CATEGORY = "BASICS/Audio"

    def __init__(self):
        """Initialize all animation modules"""
        # Core modules
        self.audio_analyzer = AudioAnalyzer()
        self.skeleton_controller = SkeletonController()
        self.facial_driver = FacialExpressionDriver()
        self.body_driver = BodyAnimationDriver()
        self.hand_controller = HandGestureController()
        self.foot_controller = FootMovementPattern()
        self.renderer = EnhancedPoseRenderer()
        
        # Animation state tracking
        self.gesture_history = []
        self.current_gesture = None
        self.gesture_transition_frame = 0
        self.step_phase_counter = 0

    def generate_pose_video(self, audio, start_time, max_duration, width, height, fps,
                           dance_style, movement_intensity, smoothing, beat_sensitivity, tempo_factor,
                           enable_facial_expressions, facial_expression_intensity,
                           arm_movement_style, leg_movement_style, hand_gesture_style, foot_pattern_style,
                           enable_coordinated_movement, enable_beat_emphasis, enable_auto_gesture_change,
                           gesture_change_interval,
                           background_color="black", show_hands=True, show_face_details=True,
                           show_visual_effects=True, visual_style_intensity=1.0,
                           seed=42, debug=False):
        """Generate complete dance animation sequence"""
        
        # Set random seeds for reproducible results
        random.seed(seed)
        np.random.seed(seed)
        
        if debug:
            print(f"🎵 Starting Advanced Dance Poser")
            print(f"   Style: {dance_style}, Intensity: {movement_intensity}")
            print(f"   Facial: {enable_facial_expressions}, Coordination: {enable_coordinated_movement}")
        
        # === PHASE 1: AUDIO PROCESSING ===
        waveform = self.audio_analyzer.process_audio(audio, start_time, max_duration, debug)
        sample_rate = audio["sample_rate"]
        
        trimmed_audio = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0).to(audio["waveform"].device),
            "sample_rate": sample_rate
        }
        
        # === PHASE 2: AUDIO ANALYSIS ===
        features = self.audio_analyzer.analyze_for_dance(waveform, sample_rate, fps)
        
        if debug:
            print(f"   Audio: {features['duration']:.1f}s, {features['total_frames']} frames, BPM: {features['bpm']:.1f}")
        
        # === PHASE 3: DANCE SEQUENCE GENERATION ===
        pose_sequence = self._create_coordinated_dance_sequence(
            features, dance_style, movement_intensity, smoothing, beat_sensitivity, tempo_factor,
            enable_facial_expressions, facial_expression_intensity,
            arm_movement_style, leg_movement_style, hand_gesture_style, foot_pattern_style,
            enable_coordinated_movement, enable_beat_emphasis, enable_auto_gesture_change,
            gesture_change_interval, fps, debug
        )
        
        # === PHASE 4: PROFESSIONAL RENDERING ===
        frames = self.renderer.render_pose_sequence(
            pose_sequence, width, height, background_color,
            show_hands, show_face_details, show_visual_effects,
            features["beat_strength"], dance_style, debug
        )
        
        if debug:
            print(f"✅ Generated {len(frames)} frames successfully!")
        
        return (frames, trimmed_audio)

    def _create_coordinated_dance_sequence(self, features, dance_style, movement_intensity, smoothing,
                                         beat_sensitivity, tempo_factor, enable_facial_expressions,
                                         facial_expression_intensity, arm_movement_style, leg_movement_style,
                                         hand_gesture_style, foot_pattern_style, enable_coordinated_movement,
                                         enable_beat_emphasis, enable_auto_gesture_change, gesture_change_interval,
                                         fps, debug):
        """Create coordinated dance sequence using all animation modules"""
        
        total_frames = features["total_frames"]
        tempo_multiplier = {"1x": 1.0, "0.5x": 0.5, "0.25x": 0.25, "2x": 2.0}[tempo_factor]
        gesture_change_frames = int(gesture_change_interval * fps)
        
        pose_sequence = []
        prev_pose = np.array(self.skeleton_controller.base_pose)
        
        # Initialize animation state
        current_body_gesture = None
        current_hand_gesture = None
        current_foot_pattern = None
        gesture_intensity = 0.0
        
        for frame_idx in range(total_frames):
            
            # === GET CURRENT AUDIO SIGNALS ===
            frame_signals = {
                "beat": features["beat_strength"][frame_idx],
                "low": features["low_band"][frame_idx],
                "mid": features["mid_band"][frame_idx],
                "high": features["high_band"][frame_idx]
            }
            
            # Start with base pose
            pose = np.array(self.skeleton_controller.base_pose)
            
            # === AUTOMATIC GESTURE CHANGES ===
            if enable_auto_gesture_change and (frame_idx % gesture_change_frames == 0 or frame_signals["beat"] > 0.8):
                current_body_gesture = self._select_body_gesture(dance_style, arm_movement_style, frame_signals["beat"])
                if hand_gesture_style == "auto":
                    current_hand_gesture = self._select_hand_gesture(dance_style, frame_signals["beat"])
                else:
                    current_hand_gesture = hand_gesture_style
                    
                if foot_pattern_style == "auto":
                    current_foot_pattern = self._select_foot_pattern(dance_style, leg_movement_style, frame_signals["beat"])
                else:
                    current_foot_pattern = foot_pattern_style
            
            # === CALCULATE MOVEMENT INTENSITY ===
            base_intensity = movement_intensity
            beat_boost = frame_signals["beat"] * 0.3 if enable_beat_emphasis else 0.0
            total_intensity = min(1.0, base_intensity + beat_boost)
            
            # === APPLY FACIAL EXPRESSIONS ===
            if enable_facial_expressions:
                pose = self.facial_driver.apply_expression(
                    pose, features["tonality"], frame_idx, facial_expression_intensity
                )
            
            # === APPLY BODY GESTURES ===
            if current_body_gesture:
                pose = self.body_driver.apply_gesture(pose, current_body_gesture, total_intensity)
            
            # === APPLY HAND GESTURES ===
            if current_hand_gesture and show_hands:
                pose = self.hand_controller.apply_hand_gesture(pose, current_hand_gesture, total_intensity)
            
            # === APPLY COORDINATED FOOT/LEG MOVEMENT ===
            if enable_coordinated_movement and current_foot_pattern:
                # Update step phase based on beat
                self.foot_controller.update_step_phase(frame_signals["beat"], tempo_multiplier)
                
                # Apply foot pattern
                pose = self.foot_controller.apply_step_pattern(
                    pose, current_foot_pattern, self.foot_controller.current_step_phase,
                    total_intensity, frame_signals["beat"]
                )
                
                # Add tapping on strong beats
                if frame_signals["beat"] > beat_sensitivity:
                    step_pattern, tap_pattern, stance_pattern = self.foot_controller.get_coordinated_movement(
                        dance_style, frame_signals["beat"], frame_idx
                    )
                    
                    if tap_pattern:
                        pose = self.foot_controller.apply_tap_pattern(
                            pose, tap_pattern, total_intensity, frame_signals["beat"]
                        )
                    
                    if stance_pattern:
                        pose = self.foot_controller.apply_stance_pattern(
                            pose, stance_pattern, total_intensity * 0.5
                        )
            
            # === APPLY DANCE STYLE MODIFICATIONS ===
            pose = self._apply_dance_style_effects(pose, dance_style, frame_idx, total_intensity, frame_signals)
            
            # === APPLY SMOOTHING ===
            if frame_idx > 0:
                pose = smoothing * pose + (1.0 - smoothing) * prev_pose
            
            # === BOUNDS CHECKING ===
            pose = np.clip(pose, 0.0, 1.0)
            
            pose_sequence.append(pose.tolist())
            prev_pose = pose.copy()
            
            # Debug output for key frames
            if debug and frame_idx % (fps * 2) == 0:  # Every 2 seconds
                print(f"   Frame {frame_idx}: Beat={frame_signals['beat']:.2f}, "
                      f"Gesture={current_body_gesture}, Hand={current_hand_gesture}")
        
        return pose_sequence

    def _select_body_gesture(self, dance_style: str, arm_style: str, beat_strength: float) -> str:
        """Intelligently select body gesture based on style and beat"""
        
        if arm_style == "gestures":
            # Use predefined gesture sequences for each dance style
            gesture_sequences = {
                'energetic': ['pump_fists', 'hands_up_victory', 'jazz_hands', 'clap_hands', 'wave_hands'],
                'smooth': ['peace_sign', 'thumbs_up', 'pointing_forward', 'reach_up_left', 'reach_up_right'],
                'dramatic': ['hands_up_victory', 'arms_crossed', 'pump_fists', 'reach_up_left', 'reach_up_right'],
                'robot': ['pointing_forward', 'arms_crossed', 'hands_on_hips', 'pump_fists'],
                'bounce': ['running_man_left', 'running_man_right', 'disco_point_up', 'disco_point_down']
            }
            
            gestures = gesture_sequences.get(dance_style, gesture_sequences['energetic'])
            
            # High beat = more dynamic gestures
            if beat_strength > 0.7:
                dynamic_gestures = ['pump_fists', 'hands_up_victory', 'jazz_hands', 'wave_hands']
                available = [g for g in gestures if g in dynamic_gestures]
                return random.choice(available) if available else random.choice(gestures)
            else:
                return random.choice(gestures)
        else:
            # Use traditional arm styles
            return arm_style

    def _select_hand_gesture(self, dance_style: str, beat_strength: float) -> str:
        """Select appropriate hand gesture"""
        
        style_hand_mapping = {
            'energetic': ['open_palm', 'rock_horns', 'peace_sign', 'thumbs_up'],
            'smooth': ['open_palm', 'peace_sign', 'wave', 'thumbs_up'],
            'dramatic': ['closed_fist', 'pointing', 'open_palm', 'peace_sign'],
            'robot': ['pointing', 'closed_fist', 'open_palm'],
            'bounce': ['open_palm', 'rock_horns', 'peace_sign', 'wave']
        }
        
        gestures = style_hand_mapping.get(dance_style, style_hand_mapping['energetic'])
        
        # Strong beats favor more expressive gestures
        if beat_strength > 0.6:
            expressive = ['rock_horns', 'peace_sign', 'open_palm']
            available = [g for g in gestures if g in expressive]
            return random.choice(available) if available else random.choice(gestures)
        
        return random.choice(gestures)

    def _select_foot_pattern(self, dance_style: str, leg_style: str, beat_strength: float) -> str:
        """Select appropriate foot movement pattern"""
        
        if leg_style in ['step', 'shuffle', 'march', 'bounce']:
            # Map leg style to foot pattern
            return f"{leg_style}_step"
        
        # Auto selection based on dance style
        style_foot_mapping = {
            'energetic': 'step_in_place',
            'smooth': 'shuffle_step', 
            'dramatic': 'march_step',
            'robot': 'step_in_place',
            'bounce': 'bounce_step'
        }
        
        return style_foot_mapping.get(dance_style, 'step_in_place')

    def _apply_dance_style_effects(self, pose: np.ndarray, dance_style: str, frame_idx: int, 
                                 intensity: float, signals: Dict) -> np.ndarray:
        """Apply dance style specific effects to the pose"""
        
        if dance_style == "robot":
            # Quantize movements for robotic effect
            for i in range(len(pose)):
                pose[i][0] = round(pose[i][0] * 20) / 20
                pose[i][1] = round(pose[i][1] * 20) / 20
                
        elif dance_style == "dramatic":
            # Exaggerate movements
            center_x, center_y = 0.5, 0.5
            for i in range(len(pose)):
                dx = pose[i][0] - center_x
                dy = pose[i][1] - center_y
                pose[i][0] = center_x + dx * (1.0 + 0.2 * intensity)
                pose[i][1] = center_y + dy * (1.0 + 0.15 * intensity)
                
        elif dance_style == "bounce":
            # Add synchronized bounce to all points
            bounce_amount = intensity * signals["beat"] * 0.08 * math.sin(frame_idx * 0.4)
            for i in range(len(pose)):
                pose[i][1] += bounce_amount
                
        elif dance_style == "smooth":
            # Add gentle sway
            sway_amount = intensity * 0.03 * math.sin(frame_idx * 0.1)
            for i in range(len(pose)):
                pose[i][0] += sway_amount * (0.5 + 0.5 * signals["low"])
        
        return pose

# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyUI_BASICSAdvancedDancePoser": ComfyUI_BASICSAdvancedDancePoser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_BASICSAdvancedDancePoser": "🕺 BASICS Advanced Dance Poser"
}