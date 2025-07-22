# ComfyUI_BASICSAdvancedDancePoser - Modular Architecture
# COCO-WholeBody 133 keypoint dance animation system

import torch
import numpy as np
import librosa
import cv2
import random
import math
import warnings
from typing import Dict, List, Tuple, Optional

# ============================================================================
# BODY ANIMATION DRIVER MODULE
# ============================================================================

class BodyAnimationDriver:
    """Handles body movement patterns and dance gestures for COCO-WholeBody"""
    
    def __init__(self):
        self.gesture_templates = self._create_gesture_templates()
        self.dance_patterns = self._create_dance_patterns()
        
    def _create_gesture_templates(self) -> Dict:
        """Create comprehensive gesture templates using COCO body keypoints (0-16)"""
        
        # Base neutral positions for reference
        base_positions = {
            'left_shoulder': [0.42, 0.25],   # 5
            'right_shoulder': [0.58, 0.25],  # 6  
            'left_elbow': [0.35, 0.40],      # 7
            'right_elbow': [0.65, 0.40],     # 8
            'left_wrist': [0.28, 0.55],      # 9
            'right_wrist': [0.72, 0.55],     # 10
            'left_hip': [0.44, 0.55],        # 11
            'right_hip': [0.56, 0.55],       # 12
        }
        
        return {
            # === ARM GESTURES ===
            'hands_up_victory': {
                'left_elbow': [0.38, 0.15],    # Arms up high
                'right_elbow': [0.62, 0.15], 
                'left_wrist': [0.40, 0.05],    # Hands up triumphant
                'right_wrist': [0.60, 0.05]
            },
            
            'hands_on_hips': {
                'left_elbow': [0.35, 0.45],    # Elbows out
                'right_elbow': [0.65, 0.45],
                'left_wrist': [0.42, 0.55],    # Hands on hips
                'right_wrist': [0.58, 0.55]
            },
            
            'arms_crossed': {
                'left_elbow': [0.52, 0.35],    # Cross body
                'right_elbow': [0.48, 0.35],
                'left_wrist': [0.58, 0.30],    # Hands on opposite arms
                'right_wrist': [0.42, 0.30]
            },
            
            'clap_hands': {
                'left_elbow': [0.45, 0.30],    # Elbows in for clap
                'right_elbow': [0.55, 0.30],
                'left_wrist': [0.48, 0.25],    # Hands together
                'right_wrist': [0.52, 0.25]
            },
            
            'pump_fists': {
                'left_elbow': [0.40, 0.20],    # Power pose
                'right_elbow': [0.60, 0.20],
                'left_wrist': [0.38, 0.10],    # Fists up
                'right_wrist': [0.62, 0.10]
            },
            
            'reach_up_left': {
                'left_elbow': [0.35, 0.15],    # Reach high
                'right_elbow': [0.65, 0.40],   # Other arm normal
                'left_wrist': [0.30, 0.05],    # Left hand up
                'right_wrist': [0.72, 0.55]    # Right hand normal
            },
            
            'reach_up_right': {
                'left_elbow': [0.35, 0.40],    # Left arm normal
                'right_elbow': [0.65, 0.15],   # Reach high
                'left_wrist': [0.28, 0.55],    # Left hand normal
                'right_wrist': [0.70, 0.05]    # Right hand up
            },
            
            'pointing_forward': {
                'left_elbow': [0.35, 0.40],    # Left normal
                'right_elbow': [0.60, 0.25],   # Right pointing
                'left_wrist': [0.28, 0.55],    # Left normal
                'right_wrist': [0.75, 0.20]    # Point forward
            },
            
            'peace_sign': {
                'left_elbow': [0.38, 0.30],    # Both arms up casual
                'right_elbow': [0.62, 0.30],
                'left_wrist': [0.35, 0.20],    # Peace signs
                'right_wrist': [0.65, 0.20]
            },
            
            'thumbs_up': {
                'left_elbow': [0.38, 0.35],    # Casual thumbs up
                'right_elbow': [0.62, 0.35],
                'left_wrist': [0.30, 0.30],    # Thumbs up
                'right_wrist': [0.70, 0.30]
            },
            
            # === DANCE MOVES ===
            'disco_point_up': {
                'left_elbow': [0.35, 0.40],    # Left normal
                'right_elbow': [0.62, 0.15],   # Right up diagonal
                'left_wrist': [0.28, 0.55],    # Left down
                'right_wrist': [0.68, 0.08]    # Point up-right
            },
            
            'disco_point_down': {
                'left_elbow': [0.38, 0.15],    # Left up diagonal  
                'right_elbow': [0.65, 0.40],   # Right normal
                'left_wrist': [0.32, 0.08],    # Point up-left
                'right_wrist': [0.72, 0.55]    # Right down
            },
            
            'running_man_left': {
                'left_elbow': [0.30, 0.35],    # Left arm forward
                'right_elbow': [0.70, 0.45],   # Right arm back
                'left_wrist': [0.25, 0.45],    # Hands in running position
                'right_wrist': [0.75, 0.35]
            },
            
            'running_man_right': {
                'left_elbow': [0.30, 0.45],    # Left arm back
                'right_elbow': [0.70, 0.35],   # Right arm forward
                'left_wrist': [0.25, 0.35],    # Opposite running position
                'right_wrist': [0.75, 0.45]
            },
            
            'wave_hands': {
                'left_elbow': [0.30, 0.25],    # Both arms up for waving
                'right_elbow': [0.70, 0.25],
                'left_wrist': [0.25, 0.15],    # Hands up waving
                'right_wrist': [0.75, 0.15]
            },
            
            'jazz_hands': {
                'left_elbow': [0.25, 0.30],    # Wide jazz hands
                'right_elbow': [0.75, 0.30],
                'left_wrist': [0.15, 0.25],    # Spread wide
                'right_wrist': [0.85, 0.25]
            }
        }
    
    def _create_dance_patterns(self) -> Dict:
        """Create sequence patterns for different dance styles"""
        return {
            'energetic': [
                'pump_fists', 'hands_up_victory', 'jazz_hands', 'wave_hands',
                'disco_point_up', 'disco_point_down', 'clap_hands'
            ],
            'smooth': [
                'peace_sign', 'pointing_forward', 'reach_up_left', 'reach_up_right',
                'thumbs_up', 'wave_hands'
            ],
            'dramatic': [
                'hands_up_victory', 'arms_crossed', 'pump_fists', 'jazz_hands',
                'reach_up_left', 'reach_up_right'
            ],
            'robot': [
                'pointing_forward', 'arms_crossed', 'hands_on_hips', 'pump_fists'
            ],
            'bounce': [
                'running_man_left', 'running_man_right', 'disco_point_up', 
                'disco_point_down', 'clap_hands'
            ]
        }
    
    def apply_gesture(self, pose: np.ndarray, gesture_name: str, intensity: float = 1.0) -> np.ndarray:
        """Apply a specific gesture to the pose"""
        if gesture_name not in self.gesture_templates:
            return pose
            
        gesture = self.gesture_templates[gesture_name]
        
        # Apply gesture modifications to specific keypoints
        keypoint_mapping = {
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12
        }
        
        for joint_name, target_pos in gesture.items():
            if joint_name in keypoint_mapping:
                idx = keypoint_mapping[joint_name]
                if idx < len(pose):
                    # Blend between current and target position
                    current_pos = np.array(pose[idx])
                    target_pos = np.array(target_pos)
                    blended_pos = current_pos + (target_pos - current_pos) * intensity
                    pose[idx] = blended_pos.tolist()
        
        return pose
    
    def get_dance_sequence(self, dance_style: str, beat_strength: float, frame_idx: int) -> str:
        """Get appropriate gesture for current frame based on dance style and beat"""
        if dance_style not in self.dance_patterns:
            dance_style = 'energetic'
            
        patterns = self.dance_patterns[dance_style]
        
        # Change gesture on strong beats
        if beat_strength > 0.7:
            return patterns[frame_idx % len(patterns)]
        elif beat_strength > 0.4:
            # Hold current gesture longer on weaker beats
            return patterns[(frame_idx // 3) % len(patterns)]
        else:
            # Neutral poses for quiet sections
            return 'peace_sign' if frame_idx % 2 == 0 else 'thumbs_up'

# ============================================================================
# HAND GESTURE CONTROLLER MODULE  
# ============================================================================

class HandGestureController:
    """Manages detailed hand animations using COCO-WholeBody hand keypoints"""
    
    def __init__(self):
        self.left_hand_start = 91   # COCO-WholeBody left hand indices 91-111
        self.right_hand_start = 112 # COCO-WholeBody right hand indices 112-132
        self.hand_gestures = self._create_hand_gestures()
        
    def _create_hand_gestures(self) -> Dict:
        """Create detailed hand gesture patterns"""
        return {
            'open_palm': self._create_open_palm(),
            'closed_fist': self._create_closed_fist(),
            'pointing': self._create_pointing(),
            'peace_sign': self._create_peace_hand(),
            'thumbs_up': self._create_thumbs_up(),
            'rock_horns': self._create_rock_horns(),
            'wave': self._create_wave_hand()
        }
    
    def _create_open_palm(self) -> Dict:
        """Open palm gesture - fingers spread"""
        # Simplified: spread fingers from wrist center
        left_offsets = []
        right_offsets = []
        
        for i in range(21):  # 21 hand keypoints each
            if i == 0:  # Wrist
                left_offsets.append([0.0, 0.0])
                right_offsets.append([0.0, 0.0])
            else:
                finger = (i - 1) // 4
                joint = (i - 1) % 4
                
                # Spread fingers more for open palm
                finger_spread = 0.15
                finger_angles = [-60, -30, 0, 30, 60]  # degrees
                
                angle = np.radians(finger_angles[finger])
                extension = 0.02 * (joint + 1)  # Each joint extends further
                
                x_offset = extension * np.sin(angle) * finger_spread
                y_offset = -extension * np.cos(angle)  # Fingers point down/forward
                
                left_offsets.append([x_offset, y_offset])
                right_offsets.append([-x_offset, y_offset])  # Mirror for right hand
        
        return {'left': left_offsets, 'right': right_offsets}
    
    def _create_closed_fist(self) -> Dict:
        """Closed fist gesture"""
        # Fingers curled in toward palm
        left_offsets = []
        right_offsets = []
        
        for i in range(21):
            if i == 0:  # Wrist
                left_offsets.append([0.0, 0.0])
                right_offsets.append([0.0, 0.0])
            else:
                finger = (i - 1) // 4
                joint = (i - 1) % 4
                
                # Curl fingers inward
                curl_factor = 0.01 * (joint + 1)
                x_offset = 0.005 * (finger - 2)  # Slight finger spread
                y_offset = curl_factor * 0.5  # Curl toward palm
                
                left_offsets.append([x_offset, y_offset])
                right_offsets.append([-x_offset, y_offset])
        
        return {'left': left_offsets, 'right': right_offsets}
    
    def _create_pointing(self) -> Dict:
        """Pointing gesture - index finger extended"""
        # Similar to fist but index finger extended
        left_offsets = []
        right_offsets = []
        
        for i in range(21):
            if i == 0:
                left_offsets.append([0.0, 0.0])
                right_offsets.append([0.0, 0.0])
            else:
                finger = (i - 1) // 4
                joint = (i - 1) % 4
                
                if finger == 1:  # Index finger - extend
                    extension = 0.025 * (joint + 1)
                    left_offsets.append([0.0, -extension])
                    right_offsets.append([0.0, -extension])
                else:  # Other fingers - curl
                    curl = 0.01 * (joint + 1)
                    x_offset = 0.005 * (finger - 2)
                    left_offsets.append([x_offset, curl * 0.5])
                    right_offsets.append([-x_offset, curl * 0.5])
        
        return {'left': left_offsets, 'right': right_offsets}
    
    def _create_peace_hand(self) -> Dict:
        """Peace sign - index and middle finger up"""
        left_offsets = []
        right_offsets = []
        
        for i in range(21):
            if i == 0:
                left_offsets.append([0.0, 0.0])
                right_offsets.append([0.0, 0.0])
            else:
                finger = (i - 1) // 4
                joint = (i - 1) % 4
                
                if finger in [1, 2]:  # Index and middle - extend
                    angle = np.radians(-20 if finger == 1 else 20)
                    extension = 0.02 * (joint + 1)
                    x_offset = extension * np.sin(angle)
                    y_offset = -extension * np.cos(angle)
                    left_offsets.append([x_offset, y_offset])
                    right_offsets.append([-x_offset, y_offset])
                else:  # Other fingers - curl
                    curl = 0.01 * (joint + 1)
                    x_offset = 0.005 * (finger - 2)
                    left_offsets.append([x_offset, curl * 0.5])
                    right_offsets.append([-x_offset, curl * 0.5])
        
        return {'left': left_offsets, 'right': right_offsets}
    
    def _create_thumbs_up(self) -> Dict:
        """Thumbs up gesture"""
        left_offsets = []
        right_offsets = []
        
        for i in range(21):
            if i == 0:
                left_offsets.append([0.0, 0.0])
                right_offsets.append([0.0, 0.0])
            else:
                finger = (i - 1) // 4
                joint = (i - 1) % 4
                
                if finger == 0:  # Thumb - point up
                    extension = 0.02 * (joint + 1)
                    left_offsets.append([-0.01, -extension])
                    right_offsets.append([0.01, -extension])
                else:  # Other fingers - curl
                    curl = 0.015 * (joint + 1)
                    x_offset = 0.005 * (finger - 2)
                    left_offsets.append([x_offset, curl])
                    right_offsets.append([-x_offset, curl])
        
        return {'left': left_offsets, 'right': right_offsets}
    
    def _create_rock_horns(self) -> Dict:
        """Rock horns - index and pinky extended"""
        left_offsets = []
        right_offsets = []
        
        for i in range(21):
            if i == 0:
                left_offsets.append([0.0, 0.0])
                right_offsets.append([0.0, 0.0])
            else:
                finger = (i - 1) // 4
                joint = (i - 1) % 4
                
                if finger in [1, 4]:  # Index and pinky
                    angle = np.radians(-30 if finger == 1 else 30)
                    extension = 0.025 * (joint + 1)
                    x_offset = extension * np.sin(angle)
                    y_offset = -extension * np.cos(angle)
                    left_offsets.append([x_offset, y_offset])
                    right_offsets.append([-x_offset, y_offset])
                else:  # Other fingers - curl
                    curl = 0.01 * (joint + 1)
                    x_offset = 0.003 * (finger - 2)
                    left_offsets.append([x_offset, curl * 0.7])
                    right_offsets.append([-x_offset, curl * 0.7])
        
        return {'left': left_offsets, 'right': right_offsets}
    
    def _create_wave_hand(self) -> Dict:
        """Waving hand - fingers slightly spread and angled"""
        left_offsets = []
        right_offsets = []
        
        for i in range(21):
            if i == 0:
                left_offsets.append([0.0, 0.0])
                right_offsets.append([0.0, 0.0])
            else:
                finger = (i - 1) // 4
                joint = (i - 1) % 4
                
                # Slight curve and spread for natural wave
                finger_angles = [-40, -20, 0, 20, 40]
                angle = np.radians(finger_angles[finger])
                extension = 0.018 * (joint + 1)
                
                x_offset = extension * np.sin(angle) * 0.8
                y_offset = -extension * np.cos(angle) * 0.9
                
                left_offsets.append([x_offset, y_offset])
                right_offsets.append([-x_offset, y_offset])
        
        return {'left': left_offsets, 'right': right_offsets}
    
    def apply_hand_gesture(self, pose: np.ndarray, gesture_name: str, intensity: float = 1.0) -> np.ndarray:
        """Apply hand gesture to pose"""
        if gesture_name not in self.hand_gestures:
            return pose
            
        gesture = self.hand_gestures[gesture_name]
        
        # Apply to left hand (indices 91-111)
        for i, offset in enumerate(gesture['left']):
            idx = self.left_hand_start + i
            if idx < len(pose):
                current_pos = np.array(pose[idx])
                offset_pos = np.array(offset) * intensity
                pose[idx] = (current_pos + offset_pos).tolist()
        
        # Apply to right hand (indices 112-132)
        for i, offset in enumerate(gesture['right']):
            idx = self.right_hand_start + i
            if idx < len(pose):
                current_pos = np.array(pose[idx])
                offset_pos = np.array(offset) * intensity
                pose[idx] = (current_pos + offset_pos).tolist()
        
        return pose

# ============================================================================
# COCO-WHOLEBODY CONSTANTS
# ============================================================================

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

# ============================================================================
# AUDIO ANALYSIS MODULE
# ============================================================================

class AudioAnalyzer:
    """Handles all audio processing and feature extraction"""
    
    def __init__(self):
        self.sample_rate = 44100
        self.hop_length = 512
        
    def process_audio(self, audio: Dict, start_time: float, max_duration: float, debug: bool = False) -> np.ndarray:
        """Process and trim audio waveform"""
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
            
        # Handle different waveform shapes
        if waveform.ndim == 3 and waveform.shape[0] >= 1:
            waveform = waveform[0]
        if waveform.ndim == 2:
            if waveform.shape[0] == 2:
                waveform = np.mean(waveform, axis=0)
            elif waveform.shape[0] == 1:
                waveform = waveform[0]
                
        total_duration = len(waveform) / sample_rate
        start_sample = int(max(0, start_time) * sample_rate)
        max_samples = int(min(max_duration, 180.0) * sample_rate)
        end_sample = min(start_sample + max_samples, len(waveform))
        
        return waveform[start_sample:end_sample]
    
    def analyze_for_dance(self, waveform: np.ndarray, sample_rate: int, fps: int) -> Dict:
        """Extract comprehensive dance features from audio"""
        try:
            # Tempo and beat analysis
            tempo = librosa.beat.tempo(y=waveform, sr=sample_rate)
            bpm = np.clip(float(tempo[0]) if hasattr(tempo, '__getitem__') else float(tempo), 60, 200)
            
            _, beat_frames = librosa.beat.beat_track(y=waveform, sr=sample_rate, units='frames', hop_length=self.hop_length)
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=self.hop_length)
            
            # Spectral analysis
            S = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=int(sample_rate / fps)))
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
            
            # Frequency bands
            low_mask = (freqs >= 20) & (freqs < 250)
            mid_mask = (freqs >= 250) & (freqs < 2000) 
            high_mask = (freqs >= 2000) & (freqs < 8000)
            
            low_energy = self._safe_normalize(S[low_mask, :].mean(axis=0))
            mid_energy = self._safe_normalize(S[mid_mask, :].mean(axis=0))
            high_energy = self._safe_normalize(S[high_mask, :].mean(axis=0))
            
            # Tonality analysis for facial expressions
            chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
            tonality_features = self._analyze_tonality(chroma)
            
            # Map to frame indices
            duration = len(waveform) / sample_rate
            total_frames = int(duration * fps)
            beat_strength = np.zeros(total_frames)
            
            for bt in beat_times:
                idx = int((bt / duration) * total_frames)
                if 0 <= idx < total_frames:
                    beat_strength[idx] = 1.0
            
            energy_len = S.shape[1]
            frame_idxs = np.linspace(0, energy_len - 1, total_frames).astype(np.int32)
            frame_idxs = np.clip(frame_idxs, 0, energy_len - 1)
            
            return {
                "beat_strength": beat_strength,
                "low_band": low_energy[frame_idxs],
                "mid_band": mid_energy[frame_idxs], 
                "high_band": high_energy[frame_idxs],
                "tonality": tonality_features,
                "total_frames": total_frames,
                "duration": duration,
                "bpm": bpm
            }
            
        except Exception as e:
            return self._create_fallback_features(waveform, sample_rate, fps)
    
    def _analyze_tonality(self, chroma: np.ndarray) -> Dict:
        """Analyze musical tonality for facial expressions"""
        # Major/minor detection
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        
        # Calculate correlation with major/minor profiles over time
        chroma_norm = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)
        major_correlation = np.dot(major_profile, chroma_norm)
        minor_correlation = np.dot(minor_profile, chroma_norm)
        
        return {
            "major_strength": major_correlation,
            "minor_strength": minor_correlation,
            "brightness": np.mean(chroma_norm[4:8], axis=0),  # Focus on brighter notes
            "darkness": np.mean(chroma_norm[0:4], axis=0)     # Focus on darker notes
        }
    
    def _safe_normalize(self, arr: np.ndarray) -> np.ndarray:
        """Safely normalize array"""
        max_val = arr.max()
        return arr / max_val if max_val > 0 else np.zeros_like(arr)
    
    def _create_fallback_features(self, waveform: np.ndarray, sample_rate: int, fps: int) -> Dict:
        """Create fallback features if analysis fails"""
        duration = len(waveform) / sample_rate
        total_frames = int(duration * fps)
        amplitude = np.abs(waveform) / (np.max(np.abs(waveform)) + 1e-8)
        
        return {
            "beat_strength": np.zeros(total_frames),
            "low_band": amplitude[:total_frames] if len(amplitude) >= total_frames else np.zeros(total_frames),
            "mid_band": amplitude[:total_frames] if len(amplitude) >= total_frames else np.zeros(total_frames),
            "high_band": (amplitude[:total_frames] * 0.3) if len(amplitude) >= total_frames else np.zeros(total_frames),
            "tonality": {
                "major_strength": np.zeros(total_frames),
                "minor_strength": np.zeros(total_frames),
                "brightness": np.zeros(total_frames),
                "darkness": np.zeros(total_frames)
            },
            "total_frames": total_frames,
            "duration": duration,
            "bpm": 120.0
        }

# ============================================================================
# SKELETON CONTROLLER MODULE  
# ============================================================================

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

# ============================================================================
# FACIAL EXPRESSION DRIVER MODULE
# ============================================================================

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

# ============================================================================
# MAIN NODE CLASS
# ============================================================================

class ComfyUI_BASICSAdvancedDancePoser:
    """Advanced dance poser with COCO-WholeBody 133 keypoints and facial expressions"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "max_duration": ("FLOAT", {"default": 180.0, "min": 3.0, "max": 180.0, "step": 0.1}),
                "width": ("INT", {"default": 480, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 832, "min": 64, "max": 2048}),
                "fps": ("INT", {"default": 24, "min": 12, "max": 60}),
                "dance_style": (["energetic", "smooth", "dramatic", "robot", "bounce"], {"default": "energetic"}),
                "movement_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smoothing": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_facial_expressions": ("BOOLEAN", {"default": True}),
                "facial_expression_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "show_face_details": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "background_color": (["black", "white", "gray"], {"default": "black"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 99999}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("pose_video", "audio")
    FUNCTION = "generate_pose_video"
    CATEGORY = "BASICS/Audio"
    
    def __init__(self):
        self.audio_analyzer = AudioAnalyzer()
        self.skeleton_controller = SkeletonController() 
        self.facial_driver = FacialExpressionDriver()
    
    def generate_pose_video(self, audio, start_time, max_duration, width, height, fps,
                           dance_style, movement_intensity, smoothing,
                           enable_facial_expressions, facial_expression_intensity,
                           show_hands, show_face_details,
                           background_color="black", seed=42, debug=False):
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Process audio
        waveform = self.audio_analyzer.process_audio(audio, start_time, max_duration, debug)
        sample_rate = audio["sample_rate"]
        
        trimmed_audio = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0).to(audio["waveform"].device),
            "sample_rate": sample_rate
        }
        
        # Analyze audio features
        features = self.audio_analyzer.analyze_for_dance(waveform, sample_rate, fps)
        
        # Generate pose sequence (simplified for now - will expand)
        pose_sequence = self._create_dance_sequence(features, dance_style, movement_intensity, 
                                                   enable_facial_expressions, facial_expression_intensity,
                                                   smoothing, fps, debug)
        
        # Render frames (simplified for now - will expand) 
        frames = self._render_pose_sequence(pose_sequence, width, height, background_color,
                                           show_hands, show_face_details, debug)
        
        return (frames, trimmed_audio)
    
    def _create_dance_sequence(self, features, dance_style, movement_intensity,
                              enable_facial_expressions, facial_expression_intensity,
                              smoothing, fps, debug):
        """Create dance sequence with all modules"""
        total_frames = features["total_frames"]
        pose_sequence = []
        prev_pose = np.array(self.skeleton_controller.base_pose)
        
        for frame_i in range(total_frames):
            pose = np.array(self.skeleton_controller.base_pose)
            
            # Apply facial expressions based on tonality
            if enable_facial_expressions:
                pose = self.facial_driver.apply_expression(
                    pose, features["tonality"], frame_i, facial_expression_intensity
                )
            
            # TODO: Add body animation drivers here
            # - BodyAnimationDriver for torso, arms, legs
            # - Hand gesture system
            # - Foot movement patterns
            
            # Apply smoothing
            if frame_i > 0:
                pose = smoothing * pose + (1.0 - smoothing) * prev_pose
                
            pose = np.clip(pose, 0.0, 1.0)
            pose_sequence.append(pose.tolist())
            prev_pose = pose.copy()
            
        return pose_sequence
    
    def _render_pose_sequence(self, pose_sequence, width, height, background_color,
                             show_hands, show_face_details, debug):
        """Render pose sequence to frames"""
        # Simplified rendering - will expand this
        bg_colors = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128)}
        bg_color = bg_colors.get(background_color, (0, 0, 0))
        
        frames = []
        connections = self.skeleton_controller.get_all_connections()
        
        for pose in pose_sequence:
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            keypoints = [(int(x * width), int(y * height)) for x, y in pose]
            
            # Draw skeleton connections
            for connection in connections:
                if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                    try:
                        cv2.line(frame, keypoints[connection[0]], keypoints[connection[1]], 
                                (255, 255, 255), 2)
                    except:
                        continue
            
            # Draw keypoints
            for i, point in enumerate(keypoints):
                color = (0, 255, 0) if i < 17 else (255, 0, 0)  # Green for body, red for others
                cv2.circle(frame, point, 2, color, -1)
            
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
            
        return torch.stack(frames)

# Node registration
NODE_CLASS_MAPPINGS = {"ComfyUI_BASICSAdvancedDancePoser": ComfyUI_BASICSAdvancedDancePoser}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyUI_BASICSAdvancedDancePoser": "BASICS Advanced Dance Poser"}