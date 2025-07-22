# core/audio_analyzer.py
# Handles all audio processing and feature extraction

import numpy as np
import librosa
import warnings
from typing import Dict

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