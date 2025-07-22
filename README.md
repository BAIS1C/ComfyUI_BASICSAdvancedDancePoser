# 🕺 ComfyUI_BASICSAdvancedDancePoser

**Professional COCO-WholeBody 133-keypoint dance animation system for ComfyUI**

Transform any audio into synchronized dance animations with intelligent gesture recognition, musical facial expressions, and beat-responsive visual effects.

![Dance Animation Preview](https://via.placeholder.com/800x400/000000/FFFFFF?text=🕺+Dance+Animation+Preview)

## ✨ Features

### 🎵 **Musical Intelligence**
- **Beat Detection** - Automatic rhythm analysis drives movement intensity
- **Tonality Analysis** - Major/minor key detection creates appropriate facial expressions
- **Frequency Band Mapping** - Bass, mids, and highs control different body parts
- **BPM Analysis** - Automatic tempo detection (60-200 BPM supported)

### 💃 **Professional Gesture Library**
- **Victory Poses** - Hands up triumphant, pump fists
- **Classic Gestures** - Peace signs, thumbs up, pointing, clapping
- **Dance Moves** - Jazz hands, disco pointing, running man, wave hands
- **Power Poses** - Arms crossed, hands on hips, reaching gestures
- **Hand Articulation** - Full 21-keypoint finger movement per hand

### 🎭 **Facial Expression System**
- **Happy** - Major keys + bright tones trigger smiles
- **Sad** - Minor keys + dark tones create contemplative expressions  
- **Surprised** - High brightness creates wide-eyed looks
- **Concentrated** - Complex music triggers focused expressions

### 🦶 **Coordinated Movement**
- **Foot Patterns** - Step in place, marching, shuffling, bouncing
- **Leg Coordination** - Knees and ankles move naturally with feet
- **Beat Tapping** - Toe taps and heel strikes on strong beats
- **Weight Distribution** - Realistic balance and support leg adjustments

### 🎨 **Professional Visualization**
- **Visual Hierarchy** - Different colors and line weights for body parts
- **Beat-Responsive Effects** - Colors pulse and glow with music
- **Dance Style Modifiers** - Each style has unique visual personality
- **Multiple Color Schemes** - Black, white, and gray background support

## 🎯 Dance Styles

| Style | Characteristics | Gestures | Movement |
|-------|----------------|----------|----------|
| **Energetic** | High-energy, dynamic | Pump fists, victory poses, jazz hands | Quick transitions, beat emphasis |
| **Smooth** | Flowing, graceful | Peace signs, gentle reaches, thumbs up | Fluid movements, sway |
| **Dramatic** | Powerful, expressive | Victory poses, arms crossed, power gestures | Exaggerated movements |
| **Robot** | Mechanical, precise | Pointing, structured poses | Quantized movements |
| **Bounce** | Rhythmic, bouncy | Running man, disco points | Synchronized bouncing |

## 🚀 Quick Start

1. **Install** the node in your ComfyUI custom_nodes folder
2. **Restart** ComfyUI
3. **Find** the node: `BASICS` → `Audio` → `🕺 BASICS Advanced Dance Poser`
4. **Connect** your audio input
5. **Choose** your dance style
6. **Generate** amazing dance animations!

## 📋 Requirements

- **ComfyUI** (latest version recommended)
- **Python Libraries**:
  - `torch` (for tensor operations)
  - `numpy` (for numerical computations)
  - `librosa` (for audio analysis)
  - `opencv-python` (for video rendering)

## 🎛️ Parameters

### Core Settings
- **audio** - Input audio file (any ComfyUI audio format)
- **start_time** - Start position in seconds (0.0-3600.0)
- **max_duration** - Maximum clip length (3.0-180.0 seconds)
- **width/height** - Output video dimensions
- **fps** - Frame rate (12-60 fps)

### Dance Control
- **dance_style** - Overall animation style (energetic, smooth, dramatic, robot, bounce)
- **movement_intensity** - Base movement strength (0.0-1.0)
- **beat_sensitivity** - How responsive to musical beats (0.0-1.0)
- **tempo_factor** - Speed modifier (0.25x, 0.5x, 1x, 2x)

### Body Parts
- **arm_movement_style** - Arm animation type (flowing, sharp, waves, reaches, gestures)
- **hand_gesture_style** - Hand poses (auto, open_palm, peace_sign, rock_horns, etc.)
- **leg_movement_style** - Leg movement pattern (step, shuffle, march, bounce)
- **foot_pattern_style** - Foot coordination (auto, step_in_place, march_step, etc.)

### Visual Effects
- **enable_facial_expressions** - Musical facial expressions on/off
- **show_hands** - Display hand details
- **show_face_details** - Display facial features
- **show_visual_effects** - Beat-responsive glow and pulse effects
- **background_color** - Black, white, or gray background

## 🏗️ Architecture

The system uses a modular architecture with specialized components:

```
ComfyUI_BASICSAdvancedDancePoser/
├── advanced_dance_poser.py        # Main node orchestrator
├── core/
│   ├── audio_analyzer.py          # Musical analysis
│   ├── skeleton_controller.py     # COCO-WholeBody structure
│   ├── facial_expression_driver.py # Tonality → expressions
│   ├── body_animation_driver.py   # Gesture library
│   └── constants.py               # COCO keypoint definitions
├── patterns/
│   └── foot_movement_patterns.py  # Coordinated foot/leg movement
└── renderers/
    └── pose_renderer.py           # Professional visualization
```

## 🎨 Technical Details

### COCO-WholeBody Keypoints (133 total)
- **Body**: 17 keypoints (COCO standard)
- **Feet**: 6 keypoints (big toe, small toe, heel per foot)  
- **Face**: 68 keypoints (facial landmarks)
- **Hands**: 42 keypoints (21 per hand)

### Audio Analysis
- **Beat Tracking** - Librosa beat detection with frame-accurate timing
- **Spectral Analysis** - FFT-based frequency band separation
- **Chroma Features** - Musical key and tonality analysis
- **Temporal Smoothing** - Anti-aliasing for smooth animation

### Animation System
- **Gesture Templates** - Pre-defined pose libraries with anatomical accuracy
- **Blend Trees** - Smooth transitions between poses
- **Beat Synchronization** - Musical timing drives gesture changes
- **Coordinated Movement** - Body parts work together naturally

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional gesture libraries
- New dance styles
- Enhanced facial expressions
- Performance optimizations
- Visual effects

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **COCO-WholeBody** dataset for keypoint standards
- **Librosa** library for audio analysis capabilities
- **ComfyUI** community for the amazing framework
- **OpenPose** research for pose estimation foundations

## 🔧 Troubleshooting

### Common Issues

**"Audio too short" error**
- Ensure audio is at least 3 seconds long
- Check start_time doesn't exceed audio duration

**Missing dependencies**
```bash
pip install librosa opencv-python numpy torch
```

**Poor beat detection**
- Try different beat_sensitivity values (0.1-0.3 range)
- Ensure audio has clear rhythmic content
- Check audio quality and format

**Robotic movements**
- Increase smoothing parameter (0.7-0.9)
- Lower movement_intensity
- Try "smooth" dance style

### Performance Tips
- Use lower FPS (12-18) for faster generation
- Reduce max_duration for testing
- Disable visual_effects for better performance
- Use smaller output dimensions for drafts

## 📞 Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check existing discussions
- Review troubleshooting section

---

**Made with ❤️ for the ComfyUI community**

*Transform your audio into expressive dance animations with professional-grade skeletal animation and musical intelligence.*
