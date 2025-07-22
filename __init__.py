# __init__.py
from .advanced_dance_poser import ComfyUI_BASICSAdvancedDancePoser

NODE_CLASS_MAPPINGS = {
    "ComfyUI_BASICSAdvancedDancePoser": ComfyUI_BASICSAdvancedDancePoser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_BASICSAdvancedDancePoser": "🕺 BASICS Advanced Dance Poser"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]