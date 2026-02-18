"""ComfyUI Stereo Gaussian - Stereoscopic VR rendering from 3D Gaussian Splats.

Renders stereo pairs from Gaussian splat PLY files (e.g., from SHARP Predict)
using the gsplat CUDA rasterizer, with optional PLY cleanup to save disk space.

Nodes:
    - StereoGaussianRender: Render left/right eye views with IPD offset
    - SBSConcat: Concatenate stereo pairs into SBS or Over-Under layout
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
