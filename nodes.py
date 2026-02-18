"""ComfyUI nodes for stereo Gaussian splat rendering.

Provides StereoGaussianRender and SBSConcat nodes for creating
stereoscopic VR content from 3D Gaussian splat PLY files.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .render import (
    build_intrinsics_4x4,
    build_stereo_extrinsics,
    concat_side_by_side,
    concat_top_bottom,
    load_gaussians,
    render_view,
)


class StereoGaussianRender:
    """Render stereo pair from a Gaussian splat PLY file.

    Accepts SHARP outputs (ply_path, intrinsics, extrinsics) and the
    original source image. Renders left and right eye views using the
    gsplat CUDA rasterizer, then deletes the PLY file to reclaim disk
    space (~60MB per frame).

    Outputs: (original_image, left_eye, right_eye)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Original source image (passed through for restoration reference)",
                }),
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to Gaussian Splatting PLY file (from SHARP Predict)",
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "3x3 camera intrinsics matrix from SHARP Predict",
                }),
            },
            "optional": {
                "extrinsics": ("EXTRINSICS", {
                    "tooltip": "4x4 camera extrinsics matrix. Default: identity (SHARP default)",
                }),
                "ipd_mm": ("FLOAT", {
                    "default": 63.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 0.1,
                    "tooltip": "Inter-pupillary distance in mm. Human average ~63mm.",
                }),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Output image width in pixels per eye",
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Output image height in pixels per eye",
                }),
                "background_color": (["white", "black", "transparent"], {
                    "default": "white",
                    "tooltip": "Background color for areas with no Gaussians",
                }),
                "delete_ply": (["enabled", "disabled"], {
                    "default": "enabled",
                    "tooltip": "Delete the PLY file after rendering to reclaim disk space (~60MB each)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("original", "left_eye", "right_eye")
    FUNCTION = "render_stereo"
    CATEGORY = "VR/Stereo"
    DESCRIPTION = (
        "Render stereo pair from Gaussian splat PLY. "
        "Outputs original image + left/right eye views. "
        "Deletes PLY after rendering to save disk space."
    )

    @torch.no_grad()
    def render_stereo(
        self,
        image: torch.Tensor,
        ply_path: str,
        intrinsics: list[list[float]],
        extrinsics: list[list[float]] | None = None,
        ipd_mm: float = 63.0,
        image_width: int = 1024,
        image_height: int = 1024,
        background_color: str = "white",
        delete_ply: str = "enabled",
    ):
        if not ply_path or not Path(ply_path).exists():
            raise ValueError(f"PLY file not found: {ply_path}")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "StereoGaussianRender requires CUDA. "
                "The gsplat rasterizer does not support CPU or MPS."
            )
        device = "cuda"

        ply_size_mb = Path(ply_path).stat().st_size / (1024 * 1024)
        print(f"[StereoGaussianRender] Loading PLY: {ply_path} ({ply_size_mb:.1f}MB)")
        gaussians, metadata = load_gaussians(ply_path, device)
        print(
            f"[StereoGaussianRender] Loaded scene, "
            f"focal_length={metadata.focal_length_px:.1f}px"
        )

        # Build camera matrices
        K4 = build_intrinsics_4x4(intrinsics, device)
        ipd_m = ipd_mm / 1000.0
        left_ext, right_ext = build_stereo_extrinsics(extrinsics, ipd_m, device)

        print(
            f"[StereoGaussianRender] Rendering stereo pair: "
            f"{image_width}x{image_height}, IPD={ipd_mm:.1f}mm, bg={background_color}"
        )

        # Render left eye
        left_color, _ = render_view(
            gaussians, left_ext, K4, image_width, image_height, background_color,
        )

        # Render right eye
        right_color, _ = render_view(
            gaussians, right_ext, K4, image_width, image_height, background_color,
        )

        # Free GPU memory from gaussians before converting
        del gaussians
        torch.cuda.empty_cache()

        # Convert from renderer format [1,3,H,W] to ComfyUI format [1,H,W,3]
        left_img = left_color[0].permute(1, 2, 0).unsqueeze(0).cpu().clamp(0, 1)
        right_img = right_color[0].permute(1, 2, 0).unsqueeze(0).cpu().clamp(0, 1)

        # Delete PLY to reclaim disk space
        if delete_ply == "enabled":
            try:
                os.remove(ply_path)
                print(f"[StereoGaussianRender] Deleted PLY ({ply_size_mb:.1f}MB): {ply_path}")
            except OSError as e:
                print(f"[StereoGaussianRender] Warning: could not delete PLY: {e}")

        print("[StereoGaussianRender] Done - outputs: original, left_eye, right_eye")
        return (image, left_img, right_img)


class SBSConcat:
    """Concatenate left and right eye images into stereo VR formats.

    Supports Side-by-Side (SBS) and Over-Under (OU/Top-Bottom) layouts
    commonly used by VR headsets and media players.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_eye": ("IMAGE", {
                    "tooltip": "Left eye image [B, H, W, C]",
                }),
                "right_eye": ("IMAGE", {
                    "tooltip": "Right eye image [B, H, W, C]",
                }),
            },
            "optional": {
                "layout": (["side_by_side", "over_under"], {
                    "default": "side_by_side",
                    "tooltip": "SBS = left|right horizontally, OU = left on top, right on bottom",
                }),
                "half_resolution": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Halve each eye's width (SBS) or height (OU) for half-SBS/OU format",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stereo_image",)
    FUNCTION = "concat"
    CATEGORY = "VR/Stereo"
    DESCRIPTION = (
        "Concatenate left/right eye images into stereo VR format. "
        "Supports Side-by-Side (SBS) and Over-Under (OU) layouts."
    )

    def concat(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        layout: str = "side_by_side",
        half_resolution: str = "disabled",
    ):
        if left_eye.shape != right_eye.shape:
            raise ValueError(
                f"Left and right eye images must have the same shape. "
                f"Got left={left_eye.shape}, right={right_eye.shape}"
            )

        left = left_eye
        right = right_eye

        if half_resolution == "enabled":
            b, h, w, c = left.shape
            left = left.permute(0, 3, 1, 2)
            right = right.permute(0, 3, 1, 2)
            if layout == "side_by_side":
                left = torch.nn.functional.interpolate(
                    left, size=(h, w // 2), mode="bilinear", align_corners=False
                )
                right = torch.nn.functional.interpolate(
                    right, size=(h, w // 2), mode="bilinear", align_corners=False
                )
            else:
                left = torch.nn.functional.interpolate(
                    left, size=(h // 2, w), mode="bilinear", align_corners=False
                )
                right = torch.nn.functional.interpolate(
                    right, size=(h // 2, w), mode="bilinear", align_corners=False
                )
            left = left.permute(0, 2, 3, 1)
            right = right.permute(0, 2, 3, 1)

        if layout == "side_by_side":
            result = concat_side_by_side(left, right)
        else:
            result = concat_top_bottom(left, right)

        b, h, w, c = result.shape
        print(f"[SBSConcat] Output: {w}x{h} ({layout}, half={'on' if half_resolution == 'enabled' else 'off'})")

        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "StereoGaussianRender": StereoGaussianRender,
    "SBSConcat": SBSConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoGaussianRender": "Stereo Gaussian Render (VR)",
    "SBSConcat": "Stereo SBS/OU Concat",
}
