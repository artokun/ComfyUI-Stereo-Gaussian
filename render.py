"""Standalone stereo Gaussian splat rendering utility.

Can be used outside ComfyUI for testing and batch processing.
Uses the gsplat library for GPU-accelerated rasterization.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch


class StereoOutput(NamedTuple):
    """Output of stereo rendering."""

    left_color: torch.Tensor   # [1, 3, H, W] float32 [0,1]
    right_color: torch.Tensor  # [1, 3, H, W] float32 [0,1]
    left_alpha: torch.Tensor   # [1, 1, H, W] float32 [0,1]
    right_alpha: torch.Tensor  # [1, 1, H, W] float32 [0,1]


def load_gaussians(ply_path: str | Path, device: torch.device | str = "cuda"):
    """Load Gaussian splat from PLY file.

    Args:
        ply_path: Path to the PLY file.
        device: Device to load tensors onto.

    Returns:
        Tuple of (Gaussians3D, SceneMetaData) from the SHARP utils.
    """
    from sharp.utils.gaussians import load_ply

    gaussians, metadata = load_ply(Path(ply_path))
    gaussians = gaussians.to(torch.device(device))
    return gaussians, metadata


def build_intrinsics_4x4(
    intrinsics_3x3: list[list[float]] | torch.Tensor,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Convert 3x3 intrinsics to 4x4 format expected by gsplat renderer.

    Args:
        intrinsics_3x3: 3x3 intrinsics matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
        device: Target device.

    Returns:
        4x4 intrinsics tensor with batch dim [1, 4, 4].
    """
    K = torch.tensor(intrinsics_3x3, dtype=torch.float32) if not isinstance(
        intrinsics_3x3, torch.Tensor
    ) else intrinsics_3x3.float()

    K4 = torch.eye(4, dtype=torch.float32)
    K4[:3, :3] = K[:3, :3]
    return K4.unsqueeze(0).to(device)


def build_stereo_extrinsics(
    base_extrinsics: list[list[float]] | torch.Tensor | None,
    ipd_meters: float,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build left and right eye extrinsics from a base extrinsics matrix.

    OpenCV world-to-camera convention: t = -R * camera_position_world.
    For identity R, moving camera RIGHT by +d means tx = -d in the matrix.

    Args:
        base_extrinsics: 4x4 base extrinsics (world-to-camera). None = identity.
        ipd_meters: Inter-pupillary distance in meters.
        device: Target device.

    Returns:
        Tuple of (left_extrinsics, right_extrinsics), each [1, 4, 4].
    """
    if base_extrinsics is not None:
        base = torch.tensor(base_extrinsics, dtype=torch.float32) if not isinstance(
            base_extrinsics, torch.Tensor
        ) else base_extrinsics.float()
    else:
        base = torch.eye(4, dtype=torch.float32)

    half_ipd = ipd_meters / 2.0

    # Left eye: camera shifted LEFT in world -> positive tx in world-to-camera
    left = base.clone()
    left[0, 3] += half_ipd

    # Right eye: camera shifted RIGHT in world -> negative tx in world-to-camera
    right = base.clone()
    right[0, 3] -= half_ipd

    return left.unsqueeze(0).to(device), right.unsqueeze(0).to(device)


def render_view(
    gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_width: int,
    image_height: int,
    background_color: str = "white",
):
    """Render a single view of the Gaussian splat scene.

    Args:
        gaussians: Gaussians3D object (from sharp.utils.gaussians).
        extrinsics: [1, 4, 4] world-to-camera matrix.
        intrinsics: [1, 4, 4] intrinsics matrix.
        image_width: Output width in pixels.
        image_height: Output height in pixels.
        background_color: "white", "black", or "random_color".

    Returns:
        Tuple of (color [1,3,H,W], alpha [1,1,H,W]).
    """
    import gsplat

    colors, alphas, _ = gsplat.rendering.rasterization(
        means=gaussians.mean_vectors[0],
        quats=gaussians.quaternions[0],
        scales=gaussians.singular_values[0],
        opacities=gaussians.opacities[0],
        colors=gaussians.colors[0],
        viewmats=extrinsics,
        Ks=intrinsics[:, :3, :3],
        width=image_width,
        height=image_height,
        render_mode="RGB+D",
        rasterize_mode="classic",
        absgrad=False,
        packed=False,
    )

    rendered_color = colors[..., 0:3].permute(0, 3, 1, 2)  # [1, 3, H, W]
    rendered_alpha = alphas.permute(0, 3, 1, 2)             # [1, 1, H, W]

    # Compose with background
    if background_color == "white":
        rendered_color = rendered_color + (1.0 - rendered_alpha)
    elif background_color == "black":
        pass  # already premultiplied
    # "transparent" leaves alpha separate for downstream compositing

    return rendered_color, rendered_alpha


def render_stereo(
    ply_path: str | Path,
    intrinsics_3x3: list[list[float]] | torch.Tensor,
    extrinsics_4x4: list[list[float]] | torch.Tensor | None = None,
    ipd_mm: float = 63.0,
    image_width: int = 1024,
    image_height: int = 1024,
    background_color: str = "white",
    device: str = "cuda",
) -> StereoOutput:
    """Render a stereo pair from a Gaussian splat PLY file.

    Args:
        ply_path: Path to the PLY file generated by SHARP.
        intrinsics_3x3: 3x3 intrinsics matrix from SharpPredict.
        extrinsics_4x4: 4x4 base extrinsics. None = identity (default for SHARP).
        ipd_mm: Inter-pupillary distance in millimeters (default 63mm).
        image_width: Output image width.
        image_height: Output image height.
        background_color: "white", "black", or "transparent".
        device: Torch device string.

    Returns:
        StereoOutput with left/right color and alpha tensors.
    """
    gaussians, _ = load_gaussians(ply_path, device)
    K4 = build_intrinsics_4x4(intrinsics_3x3, device)
    left_ext, right_ext = build_stereo_extrinsics(
        extrinsics_4x4, ipd_mm / 1000.0, device
    )

    left_color, left_alpha = render_view(
        gaussians, left_ext, K4, image_width, image_height, background_color
    )
    right_color, right_alpha = render_view(
        gaussians, right_ext, K4, image_width, image_height, background_color
    )

    return StereoOutput(
        left_color=left_color,
        right_color=right_color,
        left_alpha=left_alpha,
        right_alpha=right_alpha,
    )


def stereo_to_comfyui(output: StereoOutput) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert StereoOutput to ComfyUI IMAGE format.

    Args:
        output: StereoOutput from render_stereo.

    Returns:
        Tuple of (left_image, right_image) in ComfyUI format [1, H, W, 3].
    """
    left = output.left_color[0].permute(1, 2, 0).unsqueeze(0).cpu().clamp(0, 1)
    right = output.right_color[0].permute(1, 2, 0).unsqueeze(0).cpu().clamp(0, 1)
    return left, right


def concat_side_by_side(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Concatenate left and right images side-by-side for VR SBS format.

    Args:
        left: ComfyUI IMAGE tensor [B, H, W, C].
        right: ComfyUI IMAGE tensor [B, H, W, C].

    Returns:
        Side-by-side IMAGE tensor [B, H, W*2, C].
    """
    return torch.cat([left, right], dim=2)


def concat_top_bottom(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Concatenate left and right images top-bottom for VR OU format.

    Args:
        left: ComfyUI IMAGE tensor [B, H, W, C] (left = top).
        right: ComfyUI IMAGE tensor [B, H, W, C] (right = bottom).

    Returns:
        Over-under IMAGE tensor [B, H*2, W, C].
    """
    return torch.cat([left, right], dim=1)
