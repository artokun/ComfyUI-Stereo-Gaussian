# ComfyUI Stereo Gaussian

Stereoscopic VR rendering from 3D Gaussian Splats for ComfyUI. Renders stereo left/right eye pairs using the [gsplat](https://docs.gsplat.studio/) CUDA rasterizer — designed to work with [ComfyUI-Sharp](https://github.com/PozzettiAndrea/ComfyUI-Sharp) (Apple's SHARP model).

## Nodes

### Stereo Gaussian Render (VR)

Renders a stereo pair from a Gaussian splat PLY file. Applies a horizontal IPD offset to the camera extrinsics for left/right eye views. Passes through the original source image for downstream restoration.

**Inputs:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | required | Original source image (passed through for restoration) |
| `ply_path` | STRING | required | Path to PLY file from SHARP Predict |
| `intrinsics` | INTRINSICS | required | 3x3 camera intrinsics from SHARP |
| `extrinsics` | EXTRINSICS | identity | 4x4 camera extrinsics |
| `ipd_mm` | FLOAT | 63.0 | Inter-pupillary distance in mm |
| `image_width` | INT | 1024 | Output width per eye |
| `image_height` | INT | 1024 | Output height per eye |
| `background_color` | ENUM | white | white / black / transparent |
| `delete_ply` | ENUM | enabled | Delete PLY after render (~60MB each) |

**Outputs:**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `original` | IMAGE | Pass-through of input image |
| 1 | `left_eye` | IMAGE | Left eye view |
| 2 | `right_eye` | IMAGE | Right eye view |

### Stereo SBS/OU Concat

Concatenates left and right eye images into standard VR stereo formats.

**Inputs:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `left_eye` | IMAGE | required | Left eye image |
| `right_eye` | IMAGE | required | Right eye image |
| `layout` | ENUM | side_by_side | side_by_side / over_under |
| `half_resolution` | ENUM | disabled | Halve per-eye resolution for half-SBS/OU |

**Outputs:**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `stereo_image` | IMAGE | Combined stereo image |

## Installation

### Prerequisites

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with **CUDA GPU**
- [ComfyUI-Sharp](https://github.com/PozzettiAndrea/ComfyUI-Sharp) installed (provides `sharp.utils.gaussians`)

### Install via ComfyUI Manager

Search for **ComfyUI-Stereo-Gaussian** in ComfyUI Manager.

### Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/artokun/ComfyUI-Stereo-Gaussian.git
cd ComfyUI-Stereo-Gaussian
pip install -r requirements.txt
```

Restart ComfyUI. Nodes appear under the **VR/Stereo** category.

### gsplat Note

The `gsplat` library compiles CUDA kernels on first import. This requires the CUDA toolkit. If SHARP is working, you already have everything needed.

## Example Workflow

```
LoadImage (original frame)
    |
LoadSharpModel → SharpPredict
    |                |
    v                v
    image    ply_path + intrinsics
    |                |
    +-------+--------+
            |
            v
    StereoGaussianRender (ipd=63mm, deletes PLY)
        |          |          |
        v          v          v
    original   left_eye   right_eye
        |          |          |
        |    +-----+    +----+
        |    |          |
        v    v          v
    QwenRestore(L)  QwenRestore(R)   ← Gaussian-Splash LoRA, 4 steps
        |               |
        v               v
        SBSConcat (side_by_side)
            |
            v
        SaveImage → VR-ready SBS frame
```

## Technical Details

### Coordinate System

Uses OpenCV convention (Y-down, Z-forward, right-handed):
- Camera at origin for SHARP output (identity extrinsics)
- Scene at positive Z values
- X-axis points right

### Stereo Offset

World-to-camera extrinsics with translation:
- Left eye: `tx = +IPD/2` (camera shifted left in world)
- Right eye: `tx = -IPD/2` (camera shifted right in world)

### Rendering

Uses `gsplat.rendering.rasterization` for GPU-accelerated Gaussian splatting:
- Classic rasterization mode
- RGB+D render mode
- Configurable background compositing
- GPU memory freed after rendering (`del gaussians; torch.cuda.empty_cache()`)

### PLY Cleanup

SHARP PLY files are ~60MB each. With `delete_ply: enabled` (default), the PLY is deleted immediately after both eyes are rendered. This is critical for video processing where hundreds of frames would otherwise consume tens of GB.

## License

MIT
