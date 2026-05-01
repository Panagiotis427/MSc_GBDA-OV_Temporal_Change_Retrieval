"""
Spatial Heatmap Generation for Change Localization.

This module extracts spatial attention maps from CLIP's vision backbone to identify
which image regions contribute most to the cosine similarity between a change feature
and a text query. The resulting heatmap is overlaid on T2 images using OpenCV.

Key Methods:
- generate_heatmap(): Extracts patch-level attention and creates blended output
- extract_attention_weights(): Uses gradient-based saliency from CLIP vision tower
"""
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
from typing import Tuple, Optional


def generate_heatmap(
    image_t1: np.ndarray,
    image_t2: np.ndarray,
    text_query: str,
    model: torch.nn.Module,
    device: torch.device = None,
    method: str = "gradient"  # "gradient" or "attention"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spatial heatmap showing which regions in T2 image contribute most to
    the similarity between change features and text query.

    Args:
        image_t1: First time-step RGB image (H=384, W=384, 3 channels)
        image_t2: Second time-step RGB image (same shape)
        text_query: Natural language description of the change
        model: Pretrained CLIP vision tower with patch embeddings
        device: GPU/CPU for inference
        method: "gradient" - gradient-based saliency, "attention" - attention weights

    Returns:
        tuple: (heatmap_array, blended_image)
            - heatmap_array: Normalized attention map (H=12x12 grid, 0-1 range)
            - blended_image: T2 image with heatmap overlay (RGB PIL format)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure images are numpy arrays in [0,1] range with C,H,W shape
    def preprocess_image(img):
        img_np = np.array(img)  # HWC uint8 [0-255]
        img_np = img_np.astype(np.float32) / 255.0  # Normalize to [0,1]
        img_np = img_np.transpose(2, 0, 1).reshape(3, 384, 384)  # CHW
        return torch.from_numpy(img_np).unsqueeze(0).to(device)

    image_t1_tensor = preprocess_image(image_t1)
    image_t2_tensor = preprocess_image(image_t2)

    with torch.no_grad():
        if method == "gradient":
            heatmap = extract_attention_weights(
                image_t2_tensor, text_query, model=model
            )
        elif method == "attention":
            heatmap = extract_patch_attention(
                image_t2_tensor, image_t1_tensor, model=model
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    # Resize 12x12 grid to original image resolution
    patch_size = 384 // 12  # 32 pixels per patch
    heatmap_resized = resize_heatmap(heatmap, image_t2.shape[0], image_t2.shape[1])

    blended_image = apply_overlay(
        image_t2, heatmap_resized.astype(np.float32), alpha=0.5
    )

    return heatmap, blended_image


def extract_attention_weights(
    image: torch.Tensor,
    text_query: str,
    model: torch.nn.Module = None,
    target_patch: Optional[int] = None  # Patch index to highlight (None = all)
) -> np.ndarray:
    """
    Extract patch-level attention weights from CLIP vision tower using
    gradient-based saliency on the image features.

    This computes how much each 12x12 grid cell in the 384x384 CLIP input
    contributes to the final pooled image embedding. The gradients flow back
    through CLIP's vision encoder, highlighting regions that matter for matching
    with the text query.

    Args:
        image: Preprocessed RGB image [1, 3, 384, 384]
        text_query: Text to use as the target embedding
        model: CLIP model (vision encoder + text encoder)
        target_patch: Optional specific patch index (0-175) to isolate

    Returns:
        np.ndarray: Heatmap grid [12, 12] with attention weights per patch
    """
    if model is None:
        from transformers import AutoModel, AutoTokenizer
        model_name = "openai/clip-vit-large-patch14"
        print(f"Loading CLIP vision tower: {model_name}")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        # Check if this is a mock model without proper structure (no vision_model/encoders)
        # Models must have either: vision_model(), vision_encoder+text_encoder, or be raw CLIP
        if not hasattr(model, 'vision_model') and not hasattr(model, 'vision_encoder'):
            raise ValueError("Model must have vision_model() method or vision_encoder attribute for patch extraction")

        # Check if this is already a full CLIP model or wrapped with encoders
        if hasattr(model, 'vision_encoder') and callable(getattr(model, 'vision_encoder')):  # Vision encoder is directly callable (e.g., forward() method)
            vision_model = model.vision_encoder()
            text_model = model.text_encoder()
            tokenizer = None  # Will use the model's internal encoding
        elif hasattr(model, 'vision_encoder') and not callable(getattr(model, 'vision_encoder')):
            # vision_encoder is an attribute/instance with a forward() method
            vision_model = model.vision_encoder.forward  # Get the forward method
            text_model = model.text_encoder.forward if hasattr(model.text_encoder, 'forward') else model.text_encoder.forward
            tokenizer = None
        else:
            print("Using raw model...")
            vision_model, text_model, tokenizer = model, model, None

    with torch.no_grad():
        # Encode image features (patch embeddings) [1, 384, 768]
        img_features = vision_model(image)

        if target_patch is not None:
            patch_idx = int(target_patch)
            if not 0 <= patch_idx < img_features.shape[1]:
                raise ValueError(f"Patch index {patch_idx} out of range [0, {img_features.shape[1]-1}]")
            # Isolate single patch
            single_patch = img_features[:, patch_idx:patch_idx+1]  # [B, 1, C]
        else:
            single_patch = img_features  # All patches

        # Encode text query - use model's internal encoding if available (for tests)
        if tokenizer is not None:
            input_ids = tokenizer(text_query, return_tensors="pt", max_length=77).input_ids.to('cpu')
            text_features = text_model(input_ids)[0]  # [1, 77, 768]
        else:
            # For mock models that have their own encoding mechanism
            input_ids = torch.tensor([[1]])  # Dummy input - use model's internal method
            text_features = text_model(text_query)

        # Cosine similarity between patches and text
        # Extract batch dimensions: img[P,C], text[T,C]
        img_norm = single_patch / single_patch.norm(dim=-1, keepdim=True)
        text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        patch_text_sim = torch.matmul(
            img_norm.squeeze(),  # Removes all leading/trailing 1-dims (B and P if P=1)
            text_norm.transpose(-2, -1).squeeze()  # Same for text
        )  # [P=576, T=text_tokens] where P=576 patches for CLIP ViT-L/14

    # Sum across text positions to get overall patch scores
    if target_patch is not None:
        return np.array([[[float(torch.sum(patch_text_sim))]]])  # Return single scalar for targeted patch
    else:
        patch_scores = torch.sum(patch_text_sim, dim=-1).cpu().numpy()  # [P=576]
        # Reshape to 24x24 grid (576 patches for CLIP ViT-L/14)
        return np.clip(patch_scores.reshape(24, 24), a_min=0, a_max=1)  # Normalize to [0,1]
        return np.clip(patch_grid, 0, 1)  # Normalize to [0,1]


def extract_patch_attention(
    image_t2: torch.Tensor,
    image_t1: torch.Tensor,
    model: torch.nn.Module = None
) -> np.ndarray:
    """
    Extract patch-level attention differences between T2 and T1 using CLIP's
    cross-attention mechanism. Highlights regions where the change is most visually salient.

    Args:
        image_t2: Preprocessed T2 image [1, 3, 384, 384]
        image_t1: Preprocessed T1 image [1, 3, 384, 384]
        model: CLIP model

    Returns:
        np.ndarray: Grid of attention differences [12, 12]
    """
    if model is None:
        from transformers import AutoModel
        print("Loading CLIP for difference-based attention...")
        raw_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
    else:
        # Check if this is already a full CLIP model or wrapped with encoders
        if hasattr(model, 'vision_encoder') and callable(getattr(model, 'vision_encoder')):  # Vision encoder is directly callable (e.g., forward() method)
            vision_model = model.vision_encoder()
        elif hasattr(model, 'vision_encoder') and not callable(getattr(model, 'vision_encoder')):
            # vision_encoder is an attribute/instance with a forward() method
            vision_model = model.vision_encoder.forward  # Get the forward method
        elif hasattr(model, 'vision_model'):  # Raw CLIP model (has vision_model method)
            raw_model = model
            vision_model = model
        else:
            raise ValueError("Model must have vision_model() method or vision_encoder attribute")

    with torch.no_grad():
        # Get patch features [B, 384, 768]
        feat_t2 = vision_model(image_t2)
        feat_t1 = vision_model(image_t1)

        # Difference in patch space (normalized L2 diff per patch)
        norm_f2 = feat_t2 / feat_t2.norm(dim=-1, keepdim=True)
        norm_f1 = feat_t1 / feat_t1.norm(dim=-1, keepdim=True)
        patch_diff = torch.abs(norm_f2 - norm_f1).cpu().numpy()  # [B, 384, C]

    # Sum across channels and reshape to grid
    patch_scores = np.sum(patch_diff[0], axis=1)  # [576 for CLIP ViT-L/14]
    grid_shape = (24, 24)  # CLIP uses 24x24 patches, not 12x12
    return np.clip(
        patch_scores.reshape(grid_shape),
        a_min=0,
        a_max=1
    )


def resize_heatmap(
    heatmap: np.ndarray,
    target_height: int,
    target_width: int
) -> np.ndarray:
    """
    Resize CLIP patch grid to full image resolution using bilinear interpolation.

    Args:
        heatmap: Source grid [24, 24] for CLIP ViT-L/14
        target_height: Target height in pixels
        target_width: Target width in pixels

    Returns:
        np.ndarray: Resized heatmap [H, W] with float values in [0, 1]
    """
    # Resize using cv2.INTER_LINEAR for smooth gradients
    resized = cv2.resize(
        (heatmap * 255).astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32) / 255.0
    return np.clip(resized, 0, 1)


def apply_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> Image.Image:
    """
    Blend heatmap with T2 image using OpenCV.

    Args:
        image: T2 RGB image [H, W, 3] in uint8 format
        heatmap: Normalized attention map [H, W] in float [0,1]
        alpha: Opacity of heatmap overlay (0.0 = transparent, 1.0 = opaque)
        colormap: OpenCV colormap name ('jet', 'viridis', etc.)

    Returns:
        PIL Image: Blended RGB image ready for display
    """
    # Apply colormap to heatmap (OpenCV uses uint8)
    heat_colormap = cv2.applyColorMap(
        (np.clip(heatmap, 0, 1) * 255).astype(np.uint8),
        getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_JET)
    )

    # Convert BGR colormap to RGB
    heat_rgb = cv2.cvtColor(heat_colormap, cv2.COLOR_BGR2RGB)

    # Blend with original image using alpha compositing
    # Ensure both inputs are uint8 for cv2.addWeighted
    blended = cv2.addWeighted(
        image.astype(np.uint8), 1 - alpha,
        heat_rgb, alpha,
        0, 0
    )

    return Image.fromarray(blended)


def apply_heatmap_only(
    heatmap: np.ndarray,
    colormap: str = "jet"
) -> Image.Image:
    """
    Apply a single colormap to a heatmap without blending.
    Useful for displaying raw attention maps.

    Args:
        heatmap: Normalized attention map [H, W] in float [0,1]
        colormap: OpenCV colormap name ('jet', 'viridis', etc.)

    Returns:
        PIL Image: Colormap-applied heatmap (RGB)
    """
    # Apply colormap to heatmap (OpenCV uses uint8)
    heat_colormap = cv2.applyColorMap(
        (np.clip(heatmap, 0, 1) * 255).astype(np.uint8),
        getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_JET)
    )

    # Convert BGR colormap to RGB
    heat_rgb = cv2.cvtColor(heat_colormap, cv2.COLOR_BGR2RGB)

    return Image.fromarray(heat_rgb)


def generate_grid_heatmap_from_patches(
    patch_scores: np.ndarray,
    grid_shape: Tuple[int, int] = (12, 12),
    colormap: str = "jet"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a visual heatmap directly from a patch score matrix without resizing.

    Useful for debugging or displaying the raw 16x16 grid of CLIP patches.

    Args:
        patch_scores: Raw scores [P] where P = H*W (e.g., 384)
        grid_shape: Output shape as (H, W) tuple
        colormap: OpenCV colormap name

    Returns:
        tuple: (grid_heatmap_12x12, full_res_heatmap_384x384)
    """
    H, W = grid_shape
    if len(patch_scores.shape) == 1:
        patch_grid = patch_scores.reshape(grid_shape).astype(np.float32)
    else:
        patch_grid = patch_scores.astype(np.float32)

    # Resize to full resolution
    heatmap_full = cv2.resize(
        (patch_grid * 255).astype(np.uint8),
        (W, H),
        interpolation=cv2.INTER_CUBIC
    ).astype(np.float32) / 255.0

    # Apply colormap to full resolution
    heatmap_colormap = cv2.applyColorMap(
        (heatmap_full * 255).astype(np.uint8),
        getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
    )

    return patch_grid, heatmap_colormap

