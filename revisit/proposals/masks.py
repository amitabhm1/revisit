"""
Mask Model Module
=================

Handles SAM-based segmentation mask generation and management.
"""

import os
import pickle
import numpy as np
import cv2
import torch
from PIL import Image

from .utils import reorder_mask, remove_small_masks, masks_update

class _SAM3TransformShim:
    def __init__(self, sam3_pred):
        self.sam3_pred = sam3_pred

    def apply_coords(self, points: np.ndarray, im_size):
        # SamAutomaticMaskGenerator hands us ABSOLUTE pixel coords of the (cropped) image passed to
        # set_image(). SAM 3's transform_coords(normalize=False) assumes [0,1]-normalized input and
        # only multiplies by the model resolution (1008), so absolute coords must use normalize=True.
        pts = torch.as_tensor(points, dtype=torch.float, device=self.sam3_pred.device)
        trans = self.sam3_pred._transforms.transform_coords(
            pts, normalize=True, orig_hw=im_size
        )
        return trans.detach().cpu().numpy()


class _SAM3ModelShim:
    def __init__(self, mask_threshold=0.0):
        self.mask_threshold = mask_threshold


class _SAM3PredictorAdapter:
    """Adapts SAM3InteractiveImagePredictor to the SamPredictor interface used by SamAutomaticMaskGenerator."""

    def __init__(self, sam3_pred):
        self.sam3_pred = sam3_pred
        self.transform = _SAM3TransformShim(sam3_pred)
        self.model = _SAM3ModelShim(sam3_pred.mask_threshold)

    @property
    def device(self):
        return self.sam3_pred.device

    def set_image(self, image: np.ndarray):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.sam3_pred.set_image(image)

    def reset_image(self):
        self.sam3_pred.reset_predictor()

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        multimask_output: bool = True,
        return_logits: bool = True,
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, iou_preds, low_res_masks = self.sam3_pred._predict(
                point_coords.to(dtype=torch.float, device=self.device),
                point_labels.to(dtype=torch.int, device=self.device),
                multimask_output=multimask_output,
                return_logits=return_logits,
            )
        return masks.float(), iou_preds.float(), low_res_masks.float()


# SAM 3.1 (sam3.1_multiplex.pt) stores the point-prompt head under `tracker.model.interactive_*`, next to a
# video-only multiplex propagation branch (`tracker.model.sam_mask_decoder` with 16 multiplexed tokens,
# memory `transformer`, `maskmem_backbone`, ...) that must NOT be loaded into the single-image predictor.
_SAM31_TRACKER_TO_IMAGE = (
    ("interactive_sam_prompt_encoder.", "sam_prompt_encoder."),
    ("interactive_sam_mask_decoder.", "sam_mask_decoder."),
    ("interactivity_no_mem_embed", "no_mem_embed"),
    ("interactive_obj_ptr_proj.", "obj_ptr_proj."),
    ("interactive_mask_downsample.", "mask_downsample."),
)

# Everything the point-grid automatic mask generator touches (SAM3VLBackbone uses scalp=1, so only neck levels 0-2).
_POINT_PROMPT_PREFIXES = (
    "backbone.vision_backbone.trunk.",
    "backbone.vision_backbone.sam2_convs.0.",
    "backbone.vision_backbone.sam2_convs.1.",
    "backbone.vision_backbone.sam2_convs.2.",
    "inst_interactive_predictor.model.sam_prompt_encoder.",
    "inst_interactive_predictor.model.sam_mask_decoder.",
    "inst_interactive_predictor.model.no_mem_embed",
)


def _remap_sam31_ckpt_for_image_model(ckpt: dict) -> dict:
    out = {}
    for k, v in ckpt.items():
        if k.startswith("detector."):
            nk = k[len("detector."):]
            if ".propagation_convs." in nk:  # video-propagation neck, unused for single images
                continue
            out[nk.replace("backbone.vision_backbone.interactive_convs.", "backbone.vision_backbone.sam2_convs.")] = v
        elif k.startswith("tracker.model."):
            sub = k[len("tracker.model."):]
            for src, dst in _SAM31_TRACKER_TO_IMAGE:
                if sub.startswith(src):
                    out["inst_interactive_predictor.model." + dst + sub[len(src):]] = v
                    break
    return out


def _load_verified_point_prompt_weights(model, state: dict, label: str) -> None:
    """Load `state` and raise if any weight on the point-prompt path would stay randomly initialized."""
    model_sd = model.state_dict()
    required = [n for n, _ in model.named_parameters() if n.startswith(_POINT_PROMPT_PREFIXES)]
    # Random-at-init buffer on the prompt path (Fourier PE of the point encoder) must come from the checkpoint too.
    required += [n for n in model_sd if n.startswith(_POINT_PROMPT_PREFIXES) and "positional_encoding_gaussian_matrix" in n]
    bad = [n for n in required if n not in state or tuple(state[n].shape) != tuple(model_sd[n].shape)]
    if bad:
        raise RuntimeError(
            f"[MaskModel] {label}: {len(bad)}/{len(required)} point-prompt tensors missing or shape-mismatched "
            f"(they would stay randomly initialized), e.g. {bad[:6]}"
        )
    loadable = {k: v for k, v in state.items() if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    model.load_state_dict(loadable, strict=False)
    print(
        f"[MaskModel] {label}: point-prompt path fully populated ({len(required)}/{len(required)} tensors); "
        f"{len(loadable)}/{len(model_sd)} model tensors loaded from checkpoint"
    )


class MaskModel:
    """
    Handles segmentation mask generation using SAM (SAM 1 ViT-H, SAM 3, or SAM 3.1).
    """
    
    def __init__(self, config, device='cuda'):
        """
        Initialize mask model.
        
        Args:
            config: Configuration dictionary
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.mask_generator = None
        sam_config = self.config['models']['sam']
        self.backend = sam_config.get('backend', 'sam1').lower()

    def get_cache_filename(self):
        if self.backend in ('sam1', 'sam', 'vit_h'):
            return "sam_masks.pkl"
        return f"{self.backend}_masks.pkl"
    
    def _build_sam2_image_predictor(self, sam_config, checkpoint):
        """Official SAM 2 / 2.1 image predictor (facebookresearch/sam2, installed as a package)."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        cfg = sam_config.get('sam2_config', 'configs/sam2.1/sam2.1_hiera_l.yaml')
        # build_sam2 loads strictly (raises on any missing/unexpected key). apply_postprocessing=False matches the
        # official SAM2AutomaticMaskGenerator and avoids the optional CUDA connected-components extension.
        model = build_sam2(cfg, checkpoint, device=str(self.device), mode="eval", apply_postprocessing=False)
        print(f"[MaskModel] {self.backend} ({os.path.basename(checkpoint)}): strict checkpoint load OK ({cfg})")
        return SAM2ImagePredictor(model)

    def _initialize_sam(self):
        """Initialize SAM model and mask generator."""
        from .segment_anything_ls import build_sam, SamAutomaticMaskGenerator
        
        sam_config = self.config['models']['sam']
        checkpoint = os.path.expandvars(sam_config['checkpoint'])
        
        if self.backend in ('sam2', 'sam2.1', 'sam3', 'sam3.1'):
            if not (checkpoint and os.path.exists(checkpoint)):
                raise FileNotFoundError(f"[MaskModel] {self.backend} checkpoint not found: {checkpoint}")
            if self.backend in ('sam2', 'sam2.1'):
                interactive_pred = self._build_sam2_image_predictor(sam_config, checkpoint)
            else:
                from sam3.model_builder import build_sam3_image_model
                # Build the architecture without weights, then load + verify the checkpoint explicitly
                # (a silent strict=False load previously left the SAM 3.1 point-prompt head randomly initialized).
                sam3_model = build_sam3_image_model(
                    checkpoint_path=None,
                    device="cpu",
                    eval_mode=True,
                    load_from_HF=False,
                    enable_segmentation=True,
                    enable_inst_interactivity=True,
                    compile=False,
                )
                raw_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if "model" in raw_ckpt and isinstance(raw_ckpt["model"], dict):
                    raw_ckpt = raw_ckpt["model"]
                if self.backend == 'sam3.1':
                    state = _remap_sam31_ckpt_for_image_model(raw_ckpt)
                else:  # SAM 3 (sam3.pt): same mapping as upstream sam3.model_builder._load_checkpoint
                    state = {k.replace("detector.", ""): v for k, v in raw_ckpt.items() if "detector" in k}
                    state.update({
                        k.replace("tracker.", "inst_interactive_predictor.model."): v
                        for k, v in raw_ckpt.items() if "tracker" in k
                    })
                del raw_ckpt
                _load_verified_point_prompt_weights(
                    sam3_model, state, label=f"{self.backend} ({os.path.basename(checkpoint)})"
                )
                del state
                sam3_model = sam3_model.to(self.device).eval()
                interactive_pred = sam3_model.inst_interactive_predictor
                if getattr(interactive_pred.model, "backbone", None) is None:
                    interactive_pred.model.backbone = sam3_model.backbone
                if getattr(interactive_pred.model, "bf16_context", None) is not None:
                    try:
                        interactive_pred.model.bf16_context.__exit__(None, None, None)
                    except Exception:
                        pass
            # Create SamAutomaticMaskGenerator without building SAM1, then swap in the SAM 2/3 interactive predictor
            dummy_gen = SamAutomaticMaskGenerator.__new__(SamAutomaticMaskGenerator)
            from .segment_anything_ls.utils.amg import build_all_layer_point_grids
            crop_n_layers = sam_config['crop_n_layers']
            crop_n_points_downscale_factor = sam_config['crop_n_points_downscale_factor']
            points_per_side = sam_config['points_per_side']
            dummy_gen.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
            dummy_gen.predictor = _SAM3PredictorAdapter(interactive_pred)
            dummy_gen.points_per_batch = 128
            dummy_gen.pred_iou_thresh = min(float(sam_config['pred_iou_thresh']), 0.78)
            dummy_gen.stability_score_thresh = min(float(sam_config['stability_score_thresh']), 0.88)
            dummy_gen.stability_score_offset = 1.0
            dummy_gen.box_nms_thresh = sam_config['box_nms_thresh']
            dummy_gen.crop_n_layers = crop_n_layers
            dummy_gen.crop_nms_thresh = 0.7
            dummy_gen.crop_overlap_ratio = 512 / 1500
            dummy_gen.crop_n_points_downscale_factor = crop_n_points_downscale_factor
            dummy_gen.min_mask_region_area = 0
            dummy_gen.output_mode = "binary_mask"
            self.mask_generator = dummy_gen
            print(f"[MaskModel] Initialized {self.backend} with checkpoint: {checkpoint}")
        else:
            self.mask_generator = SamAutomaticMaskGenerator(
                model=build_sam(checkpoint=checkpoint).to(device=self.device),
                points_per_side=sam_config['points_per_side'],
                points_per_batch=128,
                pred_iou_thresh=sam_config['pred_iou_thresh'],
                box_nms_thresh=sam_config['box_nms_thresh'],
                stability_score_thresh=sam_config['stability_score_thresh'],
                crop_n_layers=sam_config['crop_n_layers'],
                crop_n_points_downscale_factor=sam_config['crop_n_points_downscale_factor'],
                min_mask_region_area=0,
            )
            print(f"[MaskModel] Initialized SAM1 with checkpoint: {checkpoint}")

    def predict_mask(self, image, mask_generator, min_size=100):
        """
        Generate non-overlapping segmentation masks using SAM. 
        Follow the code of LangSplats.
        
        Args:
            image: Input image (H, W, 3) as uint8 numpy array
            mask_generator: SAM mask generator
            min_size: Minimum region size
            
        Returns:
            Segmentation mask (H, W) with integer region IDs
        """
        masks = mask_generator.generate(image)
    
        
        masks_m, masks_l = masks
        masks_m, masks_l = masks_update(masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
        
        # Assign region IDs
        group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
        group_counter = 0
        
        for mask_level in (masks_l, masks_m):
            if (group_ids == -1).sum() < min_size:
                break
            
            mask_level = sorted(mask_level, key=lambda x: x['area'], reverse=True)
            
            for mask_dict in mask_level:
                mask_single = mask_dict["segmentation"]
                non_mask_area = (group_ids == -1)
                to_mask_area = mask_single & non_mask_area
                
                if to_mask_area.sum() < min_size:
                    continue
                
                group_ids[to_mask_area] = group_counter
                group_counter += 1
                
                if (group_ids == -1).sum() < min_size:
                    break
        
        return group_ids

    def load_or_generate_masks(self, file_list, H, W, cache_path=None):
        """
        Load cached SAM masks or generate new ones.
        
        Args:
            file_list: List of image paths
            H, W: Target dimensions
            cache_path: Path to cached masks file
            
        Returns:
            List of non-overlapping masks with each value being the index of the mask
            and -1 for no mask
        """
        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                masks_list = pickle.load(f)
            
            # Verify and resize if needed
            for i, mask in enumerate(masks_list):
                if mask.shape != (H, W):
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.int64)
                    masks_list[i] = reorder_mask(mask)
            
            print(f"[MaskModel] Loaded {len(masks_list)} masks from cache")
            return masks_list
        
        # Generate new masks
        print(f"[MaskModel] Generating masks for {len(file_list)} images...")
        
        if self.mask_generator is None:
            self._initialize_sam()
        
        masks_list = []
        for idx, filepath in enumerate(file_list):
            pil_img = Image.open(filepath).convert('RGB')
            if max(pil_img.size) > max(W * 2, H * 2):
                pil_img = pil_img.resize((W * 2, H * 2), Image.Resampling.BILINEAR)
            image = np.array(pil_img, dtype=np.uint8)
            mask = self.predict_mask(image, self.mask_generator)
            if mask.shape != (H, W):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int64)
            else:
                mask = mask.astype(np.int64)
            
            # Remove small masks
            mask = remove_small_masks(mask)
            masks_list.append(reorder_mask(mask))
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(file_list)} images")
        
        # Cache the masks
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(masks_list, f)
            print(f"[MaskModel] Saved masks to cache: {cache_path}")
        
        return masks_list
    
    def aggregate_to_regions(self, pixel_values, mask):
        """
        Aggregate pixel-level values to region-level using mask.
        
        Args:
            pixel_values: Per-pixel values (H, W)
            mask: Segmentation mask (H, W) with region IDs
            
        Returns:
            Region-aggregated values (H, W)
        """
        region_values = torch.zeros_like(mask).float()
        for region_id in mask.unique():
            if region_id == -1:
                continue
            region_mask = mask == region_id
            if region_mask.sum() > 0 and pixel_values[region_mask].numel() > 0:
                region_values[region_mask] = pixel_values[region_mask].mean()
        return region_values
    
    def get_mask_statistics(self, masks_list):
        """
        Get statistics about the masks.
        
        Args:
            masks_list: List of segmentation masks
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_images': len(masks_list),
            'regions_per_image': [],
            'avg_region_size': []
        }
        
        for mask in masks_list:
            unique_regions = np.unique(mask)
            unique_regions = unique_regions[unique_regions != -1]
            stats['regions_per_image'].append(len(unique_regions))
            
            if len(unique_regions) > 0:
                sizes = [(mask == region_id).sum() for region_id in unique_regions]
                stats['avg_region_size'].append(np.mean(sizes))
        
        stats['avg_regions_per_image'] = np.mean(stats['regions_per_image'])
        stats['avg_region_size_overall'] = np.mean(stats['avg_region_size'])
        
        return stats
    
    def cleanup(self):
        """Clean up model resources."""
        if self.mask_generator is not None:
            del self.mask_generator
            self.mask_generator = None
            torch.cuda.empty_cache()
