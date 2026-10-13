"""
Semantic Model Module
=====================

Handles semantic feature extraction using DINOv3 and appearance-based cost computation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from .utils import (
    predict_dinov3_feat,
)


class HFDINOAdapter(torch.nn.Module):
    """Adapter wrapping Hugging Face DINOv3 / DINOv2 AutoModel to expose forward_features()."""

    def __init__(self, model_dir: str, device: str = 'cuda'):
        super().__init__()
        from transformers import AutoModel
        self.hf_model = AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
        ).to(device).eval()
        self.patch_size = int(getattr(self.hf_model.config, 'patch_size', 16))
        self.num_register_tokens = int(getattr(self.hf_model.config, 'num_register_tokens', 0))
        self.feature_dim = int(getattr(self.hf_model.config, 'hidden_size', 1280))

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor):
        out = self.hf_model(pixel_values=x)
        patch_tokens = out.last_hidden_state[:, 1 + self.num_register_tokens :, :]
        return {"x_norm_patchtokens": patch_tokens}


class SemanticModel:
    """
    Handles semantic feature extraction and appearance-based cost computation
    using DINOv3 / DINOv2.
    """
    
    def __init__(self, config, device='cuda'):
        """
        Initialize semantic model.
        
        Args:
            config: Configuration dictionary
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.model = None
        self.feature_dim = config['models']['dinov3']['feature_dim']
        
    
    def _load_dinov3_model(self):
        """Load DINOv3 or DINOv2 model from local HF directory or torch.hub checkpoint."""
        dinov3_config = self.config['models']['dinov3']
        model_name = dinov3_config.get('model_name', 'dinov3_vith16plus')
        weight_url = os.path.expandvars(str(dinov3_config.get('weight_url', '')))

        if os.path.isdir(weight_url) and os.path.exists(os.path.join(weight_url, 'config.json')):
            self.model = HFDINOAdapter(weight_url, device=self.device)
            self.feature_dim = self.model.feature_dim
            print(f"[SemanticModel] Loaded HF DINO model ({model_name}) from {weight_url} "
                  f"[dim={self.feature_dim}, patch={self.model.patch_size}, reg={self.model.num_register_tokens}]")
            return

        raise FileNotFoundError(
            f"[SemanticModel] no Hugging Face DINOv3 folder at {weight_url}. "
            "Run: python scripts/download_checkpoints.py --set room"
        )
    
    def extract_features(self, image_np):
        """
        Extract DINOv3 features from an image.
        
        Args:
            image_np: Image as numpy array (H, W, 3)
            
        Returns:
            Feature map (C, H, W)
        """
        return predict_dinov3_feat(image_np, self.model)
    
    def extract_features_batch(self, images_np_list):
        """
        Extract features for a batch of images.
        
        Args:
            images_np_list: List of images as numpy arrays
            
        Returns:
            List of feature maps
        """
        features = []
        for img_np in images_np_list:
            feat = self.extract_features(img_np)
            features.append(feat)
        return features
    
    
    def compute_similarity(self, feat_1, feat_2):
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            feat_1, feat_2: Feature vectors
            
        Returns:
            Similarity score
        """
        return torch.cosine_similarity(feat_1, feat_2, dim=0)
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
