"""
SceneDiff Main Pipeline
=======================

Main orchestrator class for multi-view scene change detection.
"""

import os
import time
import pickle
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import open3d as o3d
from pycocotools import mask as mask_utils
from skimage.filters import threshold_otsu
from torch_scatter import scatter_mean

from .geometry import GeometryModel
from .masks import MaskModel
from .semantics import SemanticModel

from .utils import (
    load_images_as_tensor_from_list,
    voxelization,
    get_img_coor,
    calculate_mask_percentage,
    apply_colormap,
    threshold_maximum_entropy,
    mask_to_rle_pytorch,
    coco_encode_rle,
    merge_objects,
    reprojected_feature,
    get_dino_matched_region_cost,
)


class SceneDiff:
    """
    Main class for multi-view scene change detection.
    
    Orchestrates geometry, mask, and semantic models to detect and track
    changed objects across video sequences.
    """
    
    def __init__(self, config, device='cuda'):
        """
        Initialize SceneDiff pipeline.
        
        Args:
            config: Configuration dictionary
            device: Device for computation
        """
        self.config = config
        self.device = device
        
        # Initialize sub-models
        print("Initializing SceneDiff pipeline...")
        self.geometry_model = GeometryModel(config, device)
        self.mask_model = MaskModel(config, device)
        self.semantic_model = SemanticModel(config, device)
        self.feature_dim = config['models']['dinov3']['feature_dim']
        
        print("SceneDiff pipeline initialized successfully!")
    
    def process_scene(self, file_list, img_1_length, img_2_length, output_dir):
        """
        Main pipeline for scene change detection.
        
        Args:
            file_list: List of image paths (video1 frames + video2 frames)
            img_1_length: Number of frames in video 1
            img_2_length: Number of frames in video 2
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing detected objects and masks
        """
        start_time = time.time()
        self.stage_times = {}
        scene_name = Path(file_list[0]).parents[1].stem
        
        print(f"\n{'='*80}")
        print(f"Processing scene: {scene_name}")
        print(f"{'='*80}\n")
        
        # Stage 1: Load Images
        stage_start = time.time()
        images = load_images_as_tensor_from_list(file_list)[None].to(self.device)
        H, W = images.shape[-2:]
        _dt = time.time() - stage_start
        self.stage_times["Image loading"] = round(_dt, 3)
        print(f"[Stage 1] Image loading: {_dt:.2f}s")
        
        # Stage 2: Depth and Pose Estimation
        stage_start = time.time()
        self.geometry_model._load_pi3_model() # Load the model when needed
        res = self.geometry_model.estimate_depth_and_poses(images)
        intrinsic = self.geometry_model.compute_intrinsics(res, file_list, H, W)
        
        depth_map, point_map, poses, voxel_size = self.geometry_model.normalize_scene_scale(
            res['point_map'][0].cpu().numpy(),
            res['depth_map'],
            res['poses']
        )
        
        pi3_masks = self.geometry_model.get_valid_depth_masks(res)
        _dt = time.time() - stage_start
        self.stage_times["Depth and pose estimation"] = round(_dt, 3)
        print(f"[Stage 2] Depth and pose estimation: {_dt:.2f}s")
        
        # Stage 3: Similarity Matrix
        stage_start = time.time()
        grid_intrinsic = torch.tensor([
            2.0 / W, 0, -1,
            0, 2.0 / H, -1
        ]).reshape(2, 3).to(self.device)
        img_coors = get_img_coor(H, W).to(self.device)
        
        similarity_matrix = self._compute_similarity_matrix(
            poses, depth_map, intrinsic, grid_intrinsic, img_coors,
            H, W, img_1_length, img_2_length
        )
        similarity_weights = self._compute_similarity_weights(similarity_matrix)
        _dt = time.time() - stage_start
        self.stage_times["Similarity matrix"] = round(_dt, 3)
        print(f"[Stage 3] Similarity matrix: {_dt:.2f}s")
        
        # Stage 4: Confidence Maps
        stage_start = time.time()
        conf_list = self._compute_confidence_maps(images, file_list)
        _dt = time.time() - stage_start
        self.stage_times["Confidence computation"] = round(_dt, 3)
        print(f"[Stage 4] Confidence computation: {_dt:.2f}s")
        self.geometry_model.cleanup() # Clean up the model to free up memory

        # Stage 5: SAM Masks
        stage_start = time.time()
        input_dir = Path(file_list[0]).parents[1]
        mask_cache_dir = Path(self.config.get('cache', {}).get('mask_cache_dir', '')) if self.config.get('cache', {}).get('mask_cache_dir') else input_dir
        if str(mask_cache_dir) and mask_cache_dir != input_dir:
            mask_file_path = mask_cache_dir / scene_name / self.mask_model.get_cache_filename()
        else:
            mask_file_path = input_dir / self.mask_model.get_cache_filename()
        masks_list = self.mask_model.load_or_generate_masks(file_list, H, W, mask_file_path)
        _dt = time.time() - stage_start
        self.stage_times["SAM masks"] = round(_dt, 3)
        print(f"[Stage 5] SAM masks: {_dt:.2f}s")
        self.mask_model.cleanup() # Clean up the mask model to free up memory

        # Stage 6: Cost Map Computation
        stage_start = time.time()
        file_output_dir = Path(output_dir) / scene_name
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.semantic_model._load_dinov3_model() # Load the model when needed
        self.feature_dim = self.semantic_model.feature_dim
        total_cost_maps, total_visible_maps = self._compute_cost_maps(
            images, masks_list, poses, depth_map, intrinsic, conf_list,
            similarity_weights, pi3_masks, img_coors, grid_intrinsic,
            H, W, file_output_dir, scene_name, output_dir
        )
        _dt = time.time() - stage_start
        self.stage_times["Cost map computation"] = round(_dt, 3)
        print(f"[Stage 6] Cost map computation: {_dt:.2f}s")
        
        # Stage 7: Voxelization and Feature Extraction
        stage_start = time.time()
        pc1_results, pc2_results, pc1_dino_feat, pc2_dino_feat = self._voxelize_and_extract_features(
            point_map, total_cost_maps, total_visible_maps, images,
            img_1_length, img_2_length, voxel_size, H, W
        )
        _dt = time.time() - stage_start
        self.stage_times["Voxelization and features"] = round(_dt, 3)
        print(f"[Stage 7] Voxelization and features: {_dt:.2f}s")
        
        # Stage 8: Object Detection and Merging
        stage_start = time.time()
        cluster_cost_1, cluster_cost_2 = self._compute_cluster_costs(
            pc1_results, pc2_results, masks_list, total_visible_maps,
            img_1_length, img_2_length, H, W
        )
        
        object_results = self._detect_and_merge_objects(
            cluster_cost_1, cluster_cost_2, total_visible_maps,
            masks_list, pc1_results, pc2_results, pc1_dino_feat, pc2_dino_feat,
            img_1_length, img_2_length, H, W, voxel_size
        )
        _dt = time.time() - stage_start
        self.stage_times["Object detection"] = round(_dt, 3)
        print(f"[Stage 8] Object detection: {_dt:.2f}s")
        print(f"  Detected {len(object_results['merged_objects_1'])} objects in video 1, "
              f"{len(object_results['merged_objects_2'])} objects in video 2")
        self.semantic_model.cleanup() # Clean up the semantic model to free up memory

        # Stage 9: Object Association and Mask Generation
        stage_start = time.time()
        object_masks, unit_ids = self._associate_and_save_objects(
            object_results, images, cluster_cost_1, cluster_cost_2,
            img_1_length, img_2_length, H, W, file_output_dir,
            scene_name=scene_name,
            poses=poses,
            intrinsic=intrinsic,
            similarity_matrix=similarity_matrix,
            total_visible_maps=total_visible_maps,
            voxel_size=voxel_size,
        )
        _dt = time.time() - stage_start
        self.stage_times["Object mask generation"] = round(_dt, 3)
        print(f"[Stage 9] Object mask generation: {_dt:.2f}s")
        
        # Stage 10: Visualization (if enabled)
        if self.config['visualization']['vis_pc']:
            self._save_visualizations(
                point_map, pc1_results, pc2_results, images, object_results, unit_ids,
                img_1_length, file_output_dir
            )
        
        # Stage 11: Ground Truth Visualization (if available)
        if self.config['visualization']['save_gt_visualization']:
            self._save_ground_truth_visualization(
                scene_name, images, img_1_length, H, W, file_output_dir
            )
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"{'='*80}\n")
        
        return object_masks
    
    def _compute_similarity_matrix(self, poses, depth_map, intrinsic, grid_intrinsic,
                                   img_coors, H, W, img_1_length, img_2_length):
        """Compute pairwise similarity between images from two sets."""
        n_total = img_1_length + img_2_length
        similarity_matrix = -1 * torch.ones(n_total, n_total).cuda()
        
        for i in range(img_1_length):
            for j in range(img_2_length):
                set_2_idx = img_1_length + j
                
                sim_1 = calculate_mask_percentage(
                    poses[0][i], poses[0][set_2_idx],
                    depth_map[0][i].permute(2, 0, 1), depth_map[0][set_2_idx].permute(2, 0, 1),
                    intrinsic[0][i], intrinsic[0][set_2_idx],
                    grid_intrinsic, 0.3, img_coors, H, W
                )
                
                sim_2 = calculate_mask_percentage(
                    poses[0][set_2_idx], poses[0][i],
                    depth_map[0][set_2_idx].permute(2, 0, 1), depth_map[0][i].permute(2, 0, 1),
                    intrinsic[0][set_2_idx], intrinsic[0][i],
                    grid_intrinsic, 0.3, img_coors, H, W
                )
                
                similarity_matrix[i, set_2_idx] = (sim_1 + sim_2) / 2
                similarity_matrix[set_2_idx, i] = similarity_matrix[i, set_2_idx]
        
        return similarity_matrix
    
    def _compute_similarity_weights(self, similarity_matrix):
        """Convert similarity matrix to weights for aggregation. 
        For frames without views in another sequence with more than 50% similarity, set the highest similarity weight to 1."""
        percentage_threshold = self.config['processing']['visible_percentage']
        n = similarity_matrix.shape[0]
        weight_matrix = torch.zeros_like(similarity_matrix)
        
        for i in range(n):
            for j in range(n):
                if similarity_matrix[i, j] > percentage_threshold:
                    weight_matrix[i, j] = similarity_matrix[i, j]
            
            if weight_matrix[i].sum() == 0:
                j = torch.argmax(similarity_matrix[i])
                weight_matrix[i, j] = 1
        
        weight_matrix = weight_matrix / weight_matrix.sum(dim=1, keepdim=True)
        return weight_matrix

    
    def compute_rendering_cost(self, feat_src, feat_tgt,
                              intrinsic_src, pose_src, depth_src,
                              intrinsic_tgt, pose_tgt,
                              img_coors, grid_intrinsic, H, W,
                              occlusion_mask, sam_mask, vis_mask):
        """
        Compute rendering-based appearance cost.
        
        Args:
            feat_src, feat_tgt: Source and target features
            intrinsic_src, pose_src, depth_src: Source camera parameters
            intrinsic_tgt, pose_tgt: Target camera parameters
            img_coors: Image coordinates
            grid_intrinsic: Grid intrinsic matrix
            H, W: Image dimensions
            occlusion_mask: Occlusion mask
            sam_mask: Segmentation mask
            vis_mask: Visibility mask
            
        Returns:
            Region-aggregated rendering cost (H, W)
        """
        # Reproject features
        feat_reprojected = reprojected_feature(
            intrinsic_src.cuda(), pose_src.cuda(), depth_src.view(1, H, W),
            intrinsic_tgt.cuda(), pose_tgt.cuda(), feat_tgt,
            img_coors, grid_intrinsic, H, W
        )
        
        # Compute cosine distance
        diff = 1 - torch.cosine_similarity(feat_reprojected, feat_src, dim=0)
        diff[occlusion_mask] = 0

        # # Visualize features using PCA and plt.imshow
        # from sklearn.decomposition import PCA
        # import numpy as np

        # # Get features as (C, H*W)
        # feat_reprojected_flat = feat_reprojected.view(feat_reprojected.shape[0], -1).permute(1, 0).cpu().detach().numpy()  # shape (H*W, C)
        # feat_src_flat = feat_src.view(feat_src.shape[0], -1).permute(1, 0).cpu().detach().numpy()  # shape (H*W, C)

        # # Fit PCA on concatenated features (to align spaces)
        # X = np.concatenate([feat_reprojected_flat, feat_src_flat], axis=0)
        # pca = PCA(n_components=3)
        # X_pca = pca.fit_transform(X)
        # feat_reprojected_pca = X_pca[:feat_reprojected_flat.shape[0]].reshape(H, W, 3)
        # feat_src_pca = X_pca[feat_reprojected_flat.shape[0]:].reshape(H, W, 3)

        # # Normalize for visualization
        # feat_reprojected_img = (feat_reprojected_pca - feat_reprojected_pca.min()) / (feat_reprojected_pca.max() - feat_reprojected_pca.min())
        # feat_src_img = (feat_src_pca - feat_src_pca.min()) / (feat_src_pca.max() - feat_src_pca.min())

        # plt.figure(figsize=(10,5))
        # plt.subplot(1,2,1)
        # plt.title("feat_reprojected PCA")
        # plt.imshow(feat_reprojected_img)
        # plt.axis('off')
        # plt.subplot(1,2,2)
        # plt.title("feat_src PCA")
        # plt.imshow(feat_src_img)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        
        # Aggregate to regions
        region_diff = torch.zeros_like(sam_mask).float()
        for region_id in sam_mask.unique():
            if region_id == -1:
                continue
            region_mask = sam_mask == region_id
            valid_mask = region_mask & ~occlusion_mask
            
            if valid_mask.sum() > 0:
                region_diff[region_mask] = diff[valid_mask].mean()
        
        return region_diff
    
    def aggregate_features_to_voxels(self, images, voxel_indices, H, W):
        """
        Extract features and aggregate to voxels.
        
        Args:
            images: Input images (N, C, H, W)
            voxel_indices: Point-to-voxel mapping (N*H*W,)
            H, W: Image dimensions
            
        Returns:
            Voxel-aggregated features (N_voxels, feature_dim)
        """
        from torch_scatter import scatter_mean
        
        n_images = images.shape[0]
        feat_list = torch.zeros(n_images, H, W, self.feature_dim).to(self.device)
        
        for i in range(n_images):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            feat = self.semantic_model.extract_features(img_np)
            feat_list[i] = feat.permute(1, 2, 0)
        
        feat_flat = feat_list.view(-1, self.feature_dim)
        voxel_features = scatter_mean(
            feat_flat.cpu(),
            voxel_indices.cpu(),
            dim=0
        ).cuda()
        
        return voxel_features

    def _compute_confidence_maps(self, images, file_list):
        """Compute confidence maps for all images."""
        conf_list = []
        for i in range(len(file_list)):
            conf = self.geometry_model.compute_per_image_confidence(images[0][i:i+1].clone())
            conf_list.append([conf])
        return conf_list
    
    def _compute_cost_maps(self, images, masks_list, poses, depth_map, intrinsic,
                          conf_list, similarity_weights, pi3_masks, img_coors,
                          grid_intrinsic, H, W, file_output_dir, scene_name, output_dir):
        """Compute cost maps for all images."""
        # Check for cache
        cache_config = self.config['cache']
        cached = None
        if cache_config['use_cache']:
            cached = self._load_cached_cost_maps(output_dir, scene_name)
            if cached is not None:
                total_cost_map_list, total_visible_map_list = cached
        
        if (cached is None) or (not cache_config['use_cache']):
            # Compute costs
            total_cost_map_list = []
            total_visible_map_list = []
            
            for idx in range(len(masks_list)):
                weighted_cost, visible_map = self._compute_cost_maps_for_image(
                    idx, images, masks_list, poses, depth_map, intrinsic,
                    conf_list, similarity_weights, pi3_masks, img_coors,
                    grid_intrinsic, H, W
                )
                total_cost_map_list.append(weighted_cost)
                total_visible_map_list.append(visible_map)
            
            # Save cache if enabled
            if cache_config['save_cache']:
                self._save_cost_maps(file_output_dir, total_cost_map_list, total_visible_map_list)
        
        # Aggregate costs
        total_cost_map_list = torch.stack(total_cost_map_list)
        self._last_cost_components = total_cost_map_list.detach().cpu()
        costs_config = self.config['costs']
        total_cost_maps = (
            costs_config['lambda_geo'] * total_cost_map_list[:, 0] +
            costs_config['lambda_dino_region_match'] * total_cost_map_list[:, 1] +
            costs_config['lambda_dino'] * total_cost_map_list[:, 2]
        )
        total_cost_maps = total_cost_maps.to(self.device)
        total_visible_maps = torch.stack(total_visible_map_list, dim=0).to(self.device)
        
        return total_cost_maps, total_visible_maps
    
    def _compute_cost_maps_for_image(self, index, images, masks_list, poses, depth_map,
                                    intrinsic, conf_list, similarity_weights, pi3_masks,
                                    img_coors, grid_intrinsic, H, W):
        """Compute cost maps for a single image."""
        cost_map_list = []
        visible_map_list = []
        
        img1_np = images[0][index].permute(1, 2, 0).cpu().numpy()
        sam_mask_1 = torch.tensor(masks_list[index]).cuda()
        pose_1 = poses[0][index].clone()
        depth_1 = depth_map[0][index].view(1, H, W).clone()
        intrinsic_1 = intrinsic[0][index].clone()
        conf_1 = conf_list[index][0]
        
        # Extract features once
        dinov3_feat_1 = self.semantic_model.extract_features(img1_np)
        
        for closest_index in similarity_weights[index].argsort(dim=0, descending=True):
            if similarity_weights[index, closest_index] == 0:
                break
            
            img2_np = images[0][closest_index].permute(1, 2, 0).cpu().numpy()
            sam_mask_2 = torch.tensor(masks_list[closest_index]).cuda()
            pose_2 = poses[0][closest_index].clone()
            depth_2 = depth_map[0][closest_index].view(1, H, W).clone()
            intrinsic_2 = intrinsic[0][closest_index].clone()
            conf_2 = conf_list[closest_index][0]
            
            # Compute geometric costs
            geo_results = self.geometry_model.compute_geometric_costs(
                pose_1, depth_1, intrinsic_1, conf_1,
                pose_2, depth_2, intrinsic_2, conf_2,
                img_coors, grid_intrinsic, H, W
            )
            
            # Aggregate geometric costs to regions
            fg_region_geo_cost_1 = self.mask_model.aggregate_to_regions(
                geo_results['geo_cost_1'] * geo_results['vis_1'], sam_mask_1
            )
            
            # Compute semantic costs
            dinov3_feat_2 = self.semantic_model.extract_features(img2_np)
            
            occlusion_threshold = self.config['processing']['occlusion_threshold']
            occlusion_mask_1 = (
                (geo_results['foreground_dis_1'] < occlusion_threshold) |
                ~pi3_masks[index] | ~geo_results['vis_1']
            )
            occlusion_mask_2 = (
                (geo_results['foreground_dis_2'] < occlusion_threshold) |
                ~pi3_masks[closest_index] | ~geo_results['vis_2']
            )
            
            # DINOv3 region matching cost
            dino_region_match_cost_1, _, _, _, _, _ = get_dino_matched_region_cost(
                dinov3_feat_1, dinov3_feat_2, sam_mask_1, sam_mask_2,
                ~occlusion_mask_1, ~occlusion_mask_2
            )
            
            # DINOv3 rendering cost
            dino_render_cost_1 = self.compute_rendering_cost(
                dinov3_feat_1, dinov3_feat_2, intrinsic_1, pose_1, depth_1,
                intrinsic_2, pose_2, img_coors, grid_intrinsic, H, W,
                occlusion_mask_1, sam_mask_1, geo_results['vis_1']
            )
            
            # Stack costs
            costs = torch.stack([
                (fg_region_geo_cost_1 * geo_results['vis_1']).cpu(),
                (dino_region_match_cost_1 * geo_results['vis_1']).cpu(),
                (dino_render_cost_1 * geo_results['vis_1']).cpu()
            ], dim=0)
            
            cost_map_list.append(costs)
            visible_map_list.append((~occlusion_mask_1).cpu())
        
        # Aggregate costs across views
        cost_maps = torch.stack(cost_map_list)
        visible_maps = torch.stack(visible_map_list)
        
        visible_count = visible_maps.sum(dim=0)
        weighted_cost = (cost_maps * visible_maps[:, None, :, :]).sum(dim=0)
        weighted_cost = weighted_cost / (visible_count + 1e-6)[None, :, :]
        weighted_cost[visible_count[None].repeat(3, 1, 1) == 0] = 0
        
        return weighted_cost, visible_count > 0
    
    def _voxelize_and_extract_features(self, point_map, total_cost_maps, total_visible_maps,
                                       images, img_1_length, img_2_length, voxel_size, H, W):
        """Voxelize point clouds and extract semantic features."""
        # Process point clouds
        pc1_results = self._process_point_cloud_costs(
            point_map[:img_1_length], total_cost_maps[:img_1_length],
            total_visible_maps[:img_1_length], voxel_size, H, W
        )
        pc2_results = self._process_point_cloud_costs(
            point_map[img_1_length:], total_cost_maps[img_1_length:],
            total_visible_maps[img_1_length:], voxel_size, H, W
        )
        
        # Extract DINOv3 features
        pc1_dino_feat = self.aggregate_features_to_voxels(
            images[0][:img_1_length], pc1_results['voxel_indices'], H, W
        )
        pc2_dino_feat = self.aggregate_features_to_voxels(
            images[0][img_1_length:], pc2_results['voxel_indices'], H, W
        )
        
        return pc1_results, pc2_results, pc1_dino_feat, pc2_dino_feat
    
    def _process_point_cloud_costs(self, point_map, cost_maps, visible_masks, voxel_size, H, W):
        """Process costs for a point cloud through voxelization."""
        points = point_map.reshape(-1, 3)
        
        # Voxelization
        point_to_voxel = voxelization(
            torch.tensor(points[None], device=self.device), voxel_size
        )[0]
        
        # Aggregate costs to voxels
        visible_points = visible_masks.view(-1)
        visible_costs = cost_maps.view(-1)[visible_points]
        visible_voxel_indices = point_to_voxel[visible_points]
        
        cost_per_voxel = scatter_mean(visible_costs, visible_voxel_indices, dim=0)
        
        # Create full cost map
        max_voxel_id = point_to_voxel.unique().max()
        full_cost_map = torch.zeros(max_voxel_id + 1, device=self.device)
        for voxel_id in visible_voxel_indices.unique():
            full_cost_map[voxel_id] = cost_per_voxel[voxel_id]
        
        point_costs = full_cost_map[point_to_voxel]
        
        # Compute voxel locations
        voxel_locations = scatter_mean(
            torch.tensor(points, device=self.device), point_to_voxel, dim=0
        )
        
        return {
            'cost_maps': point_costs,
            'voxel_indices': point_to_voxel,
            'voxel_locations': voxel_locations
        }
    
    def _compute_cluster_costs(self, pc1_results, pc2_results, masks_list, total_visible_maps,
                               img_1_length, img_2_length, H, W):
        """Aggregate voxel costs to SAM regions."""
        cluster_cost_1 = self._compute_cluster_costs_single(
            pc1_results['cost_maps'], masks_list[:img_1_length],
            total_visible_maps, img_1_length, H, W, offset=0
        )
        cluster_cost_2 = self._compute_cluster_costs_single(
            pc2_results['cost_maps'], masks_list[img_1_length:],
            total_visible_maps, img_2_length, H, W, offset=img_1_length
        )
        
        return cluster_cost_1, cluster_cost_2
    
    def _compute_cluster_costs_single(self, cost_maps, masks_list, total_visible_maps,
                                      n_images, H, W, offset=0):
        """Aggregate costs to regions for a single set."""
        cluster_costs = []
        
        for idx in range(n_images):
            cluster_cost = torch.zeros(H, W).to(self.device)
            sam_mask = torch.tensor(masks_list[idx]).cuda()
            cost_map = cost_maps.view(-1, H, W)[idx].cuda()
            visible_map = total_visible_maps.view(-1, H, W)[idx + offset].cuda()
            
            for region_id in sam_mask.unique():
                if region_id == -1:
                    continue
                
                region_mask = sam_mask == region_id
                valid_mask = region_mask & visible_map
                
                if (valid_mask.sum() / region_mask.sum()) > 0.3:
                    cluster_cost[region_mask] = cost_map[valid_mask].mean()
            
            cluster_costs.append(cluster_cost)
        
        return torch.stack(cluster_costs)
    
    def _detect_and_merge_objects(self, cluster_cost_1, cluster_cost_2, total_visible_maps,
                                  masks_list, pc1_results, pc2_results, pc1_dino_feat,
                                  pc2_dino_feat, img_1_length, img_2_length, H, W, voxel_size):
        """Detect objects by thresholding and merge across frames."""
        detection_config = self.config['detection']
        merging_config = self.config['merging']
        
        combined_cost = torch.cat([cluster_cost_1, cluster_cost_2], dim=0)
        threshold = self._compute_changed_region_threshold(combined_cost, total_visible_maps)
        # Detect objects in video 1
        object_list_1, mask_index_1, threshold_1 = self._detect_objects(
            cluster_cost_1,
            total_visible_maps[:img_1_length],
            masks_list[:img_1_length],
            pc1_results['voxel_locations'],
            pc1_results['voxel_indices'],
            pc1_dino_feat,
            img_1_length, 
            threshold,
            H, W
        )
        
        # Detect objects in video 2
        object_list_2, mask_index_2, threshold_2 = self._detect_objects(
            cluster_cost_2,
            total_visible_maps[img_1_length:],
            masks_list[img_1_length:],
            pc2_results['voxel_locations'],
            pc2_results['voxel_indices'],
            pc2_dino_feat,
            img_2_length,
            threshold,
            H, W
        )
        
        # Merge objects across frames
        object_id_list_1, merged_object_list_1 = merge_objects(
            object_list_1,
            geometry_distance_threshold=merging_config['geometry_distance_threshold_of_voxel_size'] * voxel_size,
            visual_threshold_ratio=merging_config['visual_threshold_ratio'],
            geometry_threshold_ratio=merging_config['geometry_threshold_ratio'],
            general_threshold=merging_config['general_threshold']
        )
        
        object_id_list_2, merged_object_list_2 = merge_objects(
            object_list_2,
            geometry_distance_threshold=merging_config['geometry_distance_threshold_of_voxel_size'] * voxel_size,
            visual_threshold_ratio=merging_config['visual_threshold_ratio'],
            geometry_threshold_ratio=merging_config['geometry_threshold_ratio'],
            general_threshold=merging_config['general_threshold']
        )
        
        return {
            'object_list_1': object_list_1,
            'object_list_2': object_list_2,
            'mask_index_1': mask_index_1,
            'mask_index_2': mask_index_2,
            'threshold_1': threshold_1,
            'threshold_2': threshold_2,
            'object_id_list_1': object_id_list_1,
            'object_id_list_2': object_id_list_2,
            'merged_objects_1': merged_object_list_1,
            'merged_objects_2': merged_object_list_2
        }
    
    def _compute_changed_region_threshold(self, combined_cost, visible_maps):
        '''
        Compute the threshold for changed regions. 
        If change_region_threshold is not specified, compute the threshold using the visible cost values.
        '''
        detection_config = self.config['detection']
        
        # Get visible cost values
        visible_costs = combined_cost[visible_maps]
        
        # Compute threshold
        if detection_config['change_region_threshold'] is not None:
            threshold = detection_config['change_region_threshold']
        else:
            quantile_val = torch.quantile(visible_costs, detection_config['filter_percentage_before_threshold'])
            filtered_costs = visible_costs[visible_costs > quantile_val]
            
            if detection_config['threshold_method'] == 'otsu':
                threshold = threshold_otsu(filtered_costs.view(-1)[None].cpu().numpy())
            elif detection_config['threshold_method'] == 'max_entropy':
                threshold = threshold_maximum_entropy(filtered_costs)
            else:
                raise ValueError(f"Invalid threshold method: {detection_config['threshold_method']}")
            
        return threshold
    
    def _detect_objects(self, cluster_cost_maps, visible_maps, masks_list,
                       voxel_locations, voxel_indices, dino_features,
                       img_length, threshold, H, W):
        """Detect objects by thresholding cost maps."""
        detection_config = self.config['detection']
        
        # Threshold to get object masks
        object_masks = cluster_cost_maps > threshold
        mask_indices = -1 * torch.ones_like(object_masks).int()
        object_list = []
        mask_counter = 0
        
        for idx in range(img_length):
            image_objects = []
            sam_mask = torch.tensor(masks_list[idx]).cuda()
            sam_mask[~object_masks[idx]] = -1
            
            for region_id in sam_mask.unique():
                if region_id == -1:
                    continue
                
                region_mask = sam_mask == region_id
                if region_mask.sum() < detection_config['min_detection_pixel']:
                    continue
                
                # Extract voxel points and features
                voxel_ids = voxel_indices.view(-1, H, W)[idx][region_mask]
                points = voxel_locations[voxel_ids].unique(dim=0).view(-1, 3)
                feat = dino_features[voxel_ids.unique()].mean(dim=0)
                
                obj_dict = {
                    'pc': points,
                    'dino_feat': feat
                }
                image_objects.append(obj_dict)
                mask_indices[idx][region_mask] = mask_counter
                mask_counter += 1
            
            object_list.append(image_objects)
        
        return object_list, mask_indices, threshold
    
    def _associate_and_save_objects(self, object_results, images, cluster_cost_1, cluster_cost_2,
                                    img_1_length, img_2_length, H, W, output_dir,
                                    scene_name="", poses=None, intrinsic=None,
                                    similarity_matrix=None, total_visible_maps=None,
                                    voxel_size=0.01):
        """Associate objects between videos and save masks, candidates, and verifier cards."""
        # Compute object similarity matrix
        merged_1 = object_results['merged_objects_1']
        merged_2 = object_results['merged_objects_2']
        
        object_sim_matrix = torch.zeros(len(merged_1), len(merged_2))
        for i, obj1 in enumerate(merged_1):
            for j, obj2 in enumerate(merged_2):
                object_sim_matrix[i, j] = self.semantic_model.compute_similarity(
                    obj1['dino_feat'], obj2['dino_feat']
                )
        
        # Create unified IDs
        unit_ids = self._create_unified_object_ids(
            object_results['object_id_list_1'],
            object_results['object_id_list_2'],
            object_sim_matrix
        )
        
        # Prepare colors
        colors = self._create_color_palette()
        max_score = self.config['detection']['max_score']
        
        # Save object masks
        object_masks = {'H': H, 'W': W}
        
        masks_1 = self._save_object_masks(
            object_results['object_id_list_1'],
            object_results['mask_index_1'],
            unit_ids, images, cluster_cost_1,
            img_1_length, H, W, colors, max_score, 'video_1', output_dir
        )
        
        masks_2 = self._save_object_masks(
            object_results['object_id_list_2'],
            object_results['mask_index_2'],
            unit_ids, images, cluster_cost_2,
            img_2_length, H, W, colors, max_score, 'video_2', output_dir, img_1_length
        )
        
        # Merge masks
        for obj_id, obj_data in masks_1.items():
            if obj_id not in object_masks:
                object_masks[obj_id] = {}
            object_masks[obj_id].update(obj_data)
        
        for obj_id, obj_data in masks_2.items():
            if obj_id not in object_masks:
                object_masks[obj_id] = {}
            object_masks[obj_id].update(obj_data)
        
        # Save to file
        output_masks_file = output_dir / 'object_masks.pkl'
        with open(output_masks_file, 'wb') as f:
            pickle.dump(object_masks, f)

        # Export 3D candidates and multi-view verifier evidence cards
        try:
            self._export_verifier_cards_and_candidates(
                scene_name=scene_name,
                object_masks=object_masks,
                object_results=object_results,
                unit_ids=unit_ids,
                object_sim_matrix=object_sim_matrix,
                images=images,
                poses=poses,
                intrinsic=intrinsic,
                similarity_matrix=similarity_matrix,
                total_visible_maps=total_visible_maps,
                voxel_size=voxel_size,
                img_1_length=img_1_length,
                img_2_length=img_2_length,
                H=H,
                W=W,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"[Warning] Failed to export verifier cards for {scene_name}: {e}")
        
        return object_masks, unit_ids

    def _project_pc_to_bbox(self, pc_tensor, pose_c2w, K_mat, H, W):
        """Project 3D world point cloud (N, 3) into target camera and return [x0, y0, x1, y1]."""
        if pc_tensor is None or pc_tensor.numel() == 0:
            return [0, 0, W - 1, H - 1]
        pts = pc_tensor.float().to(pose_c2w.device)
        ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
        h_pts = torch.cat([pts, ones], dim=-1)
        cam_pts = (torch.inverse(pose_c2w.float()) @ h_pts.t()).t()[:, :3]
        valid_z = cam_pts[:, 2] > 1e-3
        if valid_z.sum() == 0:
            return [0, 0, W - 1, H - 1]
        cam_pts = cam_pts[valid_z]
        proj = (K_mat.float().to(pts.device) @ cam_pts.t()).t()
        uv = proj[:, :2] / proj[:, 2:3]
        u = uv[:, 0].clamp(0, W - 1)
        v = uv[:, 1].clamp(0, H - 1)
        x0 = int(torch.quantile(u, 0.02).item())
        y0 = int(torch.quantile(v, 0.02).item())
        x1 = int(torch.quantile(u, 0.98).item())
        y1 = int(torch.quantile(v, 0.98).item())
        if x1 <= x0 + 4:
            x0, x1 = max(0, x0 - 16), min(W - 1, x1 + 16)
        if y1 <= y0 + 4:
            y0, y1 = max(0, y0 - 16), min(H - 1, y1 + 16)
        return [x0, y0, x1, y1]

    def _export_verifier_cards_and_candidates(
        self, scene_name, object_masks, object_results, unit_ids, object_sim_matrix,
        images, poses, intrinsic, similarity_matrix, total_visible_maps,
        voxel_size, img_1_length, img_2_length, H, W, output_dir
    ):
        """Save scene_candidates.pkl, verifier_cards.json, and verifier_crops/ for VLM/RL."""
        import json
        import numpy as np
        from PIL import Image as PILImage, ImageDraw

        merged_1 = object_results['merged_objects_1']
        merged_2 = object_results['merged_objects_2']

        # Map unified_id -> merged object index in video_1 and video_2
        uid_to_m1 = {u: m_idx for m_idx, u in unit_ids.get('video_1', {}).items()}
        uid_to_m2 = {u: m_idx for m_idx, u in unit_ids.get('video_2', {}).items()}

        # Save compact 3D candidate data
        candidates_payload = {
            'scene_name': scene_name,
            'H': int(H),
            'W': int(W),
            'img_1_length': int(img_1_length),
            'img_2_length': int(img_2_length),
            'voxel_size': float(voxel_size),
            'unit_ids': unit_ids,
            'object_sim_matrix': object_sim_matrix.detach().cpu().numpy(),
            'poses': poses[0].detach().cpu().numpy() if poses is not None else None,
            'intrinsic': intrinsic[0].detach().cpu().numpy() if intrinsic is not None else None,
            'merged_1': [
                {
                    'pc': m['pc'].detach().cpu().numpy(),
                    'dino_feat': m['dino_feat'].detach().cpu().numpy(),
                }
                for m in merged_1
            ],
            'merged_2': [
                {
                    'pc': m['pc'].detach().cpu().numpy(),
                    'dino_feat': m['dino_feat'].detach().cpu().numpy(),
                }
                for m in merged_2
            ],
        }
        with open(output_dir / 'scene_candidates.pkl', 'wb') as f:
            pickle.dump(candidates_payload, f)

        crops_dir = output_dir / 'verifier_crops'
        crops_dir.mkdir(parents=True, exist_ok=True)
        cost_comps = getattr(self, '_last_cost_components', None)

        cards = []
        for uid, obj_data in object_masks.items():
            if uid in ('H', 'W'):
                continue
            uid_int = int(uid)
            v1_frames = obj_data.get('video_1', {})
            v2_frames = obj_data.get('video_2', {})
            has_v1 = len(v1_frames) > 0
            has_v2 = len(v2_frames) > 0
            if not has_v1 and not has_v2:
                continue

            if has_v1 and has_v2:
                cand_type = "moved"
            elif has_v1:
                cand_type = "removed"
            else:
                cand_type = "added"

            m1_idx = uid_to_m1.get(uid_int, None)
            m2_idx = uid_to_m2.get(uid_int, None)
            pc1 = merged_1[m1_idx]['pc'] if (m1_idx is not None and m1_idx < len(merged_1)) else None
            pc2 = merged_2[m2_idx]['pc'] if (m2_idx is not None and m2_idx < len(merged_2)) else None

            c1 = pc1.mean(dim=0).detach().cpu().numpy().tolist() if (pc1 is not None and pc1.numel() > 0) else None
            c2 = pc2.mean(dim=0).detach().cpu().numpy().tolist() if (pc2 is not None and pc2.numel() > 0) else None
            b1_min = pc1.min(dim=0).values.detach().cpu().numpy().tolist() if (pc1 is not None and pc1.numel() > 0) else None
            b1_max = pc1.max(dim=0).values.detach().cpu().numpy().tolist() if (pc1 is not None and pc1.numel() > 0) else None
            b2_min = pc2.min(dim=0).values.detach().cpu().numpy().tolist() if (pc2 is not None and pc2.numel() > 0) else None
            b2_max = pc2.max(dim=0).values.detach().cpu().numpy().tolist() if (pc2 is not None and pc2.numel() > 0) else None

            delta_3d = float(np.linalg.norm(np.array(c1) - np.array(c2))) if (c1 is not None and c2 is not None) else 0.0
            dino_sim = float(object_sim_matrix[m1_idx, m2_idx].item()) if (m1_idx is not None and m2_idx is not None and m1_idx < object_sim_matrix.shape[0] and m2_idx < object_sim_matrix.shape[1]) else 0.0

            # Pick best frame in video_1 and video_2
            if has_v1:
                best_f1 = int(max(v1_frames.keys(), key=lambda k: v1_frames[k]['mask_size']))
            else:
                best_f2_tmp = int(max(v2_frames.keys(), key=lambda k: v2_frames[k]['mask_size']))
                if similarity_matrix is not None:
                    best_f1 = int(torch.argmax(similarity_matrix[:img_1_length, img_1_length + best_f2_tmp]).item())
                else:
                    best_f1 = 0

            if has_v2:
                best_f2 = int(max(v2_frames.keys(), key=lambda k: v2_frames[k]['mask_size']))
            else:
                if similarity_matrix is not None:
                    best_f2 = int(torch.argmax(similarity_matrix[best_f1, img_1_length:]).item())
                else:
                    best_f2 = 0

            cross_view_sim = float(similarity_matrix[best_f1, img_1_length + best_f2].item()) if similarity_matrix is not None else 0.0

            # Decode masks and compute 2D bboxes + component costs
            c_geo_list, c_match_list, c_render_list, vis_cov_list, score_list = [], [], [], [], []

            if has_v1 and best_f1 in v1_frames:
                m1_bin = mask_utils.decode(v1_frames[best_f1]['mask']).astype(bool)
                ys, xs = np.where(m1_bin)
                bbox_2d_1 = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else [0, 0, W - 1, H - 1]
                score_list.extend([float(v['cost']) for v in v1_frames.values()])
                if cost_comps is not None and m1_bin.any():
                    cc = cost_comps[best_f1]
                    c_geo_list.append(float(cc[0][m1_bin].mean().item()))
                    c_match_list.append(float(cc[1][m1_bin].mean().item()))
                    c_render_list.append(float(cc[2][m1_bin].mean().item()))
                if total_visible_maps is not None and m1_bin.any():
                    vis_cov_list.append(float(total_visible_maps[best_f1].detach().cpu().numpy()[m1_bin].mean()))
            else:
                bbox_2d_1 = self._project_pc_to_bbox(pc2, poses[0][best_f1], intrinsic[0][best_f1], H, W) if (poses is not None and intrinsic is not None) else [0, 0, W - 1, H - 1]

            if has_v2 and best_f2 in v2_frames:
                m2_bin = mask_utils.decode(v2_frames[best_f2]['mask']).astype(bool)
                ys, xs = np.where(m2_bin)
                bbox_2d_2 = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 0 else [0, 0, W - 1, H - 1]
                score_list.extend([float(v['cost']) for v in v2_frames.values()])
                if cost_comps is not None and m2_bin.any():
                    cc = cost_comps[img_1_length + best_f2]
                    c_geo_list.append(float(cc[0][m2_bin].mean().item()))
                    c_match_list.append(float(cc[1][m2_bin].mean().item()))
                    c_render_list.append(float(cc[2][m2_bin].mean().item()))
                if total_visible_maps is not None and m2_bin.any():
                    vis_cov_list.append(float(total_visible_maps[img_1_length + best_f2].detach().cpu().numpy()[m2_bin].mean()))
            else:
                bbox_2d_2 = self._project_pc_to_bbox(pc1, poses[0][img_1_length + best_f2], intrinsic[0][img_1_length + best_f2], H, W) if (poses is not None and intrinsic is not None) else [0, 0, W - 1, H - 1]

            # Render crops and side-by-side image
            img1_rgb = (images[0][best_f1].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img2_rgb = (images[0][img_1_length + best_f2].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            def _crop_and_annotate(img_arr, box, box_color):
                pil_full = PILImage.fromarray(img_arr.copy())
                draw = ImageDraw.Draw(pil_full)
                x0, y0, x1, y1 = box
                draw.rectangle([x0, y0, x1, y1], outline=box_color, width=3)
                bw, bh = max(16, x1 - x0), max(16, y1 - y0)
                pad_x, pad_y = int(max(32, bw * 0.4)), int(max(32, bh * 0.4))
                cx0, cy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
                cx1, cy1 = min(W, x1 + pad_x), min(H, y1 + pad_y)
                pil_crop = pil_full.crop((cx0, cy0, cx1, cy1))
                return pil_full, pil_crop

            pil_v1_full, pil_v1_crop = _crop_and_annotate(img1_rgb, bbox_2d_1, (255, 64, 64) if has_v1 else (255, 200, 0))
            pil_v2_full, pil_v2_crop = _crop_and_annotate(img2_rgb, bbox_2d_2, (64, 255, 64) if has_v2 else (255, 200, 0))

            v1_crop_rel = f"verifier_crops/obj_{uid_int}_v1_crop.jpg"
            v2_crop_rel = f"verifier_crops/obj_{uid_int}_v2_crop.jpg"
            pair_rel = f"verifier_crops/obj_{uid_int}_pair.jpg"

            pil_v1_crop.save(output_dir / v1_crop_rel, quality=92)
            pil_v2_crop.save(output_dir / v2_crop_rel, quality=92)

            # Side-by-side full frame with bbox overlay
            pair_img = PILImage.new('RGB', (W * 2, H))
            pair_img.paste(pil_v1_full, (0, 0))
            pair_img.paste(pil_v2_full, (W, 0))
            pair_img.save(output_dir / pair_rel, quality=90)

            cards.append({
                'scene_name': scene_name,
                'obj_id': uid_int,
                'candidate_type': cand_type,
                'has_video_1': has_v1,
                'has_video_2': has_v2,
                'n_frames_v1': len(v1_frames),
                'n_frames_v2': len(v2_frames),
                'best_frame_v1': best_f1,
                'best_frame_v2': best_f2,
                'bbox_2d_v1': bbox_2d_1,
                'bbox_2d_v2': bbox_2d_2,
                'centroid_3d_v1': c1,
                'centroid_3d_v2': c2,
                'bbox_3d_min_v1': b1_min,
                'bbox_3d_max_v1': b1_max,
                'bbox_3d_min_v2': b2_min,
                'bbox_3d_max_v2': b2_max,
                'delta_3d': delta_3d,
                'dino_similarity': dino_sim,
                'cross_view_overlap': cross_view_sim,
                'mean_score': float(np.mean(score_list)) if score_list else 0.0,
                'max_score': float(np.max(score_list)) if score_list else 0.0,
                'c_geo': float(np.mean(c_geo_list)) if c_geo_list else 0.0,
                'c_match': float(np.mean(c_match_list)) if c_match_list else 0.0,
                'c_render': float(np.mean(c_render_list)) if c_render_list else 0.0,
                'vis_coverage': float(np.mean(vis_cov_list)) if vis_cov_list else 0.0,
                'v1_crop_path': v1_crop_rel,
                'v2_crop_path': v2_crop_rel,
                'pair_image_path': pair_rel,
            })

        with open(output_dir / 'verifier_cards.json', 'w') as f:
            json.dump({'scene_name': scene_name, 'candidates': cards}, f, indent=2)
    
    def _create_unified_object_ids(self, object_id_list_1, object_id_list_2, object_sim_matrix):
        """Create unified object IDs across two video sequences."""
        similarity_threshold = self.config['merging']['object_similarity_threshold']
        
        unit_ids = {'video_1': {}, 'video_2': {}}
        next_id = 0
        
        # Assign IDs to video 1 objects
        for obj_id in object_id_list_1.unique():
            unit_ids['video_1'][obj_id.item()] = next_id
            next_id += 1
        
        # Assign IDs to video 2 objects (reuse IDs for matched objects)
        for obj_id in object_id_list_2.unique():
            if object_sim_matrix.numel() == 0:
                max_sim = 0
            else:
                similarity_scores = object_sim_matrix[:, obj_id]
                max_sim = similarity_scores.max()
            
            if max_sim > similarity_threshold:
                matched_obj_id = object_id_list_1.unique()[similarity_scores.argmax()]
                unit_ids['video_2'][obj_id.item()] = unit_ids['video_1'][matched_obj_id.item()]
            else:
                unit_ids['video_2'][obj_id.item()] = next_id
                next_id += 1
        
        return unit_ids
    
    def _save_object_masks(self, object_id_list, mask_indices, unit_ids, images,
                          total_cluster_cost_maps, current_sequence_length, H, W, colors,
                          max_score, video_key, output_dir, first_sequence_length=0):
        """Save object masks and visualizations."""
        object_masks = {}
        vis_dir = output_dir / f'{video_key}_detection'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        for obj_idx, object_id in enumerate(object_id_list.unique()):
            object_id_indices = torch.where(object_id_list == object_id)[0]
            unified_id = unit_ids[video_key][object_id.item()]
            color = colors[unified_id % colors.shape[0]]
            
            if unified_id not in object_masks:
                object_masks[unified_id] = {}
            if video_key not in object_masks[unified_id]:
                object_masks[unified_id][video_key] = {}
            
            for single_idx in object_id_indices:
                single_mask = mask_indices == single_idx
                
                for frame_idx in range(current_sequence_length):
                    frame_mask = single_mask.view(current_sequence_length, -1)[frame_idx].view(H, W)
                    
                    if not frame_mask.any():
                        continue
                    
                    # Merge with existing mask if present
                    if frame_idx in object_masks[unified_id][video_key]:
                        old_mask = torch.tensor(mask_utils.decode(
                            object_masks[unified_id][video_key][frame_idx]['mask']
                        )).cuda()
                        frame_mask = frame_mask | old_mask
                    
                    # Convert to RLE
                    rle = mask_to_rle_pytorch(frame_mask.unsqueeze(0))[0]
                    rle = coco_encode_rle(rle)
                    
                    # Compute cost
                    cost = total_cluster_cost_maps[frame_idx][frame_mask.bool()].mean().item()
                    
                    object_masks[unified_id][video_key][frame_idx] = {
                        'mask': rle,
                        'mask_size': frame_mask.sum().item(),
                        'cost': cost / max_score
                    }
                    
                    # Save visualization
                    if self.config['visualization']['save_detections']:
                        img_offset = 0 if video_key == 'video_1' else first_sequence_length
                        rgb_image = images[0][frame_idx + img_offset].permute(1, 2, 0).cpu().numpy()
                        rgb_image[frame_mask.cpu().numpy().astype(bool)] = color.cpu().numpy()
                        vis_path = vis_dir / f'{video_key}_frame_{frame_idx}_obj_{object_id}.png'
                        plt.imsave(vis_path, rgb_image)
        
        return object_masks
    
    def _create_color_palette(self):
        """Create a color palette for object visualization."""
        cmap1 = plt.cm.tab20
        cmap2 = plt.cm.tab20b
        cmap3 = plt.cm.tab20c
        
        colors1 = torch.tensor([cmap1(i)[:3] for i in range(20)]).float().to(self.device)
        colors2 = torch.tensor([cmap2(i)[:3] for i in range(20)]).float().to(self.device)
        colors3 = torch.tensor([cmap3(i)[:3] for i in range(20)]).float().to(self.device)
        
        return torch.cat([colors1, colors2, colors3], dim=0)
    
    def _save_visualizations(self, point_map, pc1_results, pc2_results, images,
                            object_results, unit_ids, img_1_length, output_dir):
        """Save point cloud visualizations."""
        # Cost map visualizations
        total_costs = torch.cat([pc1_results['cost_maps'], pc2_results['cost_maps']])
        cost_vis = apply_colormap(total_costs).reshape(len(point_map), -1, 3)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_map.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(cost_vis.reshape(-1, 3))
        o3d.io.write_point_cloud(str(output_dir / 'cost_map_merged.ply'), pcd)

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(point_map[:img_1_length].reshape(-1, 3))
        pcd1.colors = o3d.utility.Vector3dVector(cost_vis[:img_1_length].reshape(-1, 3))
        o3d.io.write_point_cloud(str(output_dir / 'cost_map_1.ply'), pcd1)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(point_map[img_1_length:].reshape(-1, 3))
        pcd2.colors = o3d.utility.Vector3dVector(cost_vis[img_1_length:].reshape(-1, 3))
        o3d.io.write_point_cloud(str(output_dir / 'cost_map_2.ply'), pcd2)
        
        # Object instance visualizations
        colors = self._create_color_palette()
        self._save_object_point_clouds(
            point_map[:img_1_length].reshape(-1, 3),
            point_map[img_1_length:].reshape(-1, 3),
            object_results['object_id_list_1'],
            object_results['object_id_list_2'],
            object_results['mask_index_1'],
            object_results['mask_index_2'],
            images[0][:img_1_length],
            images[0][img_1_length:],
            unit_ids,
            colors, output_dir
        )
    
    def _save_object_point_clouds(self, pc1_points, pc2_points, object_id_list_1, object_id_list_2,
                                  mask_index_1, mask_index_2, images_1, images_2, unit_ids, colors, output_dir):
        """Save point clouds colored by object ID."""
        vis_pc_1 = images_1.permute(0, 2, 3, 1).reshape(-1, 3)
        vis_pc_2 = images_2.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # Color video 1 objects
        for obj_id in object_id_list_1.unique():
            obj_indices = torch.where(object_id_list_1 == obj_id)[0]
            unified_id = unit_ids['video_1'][obj_id.item()]
            color = colors[unified_id % colors.shape[0]]
            
            for single_idx in obj_indices:
                mask = mask_index_1 == single_idx
                vis_pc_1[mask.view(-1)] = color
        
        # Color video 2 objects
        for obj_id in object_id_list_2.unique():
            obj_indices = torch.where(object_id_list_2 == obj_id)[0]
            unified_id = unit_ids['video_2'][obj_id.item()]
            color = colors[unified_id % colors.shape[0]]
            
            for single_idx in obj_indices:
                mask = mask_index_2 == single_idx
                vis_pc_2[mask.view(-1)] = color
        
        # Save point clouds
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1_points)
        pcd1.colors = o3d.utility.Vector3dVector(vis_pc_1.cpu().numpy())
        o3d.io.write_point_cloud(str(output_dir / 'object_instances_1.ply'), pcd1)
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2_points)
        pcd2.colors = o3d.utility.Vector3dVector(vis_pc_2.cpu().numpy())
        o3d.io.write_point_cloud(str(output_dir / 'object_instances_2.ply'), pcd2)
    
    def _save_ground_truth_visualization(self, scene_name, images, img_1_length, H, W, output_dir):
        """Save ground truth object visualizations."""
        gt_dir = Path(self.config['dataset']['gt_dir'])
        gt_file_path = gt_dir / scene_name / 'segments.pkl'
        resample_rate = self.config['dataset']['resample_rate']
        
        if not os.path.exists(gt_file_path):
            return
        
        gt = pickle.load(open(gt_file_path, 'rb'))
        colors = self._create_color_palette()
        
        # Similar implementation as the original save_ground_truth_visualization
        # ...abbreviated for brevity
    
    def _load_cached_cost_maps(self, output_dir, scene_name):
        """Load cached cost maps if available."""
        root_dir = Path(output_dir).parent
        
        for exp_dir in root_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            scene_dir = exp_dir / scene_name
            if not scene_dir.exists():
                continue
            
            cost_path = scene_dir / 'cost_maps.pkl'
            visible_path = scene_dir / 'visible_maps.pkl'
            
            if cost_path.exists() and visible_path.exists():
                with open(cost_path, 'rb') as f:
                    cost_maps = pickle.load(f)
                with open(visible_path, 'rb') as f:
                    visible_maps = pickle.load(f)
                print(f"Loaded cached cost maps from: {exp_dir.name}")
                return (cost_maps, visible_maps)
        
        return None
    
    def _save_cost_maps(self, output_dir, cost_maps, visible_maps):
        """Save cost maps to disk."""
        with open(output_dir / 'cost_maps.pkl', 'wb') as f:
            pickle.dump(cost_maps, f)
        with open(output_dir / 'visible_maps.pkl', 'wb') as f:
            pickle.dump(visible_maps, f)
    
    def cleanup(self):
        """Clean up all model resources."""
        self.geometry_model.cleanup()
        self.mask_model.cleanup()
        self.semantic_model.cleanup()
        torch.cuda.empty_cache()
