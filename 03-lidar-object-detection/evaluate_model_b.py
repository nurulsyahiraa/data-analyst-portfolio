"""
============================================================================
ENHANCED SPATIAL GNN - COMPREHENSIVE TESTING PROGRAM
============================================================================
Program untuk testing model Enhanced Spatial GNN yang telah di-train dengan
tambahan metrics: FLOPs, AP@0.5, dan Total Parameter

Model: enhanced_spatial_gnn_20250702_104150.pth
Features: 23 advanced spatial features
Architecture: GNN-FCN with hierarchical spatial graph

Metrics yang dihasilkan:
- Classification: Accuracy, Precision, Recall, F1-Score per class
- 3D Detection: IoU per class, Mean IoU, AP@0.5, mAP
- Regression: Translation Error, MAE, RMSE
- Performance: FPS, Inference Time, Latency
- Model Complexity: Total Parameters, Trainable Parameters, FLOPs, GFLOPs
============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from datetime import datetime
import time
import warnings
import random
from collections import Counter
import torch.nn.functional as F

# Optional imports for PyTorch Geometric
try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️  PyTorch Geometric not available. Using simplified GNN implementation.")

# Optional imports for 3D visualization
try:
    import open3d as o3d
    import pyvista as pv
    VISUALIZATION_3D_AVAILABLE = True
except ImportError:
    VISUALIZATION_3D_AVAILABLE = False
    print("⚠️  3D visualization libraries not available.")

# Optional imports for FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  thop library not available. Using manual FLOPs estimation.")

warnings.filterwarnings('ignore')

# ============================================================================
# 1. ADVANCED SPATIAL FEATURE EXTRACTOR (23 Features)
# ============================================================================

class AdvancedSpatialFeatureExtractor:
    """
    Ekstraksi 23 fitur spasial lanjutan dari data point cloud.
    
    Fitur yang diekstrak:
    - Core Spatial (6): posisi normalisasi [x,y,z] + dimensi normalisasi [w,h,d]
    - Advanced Geometric (8): volume_log, aspect ratios, height_to_base, compactness, dll
    - Scene Context (5): center_distance, quadrant encoding, edge distance, dll
    - Rotation & Scale (4): rotation encoding (cos, sin, cos_2), relative_scale
    """
    INPUT_DIM = 23
    
    def __init__(self):
        self.position_scaler = RobustScaler()
        self.dimension_scaler = RobustScaler()
        self.scene_context = None
        self.fitted = False
    
    def extract_advanced_spatial_features(self, data_point, scene_bounds=None):
        """Ekstrak 23 fitur spasial dari satu data point."""
        position = np.array(data_point[:3], dtype=np.float32)
        dimensions = np.array(data_point[3:6], dtype=np.float32)
        rotation_z = float(data_point[6])
        
        # Pastikan dimensi positif
        dimensions = np.abs(dimensions)
        dimensions = np.maximum(dimensions, 1e-6)
        
        # === Core spatial features (6) ===
        if self.fitted:
            pos_normalized = self.position_scaler.transform([position])[0]
            dim_normalized = self.dimension_scaler.transform([dimensions])[0]
        else:
            pos_normalized = position.copy()
            dim_normalized = dimensions.copy()
        
        # === Advanced geometric features (8) ===
        volume_log = np.log(np.prod(dimensions) + 1e-6)
        max_dim = np.max(dimensions)
        min_dim = np.min(dimensions)
        mid_dim = np.median(dimensions)
        
        primary_aspect_ratio = max_dim / (min_dim + 1e-6)
        secondary_aspect_ratio = max_dim / (mid_dim + 1e-6)
        
        base_area = dimensions[0] * dimensions[1]
        height_to_base_ratio = dimensions[2] / (np.sqrt(base_area) + 1e-6)
        
        volume_sphere = (4/3) * np.pi * ((max_dim/2) ** 3)
        spatial_compactness = np.prod(dimensions) / (volume_sphere + 1e-6)
        
        height_above_ground = position[2] + dimensions[2]/2
        spatial_footprint = np.sqrt(base_area)
        
        # Surface area to volume ratio
        surface_area = 2 * (dimensions[0]*dimensions[1] + dimensions[1]*dimensions[2] + dimensions[0]*dimensions[2])
        surface_to_volume_ratio = surface_area / (np.prod(dimensions) + 1e-6)
        
        # === Scene-relative context features (5) ===
        if scene_bounds is not None:
            scene_center = (scene_bounds['max'] + scene_bounds['min']) / 2
            scene_size = scene_bounds['max'] - scene_bounds['min']
            scene_diagonal = np.linalg.norm(scene_size)
            
            center_distance_normalized = np.linalg.norm(position - scene_center) / (scene_diagonal + 1e-6)
            rel_pos = (position - scene_bounds['min']) / (scene_size + 1e-6)
            
            spatial_quadrant_x = 1.0 if rel_pos[0] > 0.5 else 0.0
            spatial_quadrant_y = 1.0 if rel_pos[1] > 0.5 else 0.0
            
            edge_distances = np.array([rel_pos[0], 1-rel_pos[0], rel_pos[1], 1-rel_pos[1]])
            min_edge_distance = np.min(edge_distances)
            
            normalized_x = rel_pos[0]
        else:
            center_distance_normalized = 0.5
            spatial_quadrant_x = 0.5
            spatial_quadrant_y = 0.5
            min_edge_distance = 0.5
            normalized_x = 0.5
        
        # === Rotation and scale features (4) ===
        rotation_cos = np.cos(rotation_z)
        rotation_sin = np.sin(rotation_z)
        rotation_cos_2 = np.cos(2 * rotation_z)
        
        if scene_bounds is not None:
            relative_scale = spatial_footprint / (np.linalg.norm(scene_size[:2]) + 1e-6)
        else:
            relative_scale = 0.1
        
        # Gabungkan semua 23 fitur
        spatial_features = np.concatenate([
            pos_normalized,                                                          # [3]
            dim_normalized,                                                          # [3]
            [volume_log, primary_aspect_ratio, secondary_aspect_ratio,             # [8]
             height_to_base_ratio, spatial_compactness, height_above_ground,
             spatial_footprint, surface_to_volume_ratio],
            [center_distance_normalized, spatial_quadrant_x, spatial_quadrant_y,    # [5]
             min_edge_distance, normalized_x],
            [rotation_cos, rotation_sin, rotation_cos_2, relative_scale]             # [4]
        ])
        
        assert len(spatial_features) == self.INPUT_DIM
        return spatial_features.astype(np.float32)
    
    def fit_scalers_with_spatial_context(self, all_data):
        """Fit scalers menggunakan data training."""
        positions = []
        dimensions = []
        
        for data_point in all_data:
            positions.append(data_point[:3])
            dims = np.abs(data_point[3:6])
            dims = np.maximum(dims, 1e-6)
            dimensions.append(dims)
        
        positions = np.array(positions)
        dimensions = np.array(dimensions)
        
        self.position_scaler.fit(positions)
        self.dimension_scaler.fit(dimensions)
        
        self.scene_context = {
            'bounds': {
                'min': np.min(positions, axis=0),
                'max': np.max(positions, axis=0)
            }
        }
        
        self.fitted = True
        print(f"✓ Feature extractor fitted with scene bounds")

# ============================================================================
# 2. ENHANCED HIERARCHICAL SPATIAL GRAPH
# ============================================================================

class EnhancedHierarchicalSpatialGraph:
    """
    Konstruktor graph spasial hierarkis dengan koneksi local dan global.
    """
    EDGE_DIM = 14
    
    def __init__(self, local_radius=5.0, global_radius=20.0, 
                 min_neighbors=3, max_local_neighbors=12, max_global_neighbors=6):
        self.local_radius = local_radius
        self.global_radius = global_radius
        self.min_neighbors = min_neighbors
        self.max_local_neighbors = max_local_neighbors
        self.max_global_neighbors = max_global_neighbors
    
    def compute_enhanced_spatial_edge_features(self, pos_i, pos_j, is_global=False):
        """Hitung fitur edge antara dua node."""
        rel_pos = pos_j - pos_i
        distance = torch.norm(rel_pos, p=2)
        device = pos_i.device
        
        normalized_rel_pos = rel_pos / (distance + 1e-6)
        local_distance_normalized = distance / (self.local_radius + 1e-6)
        
        angle_xy = torch.atan2(rel_pos[1], rel_pos[0])
        angle_xy_cos = torch.cos(angle_xy)
        angle_xy_sin = torch.sin(angle_xy)
        
        horizontal_distance = torch.norm(rel_pos[:2]) + 1e-6
        angle_z = torch.atan2(rel_pos[2], horizontal_distance)
        angle_z_cos = torch.cos(angle_z)
        
        if distance < self.local_radius * 0.3:
            spatial_relationship_type = 0.0
        elif distance < self.local_radius * 0.7:
            spatial_relationship_type = 0.5
        else:
            spatial_relationship_type = 1.0
        
        edge_type_encoding = 1.0 if is_global else 0.0
        
        edge_features = torch.cat([
            rel_pos,
            normalized_rel_pos,
            torch.tensor([distance, local_distance_normalized], device=device, dtype=torch.float32),
            torch.tensor([angle_xy_cos, angle_xy_sin], device=device, dtype=torch.float32),
            torch.tensor([angle_z, angle_z_cos], device=device, dtype=torch.float32),
            torch.tensor([spatial_relationship_type, edge_type_encoding], device=device, dtype=torch.float32)
        ])
        
        return edge_features
    
    def build_enhanced_hierarchical_graph(self, batch_positions, batch_indices):
        """Bangun graph hierarkis dengan koneksi local dan global."""
        edge_indices = []
        edge_attrs = []
        device = batch_positions.device
        
        unique_batches = torch.unique(batch_indices)
        
        for batch_id in unique_batches:
            mask = batch_indices == batch_id
            positions = batch_positions[mask]
            batch_size = positions.size(0)
            
            if batch_size < 2:
                if batch_size == 1:
                    global_idx = torch.where(mask)[0][0].item()
                    edge_indices.append([global_idx, global_idx])
                    edge_attrs.append(torch.zeros(self.EDGE_DIM, device=device))
                continue
            
            distances = torch.cdist(positions, positions, p=2)
            edge_set = set()
            
            for i in range(batch_size):
                local_neighbor_mask = (distances[i] <= self.local_radius) & (distances[i] > 0)
                local_neighbors = torch.where(local_neighbor_mask)[0]
                
                if len(local_neighbors) < self.min_neighbors and batch_size > 1:
                    _, closest_indices = torch.topk(distances[i], 
                                                  min(self.min_neighbors + 1, batch_size), 
                                                  largest=False)
                    local_neighbors = closest_indices[1:]
                
                if len(local_neighbors) > self.max_local_neighbors:
                    neighbor_distances = distances[i][local_neighbors]
                    _, keep_indices = torch.topk(neighbor_distances, 
                                                self.max_local_neighbors, 
                                                largest=False)
                    local_neighbors = local_neighbors[keep_indices]
                
                for j in local_neighbors:
                    global_i = torch.where(mask)[0][i].item()
                    global_j = torch.where(mask)[0][j].item()
                    
                    if (global_i, global_j) not in edge_set:
                        edge_set.add((global_i, global_j))
                        edge_set.add((global_j, global_i))
                        
                        edge_features = self.compute_enhanced_spatial_edge_features(
                            positions[i], positions[j], is_global=False)
                        
                        edge_indices.extend([[global_i, global_j], [global_j, global_i]])
                        edge_attrs.extend([edge_features, -edge_features])
            
            for i in range(batch_size):
                global_neighbor_mask = ((distances[i] > self.local_radius) & 
                                      (distances[i] <= self.global_radius))
                global_neighbors = torch.where(global_neighbor_mask)[0]
                
                if len(global_neighbors) > self.max_global_neighbors:
                    sample_indices = torch.randperm(len(global_neighbors))[:self.max_global_neighbors]
                    global_neighbors = global_neighbors[sample_indices]
                
                for j in global_neighbors:
                    global_i = torch.where(mask)[0][i].item()
                    global_j = torch.where(mask)[0][j].item()
                    
                    if (global_i, global_j) not in edge_set:
                        edge_set.add((global_i, global_j))
                        edge_set.add((global_j, global_i))
                        
                        edge_features = self.compute_enhanced_spatial_edge_features(
                            positions[i], positions[j], is_global=True)
                        
                        edge_indices.extend([[global_i, global_j], [global_j, global_i]])
                        edge_attrs.extend([edge_features, -edge_features])
        
        if len(edge_indices) == 0:
            num_nodes = len(batch_positions)
            if num_nodes > 0:
                edge_indices = [[0, 0]]
                edge_attrs = [torch.zeros(self.EDGE_DIM, device=device)]
        
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t().contiguous()
            edge_attr = torch.stack(edge_attrs)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, self.EDGE_DIM), device=device)
        
        return edge_index, edge_attr

# ============================================================================
# 3. ENHANCED SPATIAL ATTENTION
# ============================================================================

class EnhancedSpatialAttention(nn.Module):
    """Modul attention spasial dengan dual-branch attention."""
    def __init__(self, hidden_dim):
        super(EnhancedSpatialAttention, self).__init__()
        
        self.local_spatial_attention = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.global_spatial_attention = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention_fusion = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features, positions, batch):
        positions = positions.to(node_features.device)
        spatial_feature_context = torch.cat([node_features, positions], dim=1)
        
        local_attention_weights = self.local_spatial_attention(spatial_feature_context)
        global_attention_weights = self.global_spatial_attention(spatial_feature_context)
        
        attention_combination = torch.cat([local_attention_weights, global_attention_weights], dim=1)
        final_spatial_attention = self.spatial_attention_fusion(attention_combination)
        
        spatially_attended_features = node_features * final_spatial_attention
        
        if TORCH_GEOMETRIC_AVAILABLE and len(torch.unique(batch)) > 1:
            local_spatial_context = global_mean_pool(spatially_attended_features, batch)
            global_spatial_context = global_max_pool(spatially_attended_features, batch)
            combined_spatial_context = (local_spatial_context + global_spatial_context) / 2
            expanded_spatial_context = combined_spatial_context[batch]
        else:
            local_spatial_context = spatially_attended_features.mean(dim=0, keepdim=True)
            global_spatial_context = spatially_attended_features.max(dim=0, keepdim=True)[0]
            combined_spatial_context = (local_spatial_context + global_spatial_context) / 2
            expanded_spatial_context = combined_spatial_context.expand(node_features.size(0), -1)
        
        return expanded_spatial_context

# ============================================================================
# 4. ENHANCED SPATIAL GNN MODEL
# ============================================================================

if TORCH_GEOMETRIC_AVAILABLE:
    class EnhancedSpatialGNNModel(nn.Module):
        """Model GNN spasial dengan 23 input features dan 5 GNN layers."""
        def __init__(self, input_dim=None, hidden_dim=256, num_classes=6, dropout=0.12):
            super(EnhancedSpatialGNNModel, self).__init__()
            
            if input_dim is None:
                input_dim = AdvancedSpatialFeatureExtractor.INPUT_DIM
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            
            self.spatial_feature_embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            
            self.spatial_gnn_layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(5)
            ])
            
            self.spatial_layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(5)
            ])
            
            self.spatial_dropout = nn.Dropout(dropout)
            self.enhanced_spatial_attention = EnhancedSpatialAttention(hidden_dim)
            
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            
            self.spatial_regression_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 3),
                nn.LayerNorm(hidden_dim * 3),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 7)
            )
            
            self.enhanced_hierarchical_graph = EnhancedHierarchicalSpatialGraph()
        
        def forward(self, x, batch=None):
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            spatial_positions = x[:, :3]
            edge_index, edge_attr = self.enhanced_hierarchical_graph.build_enhanced_hierarchical_graph(
                spatial_positions, batch)
            
            spatial_features = self.spatial_feature_embedding(x)
            
            for layer_idx, (gnn_layer, layer_norm) in enumerate(zip(self.spatial_gnn_layers, self.spatial_layer_norms)):
                new_spatial_features = gnn_layer(spatial_features, edge_index)
                new_spatial_features = layer_norm(new_spatial_features)
                new_spatial_features = F.relu(new_spatial_features)
                new_spatial_features = self.spatial_dropout(new_spatial_features)
                
                if layer_idx > 0:
                    spatial_features = spatial_features + new_spatial_features * 0.8
                else:
                    spatial_features = new_spatial_features
            
            global_spatial_context = self.enhanced_spatial_attention(
                spatial_features, spatial_positions, batch)
            
            combined_spatial_features = torch.cat([spatial_features, global_spatial_context], dim=1)
            
            classification_output = self.classification_head(combined_spatial_features)
            spatial_output = self.spatial_regression_head(combined_spatial_features)
            
            return classification_output, spatial_output

else:
    class EnhancedSpatialGNNModel(nn.Module):
        """Fallback model tanpa PyTorch Geometric."""
        def __init__(self, input_dim=None, hidden_dim=256, num_classes=6, dropout=0.12):
            super(EnhancedSpatialGNNModel, self).__init__()
            
            if input_dim is None:
                input_dim = AdvancedSpatialFeatureExtractor.INPUT_DIM
                
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            
            self.spatial_embedding = nn.Linear(input_dim, hidden_dim)
            
            self.spatial_reasoning_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 3),
                    nn.LayerNorm(hidden_dim * 3),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                ) for _ in range(5)
            ])
            
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            
            self.spatial_regression_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 3),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, 7)
            )
        
        def forward(self, x, batch=None):
            spatial_features = self.spatial_embedding(x)
            
            for spatial_layer in self.spatial_reasoning_layers:
                residual = spatial_features
                spatial_features = spatial_layer(spatial_features)
                spatial_features = spatial_features + residual * 0.8
            
            classification_output = self.classification_head(spatial_features)
            spatial_output = self.spatial_regression_head(spatial_features)
            
            return classification_output, spatial_output

# ============================================================================
# 5. ENHANCED SPATIAL DATASET
# ============================================================================

class EnhancedSpatialDataset(Dataset):
    """Dataset untuk Enhanced Spatial GNN dengan 23 fitur."""
    def __init__(self, json_folder, feature_extractor=None, bbox_scalers=None, is_training=True):
        self.json_folder = json_folder
        self.is_training = is_training
        
        self.raw_data, self.labels, self.bbox_targets, self.file_info = self._load_spatial_data()
        
        if feature_extractor is None:
            self.feature_extractor = AdvancedSpatialFeatureExtractor()
            if is_training:
                self.feature_extractor.fit_scalers_with_spatial_context(self.raw_data)
        else:
            self.feature_extractor = feature_extractor
        
        self.spatial_features = []
        scene_bounds = self.feature_extractor.scene_context['bounds'] if self.feature_extractor.scene_context else None
        
        for data_point in self.raw_data:
            spatial_features = self.feature_extractor.extract_advanced_spatial_features(
                data_point, scene_bounds=scene_bounds)
            self.spatial_features.append(spatial_features)
        
        self.spatial_features = np.array(self.spatial_features, dtype=np.float32)
        
        if is_training:
            self.bbox_scaler_pos = RobustScaler()
            self.bbox_scaler_dim = RobustScaler()
            self.bbox_scaler_rot = StandardScaler()
            
            bbox_positions = self.bbox_targets[:, :3]
            bbox_dimensions = np.abs(self.bbox_targets[:, 3:6])
            bbox_dimensions = np.maximum(bbox_dimensions, 1e-6)
            bbox_rotations = self.bbox_targets[:, 6:7]
            
            scaled_positions = self.bbox_scaler_pos.fit_transform(bbox_positions)
            scaled_dimensions = self.bbox_scaler_dim.fit_transform(bbox_dimensions)
            scaled_rotations = self.bbox_scaler_rot.fit_transform(bbox_rotations)
            
            self.bbox_targets = np.concatenate([scaled_positions, scaled_dimensions, scaled_rotations], axis=1)
            
            self.bbox_scalers = {
                'position': self.bbox_scaler_pos,
                'dimension': self.bbox_scaler_dim,
                'rotation': self.bbox_scaler_rot
            }
        elif bbox_scalers is not None:
            self.bbox_scalers = bbox_scalers
            
            bbox_positions = self.bbox_targets[:, :3]
            bbox_dimensions = np.abs(self.bbox_targets[:, 3:6])
            bbox_dimensions = np.maximum(bbox_dimensions, 1e-6)
            bbox_rotations = self.bbox_targets[:, 6:7]
            
            scaled_positions = self.bbox_scalers['position'].transform(bbox_positions)
            scaled_dimensions = self.bbox_scalers['dimension'].transform(bbox_dimensions)
            scaled_rotations = self.bbox_scalers['rotation'].transform(bbox_rotations)
            
            self.bbox_targets = np.concatenate([scaled_positions, scaled_dimensions, scaled_rotations], axis=1)
        
        self.label_map = {"cyclist": 0, "car": 1, "human": 2, "wall": 3, "tree": 4, "cart": 5}
        self.classes = list(self.label_map.keys())
        
        print(f"✓ Enhanced Spatial Dataset: {len(self.spatial_features)} samples with 23 features")
        print(f"✓ Class distribution: {Counter(self.labels)}")
    
    def _load_spatial_data(self):
        """Load data dari folder JSON."""
        data, labels, bbox_targets, file_info = [], [], [], []
        label_map = {"cyclist": 0, "car": 1, "human": 2, "wall": 3, "tree": 4, "cart": 5}
        
        for file_name in os.listdir(self.json_folder):
            if not file_name.endswith(".json"):
                continue
                
            file_path = os.path.join(self.json_folder, file_name)
            try:
                with open(file_path, "r") as f:
                    json_data = json.load(f)
                    
                for item in json_data.get("figures", []):
                    obj_id = item.get("objectKey")
                    label = None
                    
                    for obj in json_data.get("objects", []):
                        if obj.get("key") == obj_id:
                            label = obj.get("classTitle")
                            break
                    
                    if label in label_map:
                        geometry = item.get("geometry", {})
                        position = geometry.get("position", {})
                        
                        dimensions = geometry.get("size", geometry.get("dimensions", {"x": 1.0, "y": 1.0, "z": 1.0}))
                        if isinstance(dimensions, dict):
                            dimensions = [dimensions.get(k, 1.0) for k in ["x", "y", "z"]]
                        elif not isinstance(dimensions, (list, np.ndarray)) or len(dimensions) != 3:
                            dimensions = [1.0, 1.0, 1.0]
                            
                        rotation_z = geometry.get("rotation", {}).get("z", 0.0)
                        spatial_position = list(position.values())
                        
                        if len(spatial_position) == 3:
                            complete_spatial_features = spatial_position + dimensions + [rotation_z]
                            data.append(complete_spatial_features)
                            labels.append(label_map[label])
                            bbox_targets.append(complete_spatial_features)
                            file_info.append({
                                "file": file_name,
                                "spatial_coords": spatial_position,
                                "spatial_dimensions": dimensions,
                                "spatial_rotation_z": rotation_z
                            })
                            
            except Exception as e:
                print(f"Warning: Error loading {file_name}: {e}")
                continue
        
        return (np.array(data, dtype=np.float32), 
                np.array(labels, dtype=np.int64),
                np.array(bbox_targets, dtype=np.float32), 
                file_info)
    
    def inverse_transform_spatial_bbox(self, scaled_bbox):
        """Denormalize bounding box prediction."""
        scaled_position = scaled_bbox[:3]
        scaled_dimensions = scaled_bbox[3:6] 
        scaled_rotation = scaled_bbox[6:7]
        
        original_position = self.bbox_scalers['position'].inverse_transform([scaled_position])[0]
        original_dimensions = self.bbox_scalers['dimension'].inverse_transform([scaled_dimensions])[0]
        original_rotation = self.bbox_scalers['rotation'].inverse_transform([scaled_rotation])[0]
        
        return np.concatenate([original_position, original_dimensions, original_rotation])
    
    def __len__(self):
        return len(self.spatial_features)
    
    def __getitem__(self, idx):
        spatial_features = torch.FloatTensor(self.spatial_features[idx])
        class_target = torch.LongTensor([self.labels[idx]])[0]
        spatial_bbox_target = torch.FloatTensor(self.bbox_targets[idx])
        
        original_spatial_coords = torch.FloatTensor(self.file_info[idx]["spatial_coords"])
        original_spatial_dims = torch.FloatTensor(self.file_info[idx]["spatial_dimensions"])
        spatial_rotation_z = self.file_info[idx]["spatial_rotation_z"]
        
        return (spatial_features, class_target, spatial_bbox_target, 
                original_spatial_coords, original_spatial_dims, spatial_rotation_z)

# ============================================================================
# 6. MODEL COMPLEXITY ANALYZER (FLOPs, Parameters)
# ============================================================================

class ModelComplexityAnalyzer:
    """Analyzer untuk menghitung kompleksitas model: Parameters, FLOPs, Memory."""
    def __init__(self):
        self.use_thop = THOP_AVAILABLE
    
    def count_parameters(self, model):
        """Hitung total dan trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params_millions': total_params / 1e6,
            'trainable_params_millions': trainable_params / 1e6
        }
    
    def estimate_flops_manual(self, model, input_dim=23, hidden_dim=256):
        """Estimasi FLOPs secara manual berdasarkan arsitektur model."""
        total_flops = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                flops = 2 * module.in_features * module.out_features
                total_flops += flops
            elif isinstance(module, nn.LayerNorm):
                flops = 5 * module.normalized_shape[0]
                total_flops += flops
        
        if TORCH_GEOMETRIC_AVAILABLE:
            num_gnn_layers = 5
            avg_edges = 20
            gnn_flops_per_layer = avg_edges * hidden_dim * hidden_dim * 2
            total_flops += num_gnn_layers * gnn_flops_per_layer
        
        return total_flops
    
    def analyze_model_complexity(self, model, sample_input, device):
        """Analisis lengkap kompleksitas model."""
        model.eval()
        
        param_stats = self.count_parameters(model)
        
        if self.use_thop:
            try:
                sample_input_device = sample_input[:1].to(device)
                
                class ModelWrapper(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    def forward(self, x):
                        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                        return self.model(x, batch)
                
                wrapper = ModelWrapper(model)
                flops, params = profile(wrapper, inputs=(sample_input_device,), verbose=False)
                flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
                
                flops_stats = {
                    'total_flops': flops,
                    'gflops': flops / 1e9,
                    'mflops': flops / 1e6,
                    'flops_formatted': flops_formatted
                }
            except Exception as e:
                print(f"⚠️  thop profiling failed: {e}. Using manual estimation.")
                manual_flops = self.estimate_flops_manual(model)
                flops_stats = {
                    'total_flops': manual_flops,
                    'gflops': manual_flops / 1e9,
                    'mflops': manual_flops / 1e6,
                    'flops_formatted': f"{manual_flops/1e6:.3f}M (estimated)"
                }
        else:
            manual_flops = self.estimate_flops_manual(model)
            flops_stats = {
                'total_flops': manual_flops,
                'gflops': manual_flops / 1e9,
                'mflops': manual_flops / 1e6,
                'flops_formatted': f"{manual_flops/1e6:.3f}M (estimated)"
            }
        
        model_memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_memory_mb = model_memory_bytes / (1024 ** 2)
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                sample_input_device = sample_input[:1].to(device)
                batch = torch.zeros(1, dtype=torch.long, device=device)
                _ = model(sample_input_device, batch)
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory_mb = model_memory_mb
        
        return {
            'parameters': param_stats,
            'flops': flops_stats,
            'memory': {
                'model_size_mb': model_memory_mb,
                'peak_memory_mb': peak_memory_mb
            }
        }

# ============================================================================
# 7. 3D IoU CALCULATION
# ============================================================================

def calculate_3d_iou_with_rotation(box1, box2):
    """Hitung 3D IoU dengan mempertimbangkan rotasi."""
    try:
        pos1, dim1, rot1 = np.array(box1[:3]), np.abs(np.array(box1[3:6])), box1[6]
        pos2, dim2, rot2 = np.array(box2[:3]), np.abs(np.array(box2[3:6])), box2[6]
        
        if np.any(dim1 <= 0) or np.any(dim2 <= 0):
            return 0.0
        
        def get_oriented_corners(center, dimensions, rotation_z):
            half_dims = dimensions / 2
            corners = np.array([
                [-half_dims[0], -half_dims[1], -half_dims[2]],
                [ half_dims[0], -half_dims[1], -half_dims[2]],
                [-half_dims[0],  half_dims[1], -half_dims[2]],
                [ half_dims[0],  half_dims[1], -half_dims[2]],
                [-half_dims[0], -half_dims[1],  half_dims[2]],
                [ half_dims[0], -half_dims[1],  half_dims[2]],
                [-half_dims[0],  half_dims[1],  half_dims[2]],
                [ half_dims[0],  half_dims[1],  half_dims[2]]
            ])
            
            cos_r, sin_r = np.cos(rotation_z), np.sin(rotation_z)
            rot_matrix = np.array([
                [cos_r, -sin_r, 0],
                [sin_r,  cos_r, 0], 
                [0,      0,     1]
            ])
            
            return corners @ rot_matrix.T + center
        
        corners1 = get_oriented_corners(pos1, dim1, rot1)
        corners2 = get_oriented_corners(pos2, dim2, rot2)
        
        min1, max1 = np.min(corners1, axis=0), np.max(corners1, axis=0)
        min2, max2 = np.min(corners2, axis=0), np.max(corners2, axis=0)
        
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        
        if np.any(inter_min >= inter_max):
            return 0.0
        
        inter_dims = inter_max - inter_min
        inter_volume = np.prod(inter_dims)
        
        vol1 = np.prod(dim1)
        vol2 = np.prod(dim2)
        union_volume = vol1 + vol2 - inter_volume
        
        if union_volume <= 1e-6:
            return 0.0
        
        iou = inter_volume / union_volume
        return max(0.0, min(1.0, iou))
        
    except Exception:
        return 0.0

def calculate_translation_error(pred_coords, gt_coords):
    """Hitung Euclidean distance antara predicted dan ground truth."""
    return np.linalg.norm(np.array(pred_coords) - np.array(gt_coords))

# ============================================================================
# 8. COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_model_comprehensive(model, test_loader, device, test_dataset, classes, complexity_analyzer):
    """Evaluasi komprehensif model dengan semua metrics."""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_class_predictions = []
    all_class_targets = []
    all_ious = []
    all_translation_errors = []
    all_bbox_predictions = []
    all_bbox_ground_truths = []
    
    total_inference_time = 0.0
    latencies = []
    
    print("🔍 Starting comprehensive model evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            features, class_targets, bbox_targets, orig_coords, orig_dims, rotation_z = batch_data
            
            features = features.to(device)
            class_targets = class_targets.to(device)
            
            batch_indices = torch.arange(len(features), device=device)
            
            start_time = time.time()
            class_output, bbox_output = model(features, batch_indices)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_inference_time += batch_time
            latencies.extend([batch_time * 1000 / len(features)] * len(features))
            
            class_preds = torch.argmax(class_output, dim=1)
            bbox_preds_cpu = bbox_output.cpu().numpy()
            
            for i in range(len(features)):
                pred_box = test_dataset.inverse_transform_spatial_bbox(bbox_preds_cpu[i])
                
                gt_coords = orig_coords[i].numpy()
                gt_dims = orig_dims[i].numpy()
                gt_rot = rotation_z[i] if isinstance(rotation_z[i], (int, float)) else rotation_z[i].item()
                gt_box = np.concatenate([gt_coords, gt_dims, [gt_rot]])
                
                iou = calculate_3d_iou_with_rotation(pred_box, gt_box)
                all_ious.append(iou)
                
                translation_error = calculate_translation_error(pred_box[:3], gt_box[:3])
                all_translation_errors.append(translation_error)
                
                all_predictions.append({
                    'class_id': class_preds[i].item(),
                    'class_name': classes[class_preds[i].item()],
                    'box': pred_box,
                    'confidence': torch.softmax(class_output[i], dim=0).max().item(),
                    'iou': iou,
                    'translation_error': translation_error
                })
                
                all_ground_truths.append({
                    'class_id': class_targets[i].item(),
                    'class_name': classes[class_targets[i].item()],
                    'box': gt_box
                })
                
                all_class_predictions.append(class_preds[i].item())
                all_class_targets.append(class_targets[i].item())
                
                all_bbox_predictions.append(pred_box)
                all_bbox_ground_truths.append(gt_box)
    
    print("📊 Calculating comprehensive metrics...")
    
    # Classification Metrics
    accuracy = accuracy_score(all_class_targets, all_class_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_class_targets, all_class_predictions, average=None, zero_division=0
    )
    
    per_class_metrics = []
    for i, class_name in enumerate(classes):
        per_class_metrics.append({
            'Class': class_name,
            'Precision': precision[i] if i < len(precision) else 0.0,
            'Recall': recall[i] if i < len(recall) else 0.0,
            'F1_Score': f1[i] if i < len(f1) else 0.0,
            'Support': support[i] if i < len(support) else 0
        })
    
    cm = confusion_matrix(all_class_targets, all_class_predictions)
    
    # 3D Detection Metrics
    mean_iou = np.mean(all_ious)
    mean_translation_error = np.mean(all_translation_errors)
    
    per_class_iou = {}
    per_class_ap_05 = {}
    
    for class_id, class_name in enumerate(classes):
        class_ious = [all_ious[i] for i, pred in enumerate(all_predictions) 
                     if all_ground_truths[i]['class_id'] == class_id]
        per_class_iou[class_name] = np.mean(class_ious) if class_ious else 0.0
        
        class_preds = [pred for pred in all_predictions if pred['class_id'] == class_id]
        class_gts = [gt for gt in all_ground_truths if gt['class_id'] == class_id]
        
        if len(class_gts) == 0:
            per_class_ap_05[class_name] = 0.0
        else:
            sorted_preds = sorted(class_preds, key=lambda x: x['confidence'], reverse=True)
            tp = sum(1 for pred in sorted_preds if pred['iou'] >= 0.5)
            per_class_ap_05[class_name] = tp / len(class_gts)
    
    mAP_05 = np.mean(list(per_class_ap_05.values()))
    
    # Regression Metrics
    bbox_preds_array = np.array(all_bbox_predictions)
    bbox_gts_array = np.array(all_bbox_ground_truths)
    
    mae_overall = np.mean(np.abs(bbox_preds_array - bbox_gts_array))
    rmse_overall = np.sqrt(np.mean((bbox_preds_array - bbox_gts_array) ** 2))
    
    dimension_names = ['x', 'y', 'z', 'width', 'height', 'depth', 'rotation_z']
    mae_per_dim = {}
    rmse_per_dim = {}
    for i, dim_name in enumerate(dimension_names):
        mae_per_dim[dim_name] = np.mean(np.abs(bbox_preds_array[:, i] - bbox_gts_array[:, i]))
        rmse_per_dim[dim_name] = np.sqrt(np.mean((bbox_preds_array[:, i] - bbox_gts_array[:, i]) ** 2))
    
    # Performance Metrics
    num_samples = len(all_ious)
    avg_time_per_frame_ms = (total_inference_time / num_samples) * 1000 if num_samples > 0 else 0.0
    fps = num_samples / total_inference_time if total_inference_time > 0 else 0.0
    mean_latency_ms = np.mean(latencies) if latencies else 0.0
    
    # Model Complexity
    sample_input = next(iter(test_loader))[0][:1]
    complexity_results = complexity_analyzer.analyze_model_complexity(model, sample_input, device)
    
    return {
        'classification': {
            'accuracy': accuracy,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm,
            'class_names': classes
        },
        'detection_3d': {
            'mean_iou': mean_iou,
            'per_class_iou': per_class_iou,
            'mAP_05': mAP_05,
            'per_class_ap_05': per_class_ap_05,
            'mean_translation_error': mean_translation_error,
            'all_ious': all_ious,
            'all_translation_errors': all_translation_errors
        },
        'regression': {
            'mae_overall': mae_overall,
            'rmse_overall': rmse_overall,
            'mae_per_dim': mae_per_dim,
            'rmse_per_dim': rmse_per_dim
        },
        'performance': {
            'avg_time_per_frame_ms': avg_time_per_frame_ms,
            'inference_speed_fps': fps,
            'mean_latency_ms': mean_latency_ms,
            'total_inference_time_sec': total_inference_time,
            'total_samples': num_samples
        },
        'model_complexity': complexity_results,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths
    }

# ============================================================================
# 9. VISUALIZATION FUNCTIONS
# ============================================================================

def create_testing_visualizations(results, save_path):
    """Buat visualisasi hasil testing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = results['classification']['confusion_matrix']
    classes = results['classification']['class_names']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Enhanced Spatial GNN: Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'enhanced_spatial_gnn_confusion_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class Metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics_df = pd.DataFrame(results['classification']['per_class_metrics'])
    x = np.arange(len(metrics_df))
    width = 0.25
    
    axes[0].bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8, color='skyblue')
    axes[0].bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8, color='lightgreen')
    axes[0].bar(x + width, metrics_df['F1_Score'], width, label='F1-Score', alpha=0.8, color='salmon')
    
    axes[0].set_xlabel('Classes', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Classification Metrics per Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_df['Class'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)
    
    class_names = list(results['detection_3d']['per_class_iou'].keys())
    iou_values = list(results['detection_3d']['per_class_iou'].values())
    ap_values = list(results['detection_3d']['per_class_ap_05'].values())
    
    x = np.arange(len(class_names))
    axes[1].bar(x - 0.2, iou_values, 0.4, label='3D IoU', alpha=0.8, color='gold')
    axes[1].bar(x + 0.2, ap_values, 0.4, label='AP@0.5', alpha=0.8, color='lightcoral')
    
    axes[1].set_xlabel('Classes', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('3D Detection Metrics per Class (IoU & AP@0.5)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'enhanced_spatial_gnn_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model Complexity Summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    params = results['model_complexity']['parameters']
    param_labels = ['Total', 'Trainable', 'Non-trainable']
    param_values = [params['total_params']/1e6, params['trainable_params']/1e6, params['non_trainable_params']/1e6]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = axes[0].bar(param_labels, param_values, color=colors, alpha=0.8)
    axes[0].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[0].set_title('Model Parameters Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, param_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}M', ha='center', va='bottom', fontweight='bold')
    
    perf_labels = ['FPS', 'Latency (ms)', 'Time/Frame (ms)']
    perf_values = [
        results['performance']['inference_speed_fps'],
        results['performance']['mean_latency_ms'],
        results['performance']['avg_time_per_frame_ms']
    ]
    
    bars = axes[1].bar(perf_labels, perf_values, color=['#9b59b6', '#f39c12', '#1abc9c'], alpha=0.8)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, perf_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'enhanced_spatial_gnn_complexity_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved with timestamp: {timestamp}")
    return timestamp

def save_testing_results(results, save_path, timestamp, testing_time):
    """Save testing results ke multi-sheet Excel file."""
    excel_filename = f"enhanced_spatial_gnn_testing_results_{timestamp}.xlsx"
    excel_path = os.path.join(save_path, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Classification Metrics
        classification_df = pd.DataFrame(results['classification']['per_class_metrics'])
        classification_df.to_excel(writer, sheet_name='Classification_Metrics', index=False)
        
        # Confusion Matrix
        cm_df = pd.DataFrame(
            results['classification']['confusion_matrix'],
            index=results['classification']['class_names'],
            columns=results['classification']['class_names']
        )
        cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
        
        # 3D Detection Metrics
        detection_data = []
        for class_name in results['classification']['class_names']:
            detection_data.append({
                'Class': class_name,
                'IoU': results['detection_3d']['per_class_iou'][class_name],
                'AP@0.5': results['detection_3d']['per_class_ap_05'][class_name]
            })
        detection_df = pd.DataFrame(detection_data)
        detection_df.to_excel(writer, sheet_name='3D_Detection_Metrics', index=False)
        
        # Regression Metrics
        regression_data = []
        for dim_name in results['regression']['mae_per_dim'].keys():
            regression_data.append({
                'Dimension': dim_name,
                'MAE': results['regression']['mae_per_dim'][dim_name],
                'RMSE': results['regression']['rmse_per_dim'][dim_name]
            })
        regression_df = pd.DataFrame(regression_data)
        regression_df.to_excel(writer, sheet_name='Regression_Metrics', index=False)
        
        # Sample Results
        sample_results_df = pd.DataFrame({
            'Sample_ID': range(len(results['detection_3d']['all_translation_errors'])),
            'Translation_Error': results['detection_3d']['all_translation_errors'],
            'IoU': results['detection_3d']['all_ious']
        })
        sample_results_df.to_excel(writer, sheet_name='Sample_Results', index=False)
        
        # Model Complexity
        params = results['model_complexity']['parameters']
        flops = results['model_complexity']['flops']
        memory = results['model_complexity']['memory']
        
        complexity_data = {
            'Metric': [
                'Total_Parameters', 'Trainable_Parameters', 'Non_Trainable_Parameters',
                'Total_Parameters_Millions', 'Trainable_Parameters_Millions',
                'Total_FLOPs', 'GFLOPs', 'MFLOPs', 'FLOPs_Formatted',
                'Model_Size_MB', 'Peak_Memory_MB'
            ],
            'Value': [
                params['total_params'], params['trainable_params'], params['non_trainable_params'],
                params['total_params_millions'], params['trainable_params_millions'],
                flops['total_flops'], flops['gflops'], flops['mflops'], flops['flops_formatted'],
                memory['model_size_mb'], memory['peak_memory_mb']
            ]
        }
        complexity_df = pd.DataFrame(complexity_data)
        complexity_df.to_excel(writer, sheet_name='Model_Complexity', index=False)
        
        # Performance Metrics
        performance_data = {
            'Metric': [
                'Avg_Time_Per_Frame_ms', 'Inference_Speed_FPS', 'Mean_Latency_ms',
                'Total_Inference_Time_sec', 'Total_Test_Samples'
            ],
            'Value': [
                results['performance']['avg_time_per_frame_ms'],
                results['performance']['inference_speed_fps'],
                results['performance']['mean_latency_ms'],
                results['performance']['total_inference_time_sec'],
                results['performance']['total_samples']
            ]
        }
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
        
        # Summary
        summary_data = {
            'Metric': [
                'Model_Architecture', 'Feature_Engineering', 'Input_Dimension',
                'Model_Source', 'Testing_Date',
                'Overall_Accuracy', 'Mean_IoU', 'mAP@0.5', 'Mean_Translation_Error',
                'MAE_Overall', 'RMSE_Overall',
                'Total_Parameters', 'Trainable_Parameters', 'GFLOPs',
                'Avg_Time_Per_Frame_ms', 'FPS',
                'Testing_Time_Seconds', 'Testing_Time_Minutes', 'Total_Test_Samples'
            ],
            'Value': [
                'Enhanced Spatial GNN-FCN (5 GNN layers, 256 hidden)',
                '23 advanced spatial features',
                '23',
                'enhanced_spatial_gnn_20250702_104150.pth',
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                results['classification']['accuracy'],
                results['detection_3d']['mean_iou'],
                results['detection_3d']['mAP_05'],
                results['detection_3d']['mean_translation_error'],
                results['regression']['mae_overall'],
                results['regression']['rmse_overall'],
                f"{params['total_params']:,}",
                f"{params['trainable_params']:,}",
                f"{flops['gflops']:.4f}",
                f"{results['performance']['avg_time_per_frame_ms']:.4f}",
                f"{results['performance']['inference_speed_fps']:.2f}",
                f"{testing_time:.2f}",
                f"{testing_time/60:.2f}",
                results['performance']['total_samples']
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Testing_Summary', index=False)
    
    print(f"✅ Testing results saved: {excel_filename}")
    return excel_path

# ============================================================================
# 10. MAIN TESTING PROGRAM
# ============================================================================

def main():
    """Main program untuk Testing Enhanced Spatial GNN."""
    
    # === CONFIGURATION PATHS ===
    PATHS = {
        'test_json': "data/test",
        'pcd_base': "data/pcd_test",
        'model_path': "outputs/checkpoints/enhanced_spatial_gnn_20250702_104150.pth",
        'results_save': "outputs/results",
        'images_save': "outputs/figures"
    }
    
    for path in [PATHS['results_save'], PATHS['images_save']]:
        os.makedirs(path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    total_start_time = time.time()
    
    print("=" * 80)
    print("🚀 ENHANCED SPATIAL GNN - COMPREHENSIVE TESTING")
    print("=" * 80)
    print("📊 Metrics: Classification, 3D IoU, AP@0.5, mAP, FLOPs, Parameters")
    print("🎨 Visualizations: Confusion Matrix, Per-class Metrics, Complexity")
    print("💾 Output: Multi-sheet Excel + PNG Charts")
    print("=" * 80)
    
    try:
        # Load pre-trained model
        print("\n📥 Loading pre-trained model...")
        
        if not os.path.exists(PATHS['model_path']):
            print(f"❌ Model file not found: {PATHS['model_path']}")
            return
        
        checkpoint = torch.load(PATHS['model_path'], map_location=device)
        print(f"✅ Checkpoint loaded from: {PATHS['model_path']}")
        
        feature_extractor = checkpoint.get('feature_extractor')
        bbox_scalers = checkpoint.get('bbox_scalers')
        
        if feature_extractor is None or bbox_scalers is None:
            print("⚠️  Feature extractor or scalers not found in checkpoint")
            print("    Creating new ones from test data...")
            temp_dataset = EnhancedSpatialDataset(PATHS['test_json'], is_training=True)
            feature_extractor = temp_dataset.feature_extractor
            bbox_scalers = temp_dataset.bbox_scalers
        
        print("✅ Feature extractor and scalers ready")
        
        # Load test dataset
        print("\n📂 Loading test dataset...")
        test_dataset = EnhancedSpatialDataset(
            PATHS['test_json'], 
            feature_extractor=feature_extractor, 
            bbox_scalers=bbox_scalers,
            is_training=False
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        classes = ["cyclist", "car", "human", "wall", "tree", "cart"]
        
        # Initialize model
        print("\n🏗️  Initializing model architecture...")
        model = EnhancedSpatialGNNModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Model loaded and ready for testing")
        
        # Initialize complexity analyzer
        complexity_analyzer = ModelComplexityAnalyzer()
        
        # Testing phase
        print("\n🧪 Starting comprehensive testing...")
        testing_start_time = time.time()
        
        results = evaluate_model_comprehensive(
            model, test_loader, device, test_dataset, classes, complexity_analyzer
        )
        
        testing_time = time.time() - testing_start_time
        
        # Print results
        print("\n" + "=" * 80)
        print("📊 TESTING RESULTS")
        print("=" * 80)
        
        print("\n🎯 CLASSIFICATION METRICS:")
        print(f"   • Accuracy: {results['classification']['accuracy']:.4f} ({results['classification']['accuracy']*100:.2f}%)")
        
        print("\n🎯 3D DETECTION METRICS:")
        print(f"   • Mean IoU: {results['detection_3d']['mean_iou']:.4f}")
        print(f"   • mAP@0.5: {results['detection_3d']['mAP_05']:.4f}")
        print(f"   • Mean Translation Error: {results['detection_3d']['mean_translation_error']:.4f}")
        
        print("\n🎯 REGRESSION METRICS:")
        print(f"   • MAE Overall: {results['regression']['mae_overall']:.4f}")
        print(f"   • RMSE Overall: {results['regression']['rmse_overall']:.4f}")
        
        print("\n🎯 PERFORMANCE METRICS:")
        print(f"   • Avg Time per Frame: {results['performance']['avg_time_per_frame_ms']:.4f} ms")
        print(f"   • Inference Speed: {results['performance']['inference_speed_fps']:.2f} FPS")
        print(f"   • Testing Duration: {testing_time:.2f} sec ({testing_time/60:.2f} min)")
        
        print("\n🎯 MODEL COMPLEXITY:")
        params = results['model_complexity']['parameters']
        flops = results['model_complexity']['flops']
        print(f"   • Total Parameters: {params['total_params']:,} ({params['total_params_millions']:.3f}M)")
        print(f"   • Trainable Parameters: {params['trainable_params']:,} ({params['trainable_params_millions']:.3f}M)")
        print(f"   • FLOPs: {flops['flops_formatted']}")
        print(f"   • GFLOPs: {flops['gflops']:.4f}")
        
        print("\n📈 PER-CLASS SUMMARY:")
        print("-" * 60)
        print(f"{'Class':<12} {'IoU':>8} {'AP@0.5':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 60)
        for i, class_name in enumerate(classes):
            iou = results['detection_3d']['per_class_iou'][class_name]
            ap = results['detection_3d']['per_class_ap_05'][class_name]
            metrics = results['classification']['per_class_metrics'][i]
            print(f"{class_name:<12} {iou:>8.3f} {ap:>8.3f} {metrics['Precision']:>10.3f} {metrics['Recall']:>8.3f} {metrics['F1_Score']:>8.3f}")
        print("-" * 60)
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        timestamp = create_testing_visualizations(results, PATHS['images_save'])
        
        # Save results to Excel
        excel_path = save_testing_results(results, PATHS['results_save'], timestamp, testing_time)
        
        # Final summary
        total_time = time.time() - total_start_time
        
        print("\n" + "=" * 80)
        print("🎉 TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"✅ Excel Results: {excel_path}")
        print(f"✅ Visualizations: {PATHS['images_save']}")
        print(f"⏱️  Total Execution Time: {total_time:.2f} sec ({total_time/60:.2f} min)")
        
        if results['detection_3d']['mean_iou'] > 0.6:
            print("🏆 EXCELLENT! Mean IoU > 0.6 achieved!")
        elif results['detection_3d']['mean_iou'] > 0.5:
            print("🥈 VERY GOOD! Mean IoU > 0.5 achieved!")
        else:
            print("📊 Good results. Consider model improvements for higher IoU.")
        
        print("\n📋 KEY METRICS SUMMARY:")
        print(f"   • Accuracy: {results['classification']['accuracy']*100:.2f}%")
        print(f"   • Mean IoU: {results['detection_3d']['mean_iou']:.4f}")
        print(f"   • mAP@0.5: {results['detection_3d']['mAP_05']:.4f}")
        print(f"   • Total Parameters: {params['total_params']:,}")
        print(f"   • GFLOPs: {flops['gflops']:.4f}")
        print(f"   • FPS: {results['performance']['inference_speed_fps']:.2f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting tips:")
        print("   1. Verify model file exists at specified path")
        print("   2. Check test data paths are accessible")
        print("   3. Ensure GPU memory availability (if using CUDA)")
        print("   4. Verify data format consistency")
        print("   5. Check PyTorch and dependencies versions")

if __name__ == "__main__":
    main()