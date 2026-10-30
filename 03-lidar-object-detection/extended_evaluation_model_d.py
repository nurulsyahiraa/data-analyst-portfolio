"""
ENHANCED TESTING MODEL - Hybrid Transformer-GNN-FCN
====================================================
Enhancements dari original testing_model_5_fix.py:
1. FLOPs calculation - Mengukur computational complexity
2. Total model parameters - Trainable dan non-trainable
3. Multi-threshold AP - AP@0.5, AP@0.7, AP@0.9
4. Failure Case Analysis - Comprehensive error analysis

Author: Enhanced version
Date: 2025
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import time
import warnings
import random
from collections import Counter, defaultdict
import torch.nn.functional as F

# Optional 3D visualization libraries
try:
    import open3d as o3d
    import pyvista as pv
    VISUALIZATION_3D_AVAILABLE = True
except ImportError:
    VISUALIZATION_3D_AVAILABLE = False
    print("⚠️  3D visualization libraries not available. Will use basic visualization only.")

# Optional PyTorch Geometric for advanced GNN
try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️  PyTorch Geometric not available. Using simplified GNN implementation.")

# Optional thop for FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  thop library not available. Will use manual FLOPs estimation.")

warnings.filterwarnings('ignore')

# ============================================================================
# 1. UNIFIED DATASET - CONSISTENT WITH PROVEN TRANSFORMER APPROACH
# ============================================================================

class UnifiedHybridDataset(Dataset):
    """
    Dataset yang menggabungkan pendekatan terbaik dari kedua program:
    - Menggunakan 7 features proven dari Transformer model (sederhana & efektif)
    - Normalization strategy yang consistent
    - File handling yang robust
    """
    def __init__(self, json_folder, scaler=None, bbox_scaler=None, is_training=True):
        self.json_folder = json_folder
        self.is_training = is_training
        
        # Load data menggunakan proven approach dari Transformer model
        self.data, self.labels, self.bbox_targets, self.file_info = self._load_json_data()
        
        # Consistent label mapping
        self.label_map = {"cyclist": 0, "car": 1, "human": 2, "wall": 3, "tree": 4, "cart": 5}
        self.classes = list(self.label_map.keys())
        
        # CRITICAL: Unified normalization strategy (proven dari Transformer model)
        if scaler is None and is_training:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
            print("✓ Unified StandardScaler fitted pada training data")
        elif scaler is not None:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)
            print("✓ Using shared StandardScaler untuk consistency")
        else:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
            
        # Bbox normalization (consistent approach)
        if bbox_scaler is None and is_training:
            self.bbox_scaler = StandardScaler() 
            self.bbox_targets = self.bbox_scaler.fit_transform(self.bbox_targets)
            print("✓ BBox normalization fitted")
        elif bbox_scaler is not None:
            self.bbox_scaler = bbox_scaler
            self.bbox_targets = self.bbox_scaler.transform(self.bbox_targets)
            print("✓ Using shared BBox StandardScaler untuk consistency")
        else:
            self.bbox_scaler = StandardScaler()
            self.bbox_targets = self.bbox_scaler.fit_transform(self.bbox_targets)
        
        print(f"✓ Unified Dataset: {len(self.data)} samples with 7 features")
        print(f"✓ Class distribution: {Counter(self.labels)}")
    
    def _load_json_data(self):
        """
        Proven data loading dari Transformer model - tidak ada perubahan
        Menggunakan 7 features: [x, y, z, width, height, depth, rotation_z]
        """
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
                    
                    # Find corresponding object label
                    for obj in json_data.get("objects", []):
                        if obj.get("key") == obj_id:
                            label = obj.get("classTitle")
                            break
                    
                    if label in label_map:
                        geometry = item.get("geometry", {})
                        position = geometry.get("position", {})
                        dimensions = geometry.get("size", geometry.get("dimensions", {"x": 1.0, "y": 1.0, "z": 1.0}))
                        
                        # Handle different dimension formats consistently
                        if isinstance(dimensions, dict):
                            dimensions = [dimensions.get(k, 1.0) for k in ["x", "y", "z"]]
                        elif not isinstance(dimensions, (list, np.ndarray)) or len(dimensions) != 3:
                            dimensions = [1.0, 1.0, 1.0]
                            
                        rotation_z = geometry.get("rotation", {}).get("z", 0.0)
                        features = list(position.values())
                        
                        if len(features) == 3:
                            # Create consistent 7-feature vector
                            all_features = features + dimensions + [rotation_z]
                            data.append(all_features)
                            labels.append(label_map[label])
                            bbox_targets.append(all_features)
                            file_info.append({
                                "file": file_name, 
                                "coords": features, 
                                "dimensions": dimensions, 
                                "rotation_z": rotation_z
                            })
                            
            except Exception as e:
                print(f"Warning: Error loading {file_name}: {e}")
                continue
        
        return (np.array(data, dtype=np.float32), 
                np.array(labels, dtype=np.int64),
                np.array(bbox_targets, dtype=np.float32), 
                file_info)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.data[idx])
        class_target = torch.LongTensor([self.labels[idx]])[0]
        bbox_target = torch.FloatTensor(self.bbox_targets[idx])
        
        # Keep original data untuk evaluation
        original_coords = torch.FloatTensor(self.file_info[idx]["coords"])
        original_dims = torch.FloatTensor(self.file_info[idx]["dimensions"])
        rotation_z = self.file_info[idx]["rotation_z"]
        
        return (features, class_target, bbox_target, 
                original_coords, original_dims, rotation_z)

# ============================================================================
# 2. SIMPLIFIED BUT EFFECTIVE GRAPH CONSTRUCTOR
# ============================================================================

class SmartGraphConstructor:
    """
    Graph constructor yang mengambil pendekatan balanced:
    - Tidak sekompleks GNN model (yang 23 features)
    - Tidak sesederhana basic fully connected
    - Fokus pada spatial proximity dengan parameter reasonable
    """
    def __init__(self, connection_radius=5.0, max_neighbors=8):
        self.connection_radius = connection_radius
        self.max_neighbors = max_neighbors
        print(f"✓ SmartGraphConstructor: radius={connection_radius}, max_neighbors={max_neighbors}")
    
    def build_spatial_graph(self, batch_features, batch_indices):
        """
        Build graph berdasarkan spatial coordinates (first 3 features: x, y, z)
        Returns edge_index dan edge_attr yang compatible dengan PyTorch Geometric atau custom GNN
        """
        device = batch_features.device
        positions = batch_features[:, :3]  # Extract spatial coordinates
        edge_indices = []
        edge_attrs = []
        
        unique_batches = torch.unique(batch_indices)
        
        for batch_id in unique_batches:
            mask = batch_indices == batch_id
            batch_positions = positions[mask]
            batch_size = batch_positions.size(0)
            
            if batch_size < 2:
                # Handle single node case dengan self-connection
                if batch_size == 1:
                    global_idx = torch.where(mask)[0][0].item()
                    edge_indices.append([global_idx, global_idx])
                    # Simple edge features: [relative_position, distance]
                    edge_attrs.append(torch.zeros(4, device=device))
                continue
            
            # Calculate pairwise distances
            distances = torch.cdist(batch_positions, batch_positions, p=2)
            
            # Build smart connections
            for i in range(batch_size):
                # Find neighbors within radius
                neighbor_mask = (distances[i] <= self.connection_radius) & (distances[i] > 0)
                neighbors = torch.where(neighbor_mask)[0]
                
                # Limit neighbors untuk avoid over-connectivity
                if len(neighbors) > self.max_neighbors:
                    neighbor_distances = distances[i][neighbors]
                    _, keep_indices = torch.topk(neighbor_distances, self.max_neighbors, largest=False)
                    neighbors = neighbors[keep_indices]
                
                # Create edges
                global_i = torch.where(mask)[0][i].item()
                for j in neighbors:
                    global_j = torch.where(mask)[0][j].item()
                    
                    # Bidirectional edges
                    edge_indices.extend([[global_i, global_j], [global_j, global_i]])
                    
                    # Simple but effective edge features
                    rel_pos = batch_positions[j] - batch_positions[i]
                    distance = distances[i][j]
                    edge_feat = torch.cat([rel_pos, distance.unsqueeze(0)])
                    edge_attrs.extend([edge_feat, -edge_feat])  # Symmetric
        
        # Convert to tensors atau handle empty case
        if len(edge_indices) == 0:
            num_nodes = len(batch_features)
            if num_nodes > 0:
                edge_indices = [[0, 0]]
                edge_attrs = [torch.zeros(4, device=device)]
        
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t().contiguous()
            edge_attr = torch.stack(edge_attrs)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, 4), device=device)
        
        return edge_index, edge_attr

# ============================================================================
# 3. FLEXIBLE GNN LAYER (WORKS WITH OR WITHOUT PYTORCH GEOMETRIC)
# ============================================================================

class FlexibleGNNLayer(nn.Module):
    """
    GNN layer yang bekerja dengan atau tanpa PyTorch Geometric
    Menggunakan approach yang proven effective tapi tidak overly complex
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            # Use advanced GCN layer
            self.gnn_layer = GCNConv(in_dim, out_dim)
            self.use_advanced = True
            print("✓ Using PyTorch Geometric GCNConv")
        else:
            # Use simple but effective message passing
            self.node_transform = nn.Linear(in_dim, out_dim)
            self.message_transform = nn.Linear(in_dim * 2, out_dim)
            self.use_advanced = False
            print("✓ Using simplified message passing")
        
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        if self.use_advanced:
            # PyTorch Geometric approach
            out = self.gnn_layer(x, edge_index)
        else:
            # Simple message passing
            out = self.node_transform(x)
            
            if edge_index.size(1) > 0:
                row, col = edge_index
                # Simple neighbor aggregation
                neighbor_features = x[col]
                source_features = x[row]
                messages = self.message_transform(torch.cat([source_features, neighbor_features], dim=1))
                
                # Aggregate messages per node
                out = out.index_add(0, row, messages) / 2  # Simple averaging
        
        out = self.activation(out)
        out = self.layer_norm(out)
        return out

# ============================================================================
# 4. HYBRID TRANSFORMER-GNN-FCN MODEL
# ============================================================================

class HybridTransformerGNNFCN(nn.Module):
    """
    Model hybrid yang menggabungkan:
    - Transformer branch: untuk attention-based feature learning (FIRST)
    - GNN branch: untuk spatial relationship modeling (SECOND) 
    - Smart fusion: menggabungkan kedua representation
    - FCN heads: untuk classification dan regression
    
    Design principle: Simple but powerful, avoid over-engineering
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_classes=6, 
                 num_heads=8, num_transformer_layers=3, num_gnn_layers=2, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        print(f"✓ Initializing Hybrid Model (Transformer+GNN): {input_dim}D -> {hidden_dim}D hidden")
        
        # Shared input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # === TRANSFORMER BRANCH (FIRST) ===
        # Proven configuration dari Transformer model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout, 
            batch_first=True, 
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # === GNN BRANCH (SECOND) ===
        self.graph_constructor = SmartGraphConstructor(connection_radius=5.0, max_neighbors=8)
        
        # Multiple GNN layers dengan residual connections
        self.gnn_layers = nn.ModuleList([
            FlexibleGNNLayer(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # === INTELLIGENT FUSION ===
        # Multi-head attention untuk smart fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=dropout
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # === OUTPUT HEADS (PROVEN DARI TRANSFORMER MODEL) ===
        # Classification head
        self.fc_class = nn.Linear(hidden_dim, num_classes)
        
        # Regression head
        self.fc_bbox = nn.Linear(hidden_dim, 7)
        
        # Activation
        self.gelu = nn.GELU()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"✓ Hybrid model ready: Transformer({num_transformer_layers}) + GNN({num_gnn_layers})")
    
    def _init_weights(self, module):
        """Initialize weights dengan Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, batch=None):
        batch_size = x.size(0)
        device = x.device
        
        # Create batch indices jika tidak provided
        if batch is None:
            batch = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Shared input embedding
        embedded = self.input_embedding(x)  # [N, hidden_dim]
        
        # === TRANSFORMER BRANCH (FIRST) ===
        # Process each sample sebagai sequence element
        transformer_input = embedded.unsqueeze(1)  # [N, 1, hidden_dim]
        transformer_features = self.transformer(transformer_input)
        transformer_features = transformer_features.squeeze(1)  # [N, hidden_dim]
        
        # === GNN BRANCH (SECOND) ===
        # Build graph berdasarkan spatial coordinates
        edge_index, edge_attr = self.graph_constructor.build_spatial_graph(x, batch)
        
        # Process through GNN layers dengan residual connections
        gnn_features = embedded
        for i, gnn_layer in enumerate(self.gnn_layers):
            residual = gnn_features
            gnn_features = gnn_layer(gnn_features, edge_index, edge_attr)
            
            # Add residual connection starting from second layer
            if i > 0:
                gnn_features = gnn_features + residual * 0.5  # Weighted residual
        
        # === INTELLIGENT FUSION ===
        # Use attention untuk smart combination
        transformer_query = transformer_features.unsqueeze(1)  # [N, 1, hidden_dim]
        gnn_kv = gnn_features.unsqueeze(1)  # [N, 1, hidden_dim]
        
        # Cross-attention: Transformer sebagai query, GNN sebagai key-value
        fused_features, attention_weights = self.fusion_attention(
            query=transformer_query, 
            key=gnn_kv, 
            value=gnn_kv
        )
        fused_features = fused_features.squeeze(1)  # [N, hidden_dim]
        
        # Layer normalization dengan residual dari embedding
        fused_features = self.fusion_norm(fused_features + embedded)
        
        # === OUTPUT HEADS ===
        # Apply GELU activation sebelum output (proven dari Transformer model)
        activated = self.gelu(fused_features)
        
        class_output = self.fc_class(activated)
        bbox_output = self.fc_bbox(activated)
        
        return class_output, bbox_output

# ============================================================================
# 5. MODEL COMPLEXITY ANALYSIS (FLOPs & Parameters)
# ============================================================================

def count_model_parameters(model):
    """
    Menghitung total parameters model secara detail.
    
    Returns:
        dict: Dictionary berisi informasi parameter lengkap
    """
    # Total trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Total non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    # Total semua parameters
    total_params = trainable_params + non_trainable_params
    
    # Per-layer breakdown
    layer_params = {}
    for name, param in model.named_parameters():
        layer_params[name] = {
            'shape': list(param.shape),
            'count': param.numel(),
            'trainable': param.requires_grad
        }
    
    # Group by component
    component_params = defaultdict(int)
    for name, param in model.named_parameters():
        # Extract component name (first part of parameter name)
        component = name.split('.')[0]
        component_params[component] += param.numel()
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'layer_params': layer_params,
        'component_params': dict(component_params),
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6
    }


def estimate_flops_manual(model, input_dim=7, hidden_dim=64, num_classes=6, 
                          num_heads=8, num_transformer_layers=3, num_gnn_layers=2):
    """
    Manual FLOPs estimation untuk Hybrid Transformer-GNN model.
    
    FLOPs dihitung berdasarkan operasi matematika utama:
    - Linear layers: 2 * input_features * output_features (multiply-add)
    - Attention: 4 * seq_len^2 * d_model (Q, K, V projections + attention)
    - LayerNorm: 5 * features (mean, var, normalize, scale, shift)
    - GELU: ~8 * features (approximation)
    
    Returns:
        dict: Dictionary berisi FLOPs breakdown per component
    """
    flops = {}
    
    # 1. Input Embedding: Linear(7, 64)
    # FLOPs = 2 * input_dim * hidden_dim (multiply-add)
    flops['input_embedding'] = 2 * input_dim * hidden_dim
    
    # 2. Transformer Encoder (per layer):
    # Each TransformerEncoderLayer contains:
    # - Multi-head Self-Attention
    # - Feed-forward network
    # - LayerNorms
    
    # Self-Attention for sequence length = 1:
    # Q, K, V projections: 3 * (2 * hidden_dim * hidden_dim)
    # Attention scores: 2 * seq_len * seq_len * hidden_dim
    # Output projection: 2 * hidden_dim * hidden_dim
    seq_len = 1  # Each sample treated as single token
    attention_flops_per_layer = (
        3 * 2 * hidden_dim * hidden_dim +  # Q, K, V projections
        2 * seq_len * seq_len * hidden_dim +  # Attention scores
        2 * hidden_dim * hidden_dim  # Output projection
    )
    
    # Feed-forward: Linear(64, 256) -> GELU -> Linear(256, 64)
    ff_dim = hidden_dim * 4  # 256
    ff_flops_per_layer = (
        2 * hidden_dim * ff_dim +  # First linear
        8 * ff_dim +  # GELU approximation
        2 * ff_dim * hidden_dim  # Second linear
    )
    
    # LayerNorms: 2 per layer, each ~5 * hidden_dim
    ln_flops_per_layer = 2 * 5 * hidden_dim
    
    transformer_flops_per_layer = attention_flops_per_layer + ff_flops_per_layer + ln_flops_per_layer
    flops['transformer_total'] = transformer_flops_per_layer * num_transformer_layers
    flops['transformer_per_layer'] = transformer_flops_per_layer
    
    # 3. GNN Layers (simplified estimation):
    # Each GNN layer: Linear transformation + message passing
    # Assuming average 8 neighbors per node
    avg_neighbors = 8
    gnn_flops_per_layer = (
        2 * hidden_dim * hidden_dim +  # Node transform
        avg_neighbors * 2 * (hidden_dim * 2) * hidden_dim +  # Message transform
        5 * hidden_dim +  # LayerNorm
        hidden_dim  # ReLU
    )
    flops['gnn_total'] = gnn_flops_per_layer * num_gnn_layers
    flops['gnn_per_layer'] = gnn_flops_per_layer
    
    # 4. Fusion Attention:
    # Similar to self-attention but cross-attention
    fusion_flops = (
        3 * 2 * hidden_dim * hidden_dim +  # Q, K, V projections
        2 * 1 * 1 * hidden_dim +  # Attention (seq_len=1)
        2 * hidden_dim * hidden_dim +  # Output projection
        5 * hidden_dim  # LayerNorm
    )
    flops['fusion_attention'] = fusion_flops
    
    # 5. Output Heads:
    # Classification: Linear(64, 6)
    flops['classification_head'] = 2 * hidden_dim * num_classes
    
    # Regression: Linear(64, 7)
    flops['regression_head'] = 2 * hidden_dim * 7
    
    # GELU before output
    flops['output_gelu'] = 8 * hidden_dim
    
    # Total FLOPs
    total_flops = sum(flops.values())
    flops['total'] = total_flops
    flops['total_mega'] = total_flops / 1e6
    flops['total_giga'] = total_flops / 1e9
    
    return flops


def calculate_model_complexity(model, device, input_dim=7):
    """
    Calculate comprehensive model complexity metrics including FLOPs and parameters.
    
    Args:
        model: The PyTorch model
        device: Device for computation
        input_dim: Input feature dimension
        
    Returns:
        dict: Comprehensive complexity metrics
    """
    print("📊 Calculating model complexity...")
    
    # 1. Parameter count
    param_info = count_model_parameters(model)
    
    # 2. FLOPs estimation
    if THOP_AVAILABLE:
        # Use thop for accurate FLOPs calculation
        try:
            # Create dummy input
            dummy_input = torch.randn(1, input_dim).to(device)
            dummy_batch = torch.zeros(1, dtype=torch.long).to(device)
            
            # Profile the model
            macs, params = profile(model, inputs=(dummy_input, dummy_batch), verbose=False)
            flops = macs * 2  # MACs to FLOPs (multiply-add = 2 operations)
            
            flops_info = {
                'total': int(flops),
                'total_mega': flops / 1e6,
                'total_giga': flops / 1e9,
                'method': 'thop_library'
            }
            print("✓ FLOPs calculated using thop library")
        except Exception as e:
            print(f"⚠️ thop failed: {e}, falling back to manual estimation")
            flops_info = estimate_flops_manual(model)
            flops_info['method'] = 'manual_estimation'
    else:
        # Use manual estimation
        flops_info = estimate_flops_manual(model)
        flops_info['method'] = 'manual_estimation'
    
    # 3. Memory estimation
    # Approximate memory usage in bytes
    param_memory = param_info['total_params'] * 4  # 4 bytes for float32
    
    # Combine all metrics
    complexity = {
        'parameters': param_info,
        'flops': flops_info,
        'memory': {
            'param_memory_bytes': param_memory,
            'param_memory_mb': param_memory / (1024 ** 2),
            'estimated_inference_memory_mb': param_memory / (1024 ** 2) * 2  # ~2x for activations
        }
    }
    
    print(f"✓ Total Parameters: {param_info['total_params']:,} ({param_info['total_params_millions']:.3f}M)")
    print(f"✓ Trainable Parameters: {param_info['trainable_params']:,}")
    print(f"✓ Estimated FLOPs: {flops_info['total_mega']:.3f}M ({flops_info['method']})")
    print(f"✓ Estimated Memory: {complexity['memory']['param_memory_mb']:.2f} MB")
    
    return complexity

# ============================================================================
# 6. PROVEN IoU CALCULATION (FROM TRANSFORMER MODEL)
# ============================================================================

def calculate_3d_iou_with_rotation(box1, box2):
    """
    Proven IoU calculation dari Transformer model - tidak ada perubahan
    Sudah terbukti menghasilkan hasil yang akurat
    """
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
            
            rotated_corners = corners @ rot_matrix.T + center
            return rotated_corners
        
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
        
    except Exception as e:
        return 0.0

def calculate_translation_error(pred_coords, gt_coords):
    """Calculate Euclidean distance between predicted dan ground truth centers"""
    return np.linalg.norm(np.array(pred_coords) - np.array(gt_coords))

# ============================================================================
# 7. MULTI-THRESHOLD AP CALCULATION
# ============================================================================

def calculate_ap_at_threshold(predictions, ground_truths, classes, iou_threshold):
    """
    Calculate Average Precision at a specific IoU threshold.
    
    Menggunakan pendekatan precision-recall curve integration untuk
    mendapatkan AP yang akurat pada threshold tertentu.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        classes: List of class names
        iou_threshold: IoU threshold untuk menentukan TP/FP
        
    Returns:
        dict: Per-class AP dan mAP pada threshold yang diberikan
    """
    per_class_ap = {}
    
    for class_id, class_name in enumerate(classes):
        # Filter predictions dan ground truths untuk class ini
        class_preds = [(i, pred) for i, pred in enumerate(predictions) 
                       if pred['class_id'] == class_id]
        class_gts = [gt for gt in ground_truths if gt['class_id'] == class_id]
        
        if len(class_gts) == 0:
            per_class_ap[class_name] = 0.0
            continue
        
        if len(class_preds) == 0:
            per_class_ap[class_name] = 0.0
            continue
        
        # Sort predictions by confidence (descending)
        class_preds_sorted = sorted(class_preds, key=lambda x: x[1]['confidence'], reverse=True)
        
        # Calculate TP and FP for each prediction
        tp = np.zeros(len(class_preds_sorted))
        fp = np.zeros(len(class_preds_sorted))
        
        # Track which ground truths have been matched
        gt_matched = [False] * len(class_gts)
        
        for pred_idx, (orig_idx, pred) in enumerate(class_preds_sorted):
            # Find matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_gts):
                if gt_matched[gt_idx]:
                    continue
                
                iou = pred['iou']  # IoU sudah dihitung sebelumnya
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if this is a true positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_idx] = 1
        
        # Calculate precision and recall at each threshold
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        recalls = cumsum_tp / len(class_gts)
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
        
        # Calculate AP using 11-point interpolation (PASCAL VOC style)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if np.any(mask):
                ap += np.max(precisions[mask])
        ap /= 11
        
        per_class_ap[class_name] = ap
    
    # Calculate mAP
    mAP = np.mean(list(per_class_ap.values()))
    
    return {
        'per_class_ap': per_class_ap,
        'mAP': mAP,
        'threshold': iou_threshold
    }


def calculate_multi_threshold_ap(predictions, ground_truths, classes):
    """
    Calculate AP at multiple IoU thresholds (0.5, 0.7, 0.9).
    
    Ini memberikan gambaran yang lebih komprehensif tentang performa model
    pada berbagai tingkat keketatan localization.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        classes: List of class names
        
    Returns:
        dict: AP metrics at all thresholds
    """
    thresholds = [0.5, 0.7, 0.9]
    results = {}
    
    for threshold in thresholds:
        key = f'AP@{threshold}'
        results[key] = calculate_ap_at_threshold(predictions, ground_truths, classes, threshold)
        print(f"  ✓ {key}: mAP = {results[key]['mAP']:.4f}")
    
    # Calculate AP across all thresholds (COCO-style mAP)
    all_maps = [results[f'AP@{t}']['mAP'] for t in thresholds]
    results['mAP_average'] = np.mean(all_maps)
    
    return results

# ============================================================================
# 8. FAILURE CASE ANALYSIS
# ============================================================================

def analyze_failure_cases(predictions, ground_truths, classes, iou_threshold=0.5):
    """
    Comprehensive failure case analysis untuk memahami kelemahan model.
    
    Menganalisis berbagai jenis kegagalan:
    1. Classification errors (misclassification)
    2. Localization errors (low IoU)
    3. False positives dan false negatives
    4. Per-class error patterns
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        classes: List of class names
        iou_threshold: IoU threshold untuk definisi TP/FP
        
    Returns:
        dict: Comprehensive failure analysis
    """
    print("🔍 Analyzing failure cases...")
    
    # Initialize containers
    misclassifications = []
    low_iou_detections = []
    false_positives = []
    false_negatives = []
    
    # Confusion matrix untuk misclassification analysis
    class_confusion = defaultdict(lambda: defaultdict(int))
    
    # IoU distribution per class
    iou_by_class = defaultdict(list)
    
    # Translation error analysis
    translation_errors_by_class = defaultdict(list)
    
    # Analyze each prediction
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_class = pred['class_id']
        gt_class = gt['class_id']
        iou = pred['iou']
        trans_error = pred['translation_error']
        
        # Track IoU by class
        iou_by_class[classes[gt_class]].append(iou)
        
        # Track translation error by class
        translation_errors_by_class[classes[gt_class]].append(trans_error)
        
        # 1. Misclassification analysis
        if pred_class != gt_class:
            misclassifications.append({
                'sample_id': i,
                'predicted_class': classes[pred_class],
                'true_class': classes[gt_class],
                'confidence': pred['confidence'],
                'iou': iou,
                'translation_error': trans_error,
                'pred_box': pred['box'].tolist() if hasattr(pred['box'], 'tolist') else list(pred['box']),
                'gt_box': gt['box'].tolist() if hasattr(gt['box'], 'tolist') else list(gt['box'])
            })
            class_confusion[classes[gt_class]][classes[pred_class]] += 1
        
        # 2. Low IoU detection (correct class but poor localization)
        if pred_class == gt_class and iou < iou_threshold:
            low_iou_detections.append({
                'sample_id': i,
                'class': classes[gt_class],
                'iou': iou,
                'translation_error': trans_error,
                'confidence': pred['confidence'],
                'pred_box': pred['box'].tolist() if hasattr(pred['box'], 'tolist') else list(pred['box']),
                'gt_box': gt['box'].tolist() if hasattr(gt['box'], 'tolist') else list(gt['box'])
            })
        
        # 3. False positive (high confidence but low IoU)
        if pred['confidence'] > 0.7 and iou < 0.3:
            false_positives.append({
                'sample_id': i,
                'predicted_class': classes[pred_class],
                'true_class': classes[gt_class],
                'confidence': pred['confidence'],
                'iou': iou
            })
    
    # 4. Calculate per-class failure statistics
    per_class_stats = {}
    for class_name in classes:
        class_ious = iou_by_class[class_name]
        class_trans_errors = translation_errors_by_class[class_name]
        
        if len(class_ious) > 0:
            per_class_stats[class_name] = {
                'total_samples': len(class_ious),
                'mean_iou': np.mean(class_ious),
                'std_iou': np.std(class_ious),
                'min_iou': np.min(class_ious),
                'max_iou': np.max(class_ious),
                'low_iou_count': sum(1 for iou in class_ious if iou < iou_threshold),
                'low_iou_rate': sum(1 for iou in class_ious if iou < iou_threshold) / len(class_ious),
                'mean_translation_error': np.mean(class_trans_errors),
                'std_translation_error': np.std(class_trans_errors)
            }
        else:
            per_class_stats[class_name] = {
                'total_samples': 0,
                'mean_iou': 0.0,
                'std_iou': 0.0,
                'min_iou': 0.0,
                'max_iou': 0.0,
                'low_iou_count': 0,
                'low_iou_rate': 0.0,
                'mean_translation_error': 0.0,
                'std_translation_error': 0.0
            }
    
    # 5. Identify most common misclassification pairs
    misclass_pairs = []
    for true_class, pred_dict in class_confusion.items():
        for pred_class, count in pred_dict.items():
            misclass_pairs.append({
                'true_class': true_class,
                'predicted_as': pred_class,
                'count': count
            })
    misclass_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    # 6. Calculate overall failure statistics
    total_samples = len(predictions)
    total_misclassifications = len(misclassifications)
    total_low_iou = len(low_iou_detections)
    total_false_positives = len(false_positives)
    
    # 7. Identify hardest samples (lowest IoU)
    all_samples = [(i, pred['iou'], classes[gt['class_id']]) 
                   for i, (pred, gt) in enumerate(zip(predictions, ground_truths))]
    hardest_samples = sorted(all_samples, key=lambda x: x[1])[:20]  # Bottom 20
    
    failure_analysis = {
        'summary': {
            'total_samples': total_samples,
            'total_misclassifications': total_misclassifications,
            'misclassification_rate': total_misclassifications / total_samples * 100,
            'total_low_iou_detections': total_low_iou,
            'low_iou_rate': total_low_iou / total_samples * 100,
            'total_false_positives': total_false_positives,
            'false_positive_rate': total_false_positives / total_samples * 100
        },
        'misclassifications': misclassifications[:50],  # Top 50 untuk detail
        'low_iou_detections': low_iou_detections[:50],  # Top 50 untuk detail
        'false_positives': false_positives,
        'per_class_stats': per_class_stats,
        'common_misclassification_pairs': misclass_pairs[:10],  # Top 10 pairs
        'hardest_samples': hardest_samples,
        'class_confusion_matrix': dict(class_confusion)
    }
    
    print(f"  ✓ Total samples: {total_samples}")
    print(f"  ✓ Misclassifications: {total_misclassifications} ({total_misclassifications/total_samples*100:.2f}%)")
    print(f"  ✓ Low IoU detections: {total_low_iou} ({total_low_iou/total_samples*100:.2f}%)")
    print(f"  ✓ False positives (high conf, low IoU): {total_false_positives}")
    
    return failure_analysis

# ============================================================================
# 9. COMPREHENSIVE EVALUATION (ENHANCED)
# ============================================================================

def evaluate_hybrid_model_comprehensive(model, test_loader, device, bbox_scaler, classes):
    """
    ENHANCED Comprehensive evaluation dengan semua metrics yang dibutuhkan:
    - Classification metrics
    - 3D detection metrics
    - Multi-threshold AP (0.5, 0.7, 0.9)
    - Performance metrics
    - Failure case analysis
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []
    all_class_predictions = []
    all_class_targets = []
    all_ious = []
    all_translation_errors = []
    
    print("🔎 Starting ENHANCED comprehensive hybrid model evaluation...")
    
    inference_time = 0.0  # Total inference time in seconds
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            features, class_targets, bbox_targets, orig_coords, orig_dims, rotation_z = batch_data
            
            features = features.to(device)
            class_targets = class_targets.to(device)
            
            # Create batch indices untuk graph construction
            batch_indices = torch.arange(len(features), device=device)
            
            # Measure inference time for model forward pass
            start_time = time.time()
            class_output, bbox_output = model(features, batch_indices)
            if device.type == 'cuda':
                torch.cuda.synchronize()  # Ensure GPU operations are complete
            end_time = time.time()
            inference_time += (end_time - start_time)
            
            class_preds = torch.argmax(class_output, dim=1)
            
            # Denormalize bbox predictions
            bbox_preds_denorm = bbox_scaler.inverse_transform(bbox_output.cpu().numpy())
            
            for i in range(len(features)):
                pred_box = bbox_preds_denorm[i]
                
                # Reconstruct ground truth box
                gt_coords = orig_coords[i].numpy()
                gt_dims = orig_dims[i].numpy()
                gt_rot = rotation_z[i] if isinstance(rotation_z[i], (int, float)) else rotation_z[i].item()
                gt_box = np.concatenate([gt_coords, gt_dims, [gt_rot]])
                
                # Calculate 3D IoU
                iou = calculate_3d_iou_with_rotation(pred_box, gt_box)
                all_ious.append(iou)
                
                # Calculate translation error
                translation_error = calculate_translation_error(pred_box[:3], gt_box[:3])
                all_translation_errors.append(translation_error)
                
                # Store predictions dan ground truths
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
    
    print("📊 Calculating ENHANCED comprehensive metrics...")
    
    # Classification metrics
    accuracy = accuracy_score(all_class_targets, all_class_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_class_targets, all_class_predictions, average=None, zero_division=0
    )
    
    # Per-class metrics
    per_class_metrics = []
    for i, class_name in enumerate(classes):
        per_class_metrics.append({
            'Class': class_name,
            'Precision': precision[i] if i < len(precision) else 0.0,
            'Recall': recall[i] if i < len(recall) else 0.0,
            'F1_Score': f1[i] if i < len(f1) else 0.0,
            'Support': support[i] if i < len(support) else 0
        })
    
    # Confusion matrix
    cm = confusion_matrix(all_class_targets, all_class_predictions)
    
    # 3D detection metrics
    mean_iou = np.mean(all_ious)
    mean_translation_error = np.mean(all_translation_errors)
    
    # Per-class IoU
    per_class_iou = {}
    for class_id, class_name in enumerate(classes):
        class_ious = [all_ious[i] for i, gt in enumerate(all_ground_truths) 
                     if gt['class_id'] == class_id]
        per_class_iou[class_name] = np.mean(class_ious) if class_ious else 0.0
    
    # === ENHANCED: Multi-threshold AP calculation ===
    print("\n📈 Calculating multi-threshold AP...")
    multi_threshold_ap = calculate_multi_threshold_ap(all_predictions, all_ground_truths, classes)
    
    # Extract AP@0.5 for backward compatibility
    per_class_ap = multi_threshold_ap['AP@0.5']['per_class_ap']
    mAP = multi_threshold_ap['AP@0.5']['mAP']
    
    # === ENHANCED: Failure case analysis ===
    print("\n🔍 Performing failure case analysis...")
    failure_analysis = analyze_failure_cases(all_predictions, all_ground_truths, classes)
    
    # Calculate average time per frame (ms/frame) and FPS
    num_samples = len(all_ious)
    avg_time_per_frame_ms = (inference_time / num_samples) * 1000 if num_samples > 0 else 0.0
    fps = num_samples / inference_time if inference_time > 0 else 0.0
    
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
            'mAP': mAP,
            'per_class_ap': per_class_ap,
            'mean_translation_error': mean_translation_error,
            'all_ious': all_ious,
            'all_translation_errors': all_translation_errors
        },
        'multi_threshold_ap': multi_threshold_ap,  # NEW
        'failure_analysis': failure_analysis,  # NEW
        'performance': {
            'avg_time_per_frame_ms': avg_time_per_frame_ms,
            'inference_speed_fps': fps,
            'total_inference_time': inference_time
        },
        'predictions': all_predictions,
        'ground_truths': all_ground_truths
    }

# ============================================================================
# 10. ENHANCED VISUALIZATION
# ============================================================================

def create_testing_visualizations(results, save_path):
    """
    Create ENHANCED visualizations untuk testing results.
    Termasuk multi-threshold AP dan failure analysis visualizations.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = results['classification']['confusion_matrix']
    classes = results['classification']['class_names']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Hybrid Transformer-GNN: Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'hybrid_confusion_matrix_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Classification metrics per class
    metrics_df = pd.DataFrame(results['classification']['per_class_metrics'])
    x = np.arange(len(metrics_df))
    width = 0.25
    
    axes[0].bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8, color='skyblue')
    axes[0].bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8, color='lightgreen')
    axes[0].bar(x + width, metrics_df['F1_Score'], width, label='F1-Score', alpha=0.8, color='salmon')
    
    axes[0].set_xlabel('Classes', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Hybrid Model: Classification Metrics per Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_df['Class'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)
    
    # 3D Detection metrics per class
    class_names = list(results['detection_3d']['per_class_iou'].keys())
    iou_values = list(results['detection_3d']['per_class_iou'].values())
    ap_values = list(results['detection_3d']['per_class_ap'].values())
    
    x = np.arange(len(class_names))
    axes[1].bar(x - 0.2, iou_values, 0.4, label='3D IoU', alpha=0.8, color='gold')
    axes[1].bar(x + 0.2, ap_values, 0.4, label='AP@0.5', alpha=0.8, color='lightcoral')
    
    axes[1].set_xlabel('Classes', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Hybrid Model: 3D Detection Metrics per Class', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'hybrid_per_class_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. NEW: Multi-threshold AP comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Multi-threshold mAP bar chart
    thresholds = ['AP@0.5', 'AP@0.7', 'AP@0.9']
    maps = [results['multi_threshold_ap'][t]['mAP'] for t in thresholds]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    bars = axes[0].bar(thresholds, maps, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, maps):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[0].set_ylabel('mAP Score', fontsize=12)
    axes[0].set_title('Mean Average Precision at Different IoU Thresholds', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Per-class AP comparison across thresholds
    x = np.arange(len(class_names))
    width = 0.25
    
    ap_05 = [results['multi_threshold_ap']['AP@0.5']['per_class_ap'][c] for c in class_names]
    ap_07 = [results['multi_threshold_ap']['AP@0.7']['per_class_ap'][c] for c in class_names]
    ap_09 = [results['multi_threshold_ap']['AP@0.9']['per_class_ap'][c] for c in class_names]
    
    axes[1].bar(x - width, ap_05, width, label='AP@0.5', color='#2ecc71', alpha=0.8)
    axes[1].bar(x, ap_07, width, label='AP@0.7', color='#f39c12', alpha=0.8)
    axes[1].bar(x + width, ap_09, width, label='AP@0.9', color='#e74c3c', alpha=0.8)
    
    axes[1].set_xlabel('Classes', fontsize=12)
    axes[1].set_ylabel('Average Precision', fontsize=12)
    axes[1].set_title('Per-Class AP at Different IoU Thresholds', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'hybrid_multi_threshold_ap_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. NEW: Failure Analysis Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # IoU distribution histogram
    all_ious = results['detection_3d']['all_ious']
    axes[0, 0].hist(all_ious, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='IoU=0.5')
    axes[0, 0].axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='IoU=0.7')
    axes[0, 0].axvline(x=0.9, color='red', linestyle='--', linewidth=2, label='IoU=0.9')
    axes[0, 0].set_xlabel('IoU Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('IoU Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Per-class failure rate
    failure_stats = results['failure_analysis']['per_class_stats']
    class_names_sorted = list(failure_stats.keys())
    low_iou_rates = [failure_stats[c]['low_iou_rate'] * 100 for c in class_names_sorted]
    
    bars = axes[0, 1].bar(class_names_sorted, low_iou_rates, color='coral', alpha=0.8, edgecolor='black')
    axes[0, 1].set_xlabel('Classes', fontsize=12)
    axes[0, 1].set_ylabel('Low IoU Rate (%)', fontsize=12)
    axes[0, 1].set_title('Per-Class Low IoU Rate (IoU < 0.5)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticklabels(class_names_sorted, rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, low_iou_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Translation error distribution
    all_trans_errors = results['detection_3d']['all_translation_errors']
    axes[1, 0].hist(all_trans_errors, bins=50, color='mediumpurple', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=np.mean(all_trans_errors), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(all_trans_errors):.2f}m')
    axes[1, 0].set_xlabel('Translation Error (m)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Translation Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Common misclassification pairs
    misclass_pairs = results['failure_analysis']['common_misclassification_pairs']
    if misclass_pairs:
        pair_labels = [f"{p['true_class']}→{p['predicted_as']}" for p in misclass_pairs[:8]]
        pair_counts = [p['count'] for p in misclass_pairs[:8]]
        
        bars = axes[1, 1].barh(pair_labels, pair_counts, color='lightcoral', alpha=0.8, edgecolor='black')
        axes[1, 1].set_xlabel('Count', fontsize=12)
        axes[1, 1].set_ylabel('True → Predicted', fontsize=12)
        axes[1, 1].set_title('Top Misclassification Pairs', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, pair_counts):
            axes[1, 1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                           str(val), ha='left', va='center', fontsize=10)
    else:
        axes[1, 1].text(0.5, 0.5, 'No misclassifications found!', 
                       ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Top Misclassification Pairs', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'hybrid_failure_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Enhanced testing visualizations saved with timestamp: {timestamp}")
    return timestamp


def save_testing_results(results, save_path, timestamp, testing_time, complexity=None):
    """
    Save ENHANCED testing results ke multi-sheet Excel file.
    Termasuk FLOPs, parameters, multi-threshold AP, dan failure analysis.
    """
    excel_filename = f"hybrid_testing_results_enhanced_{timestamp}.xlsx"
    excel_path = os.path.join(save_path, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Classification Metrics
        classification_df = pd.DataFrame(results['classification']['per_class_metrics'])
        classification_df.to_excel(writer, sheet_name='Classification_Metrics', index=False)
        
        # 2. Confusion Matrix
        cm_df = pd.DataFrame(
            results['classification']['confusion_matrix'],
            index=results['classification']['class_names'],
            columns=results['classification']['class_names']
        )
        cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
        
        # 3. 3D Detection Metrics (basic)
        detection_data = []
        for class_name in results['classification']['class_names']:
            detection_data.append({
                'Class': class_name,
                'IoU': results['detection_3d']['per_class_iou'][class_name],
                'AP@0.5': results['detection_3d']['per_class_ap'][class_name]
            })
        
        detection_df = pd.DataFrame(detection_data)
        detection_df.to_excel(writer, sheet_name='3D_Detection_Metrics', index=False)
        
        # 4. NEW: Multi-Threshold AP
        multi_ap_data = []
        for class_name in results['classification']['class_names']:
            row = {'Class': class_name}
            for threshold in ['AP@0.5', 'AP@0.7', 'AP@0.9']:
                row[threshold] = results['multi_threshold_ap'][threshold]['per_class_ap'][class_name]
            multi_ap_data.append(row)
        
        # Add mAP row
        map_row = {'Class': 'mAP (mean)'}
        for threshold in ['AP@0.5', 'AP@0.7', 'AP@0.9']:
            map_row[threshold] = results['multi_threshold_ap'][threshold]['mAP']
        multi_ap_data.append(map_row)
        
        multi_ap_df = pd.DataFrame(multi_ap_data)
        multi_ap_df.to_excel(writer, sheet_name='Multi_Threshold_AP', index=False)
        
        # 5. NEW: Failure Analysis Summary
        failure_summary = results['failure_analysis']['summary']
        failure_df = pd.DataFrame([
            {'Metric': 'Total Samples', 'Value': failure_summary['total_samples']},
            {'Metric': 'Misclassifications', 'Value': failure_summary['total_misclassifications']},
            {'Metric': 'Misclassification Rate (%)', 'Value': f"{failure_summary['misclassification_rate']:.2f}"},
            {'Metric': 'Low IoU Detections', 'Value': failure_summary['total_low_iou_detections']},
            {'Metric': 'Low IoU Rate (%)', 'Value': f"{failure_summary['low_iou_rate']:.2f}"},
            {'Metric': 'False Positives', 'Value': failure_summary['total_false_positives']},
            {'Metric': 'False Positive Rate (%)', 'Value': f"{failure_summary['false_positive_rate']:.2f}"}
        ])
        failure_df.to_excel(writer, sheet_name='Failure_Summary', index=False)
        
        # 6. NEW: Per-Class Failure Stats
        per_class_failure = []
        for class_name, stats in results['failure_analysis']['per_class_stats'].items():
            per_class_failure.append({
                'Class': class_name,
                'Total_Samples': stats['total_samples'],
                'Mean_IoU': f"{stats['mean_iou']:.4f}",
                'Std_IoU': f"{stats['std_iou']:.4f}",
                'Min_IoU': f"{stats['min_iou']:.4f}",
                'Max_IoU': f"{stats['max_iou']:.4f}",
                'Low_IoU_Count': stats['low_iou_count'],
                'Low_IoU_Rate': f"{stats['low_iou_rate']*100:.2f}%",
                'Mean_Trans_Error': f"{stats['mean_translation_error']:.4f}",
                'Std_Trans_Error': f"{stats['std_translation_error']:.4f}"
            })
        per_class_failure_df = pd.DataFrame(per_class_failure)
        per_class_failure_df.to_excel(writer, sheet_name='Per_Class_Failure_Stats', index=False)
        
        # 7. NEW: Common Misclassification Pairs
        misclass_pairs = results['failure_analysis']['common_misclassification_pairs']
        if misclass_pairs:
            misclass_df = pd.DataFrame(misclass_pairs)
            misclass_df.to_excel(writer, sheet_name='Misclassification_Pairs', index=False)
        
        # 8. Sample-wise Results
        sample_results_df = pd.DataFrame({
            'Sample_ID': range(len(results['detection_3d']['all_translation_errors'])),
            'Translation_Error': results['detection_3d']['all_translation_errors'],
            'IoU': results['detection_3d']['all_ious']
        })
        sample_results_df.to_excel(writer, sheet_name='Sample_Results', index=False)
        
        # 9. Model Summary dengan complexity info
        summary_data = {
            'Metric': [
                'Model_Architecture', 'Feature_Representation', 'Model_Source',
                'Overall_Accuracy', 'Mean_IoU', 
                'mAP@0.5', 'mAP@0.7', 'mAP@0.9', 'mAP_Average',
                'Mean_Translation_Error',
                'Testing_Time_Seconds', 'Testing_Time_Minutes', 'Total_Test_Samples',
                'Avg_Time_Per_Frame_ms', 'Inference_Speed_FPS'
            ],
            'Value': [
                'Hybrid Transformer-GNN-FCN (Testing Only)',
                '7 features (x,y,z,w,h,d,rot_z)',
                'Pre-trained Model: hybrid_transformer_gnn_20250702_224522.pth',
                f"{results['classification']['accuracy']:.4f}",
                f"{results['detection_3d']['mean_iou']:.4f}",
                f"{results['multi_threshold_ap']['AP@0.5']['mAP']:.4f}",
                f"{results['multi_threshold_ap']['AP@0.7']['mAP']:.4f}",
                f"{results['multi_threshold_ap']['AP@0.9']['mAP']:.4f}",
                f"{results['multi_threshold_ap']['mAP_average']:.4f}",
                f"{results['detection_3d']['mean_translation_error']:.4f}",
                f"{testing_time:.2f}",
                f"{testing_time/60:.2f}",
                len(results['detection_3d']['all_ious']),
                f"{results['performance']['avg_time_per_frame_ms']:.2f}",
                f"{results['performance']['inference_speed_fps']:.2f}"
            ]
        }
        
        # Add complexity info if available
        if complexity:
            summary_data['Metric'].extend([
                'Total_Parameters', 'Trainable_Parameters', 'Parameters_Millions',
                'Estimated_FLOPs', 'FLOPs_Millions', 'FLOPs_Method',
                'Param_Memory_MB', 'Estimated_Inference_Memory_MB'
            ])
            summary_data['Value'].extend([
                f"{complexity['parameters']['total_params']:,}",
                f"{complexity['parameters']['trainable_params']:,}",
                f"{complexity['parameters']['total_params_millions']:.3f}",
                f"{complexity['flops']['total']:,}",
                f"{complexity['flops']['total_mega']:.3f}",
                complexity['flops']['method'],
                f"{complexity['memory']['param_memory_mb']:.2f}",
                f"{complexity['memory']['estimated_inference_memory_mb']:.2f}"
            ])
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Testing_Summary', index=False)
        
        # 10. NEW: Model Complexity Details (if available)
        if complexity:
            # Component-wise parameters
            component_params = [{'Component': k, 'Parameters': v} 
                               for k, v in complexity['parameters']['component_params'].items()]
            component_df = pd.DataFrame(component_params)
            component_df.to_excel(writer, sheet_name='Component_Parameters', index=False)
    
    print(f"✅ Enhanced testing results saved: {excel_filename}")
    return excel_path

# ============================================================================
# 11. MAIN TESTING PROGRAM (ENHANCED)
# ============================================================================

def main():
    """
    ENHANCED Main program untuk Testing & Visualization
    Dengan tambahan:
    - FLOPs calculation
    - Model parameters counting
    - Multi-threshold AP (0.5, 0.7, 0.9)
    - Failure case analysis
    """
    # === CONFIGURATION PATHS ===
    PATHS = {
        'test_json': 'data/test',
        'pcd_base': 'data/pcd_test',
        'model_path': 'outputs/checkpoints/model_d.pth',
        'results_save': 'outputs/results',
        'images_save': 'outputs/figures'
    }
    
    # Create directories jika belum ada
    for path in [PATHS['results_save'], PATHS['images_save']]:
        os.makedirs(path, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Start timing
    total_start_time = time.time()
    
    print("=" * 80)
    print("🚀 ENHANCED HYBRID TRANSFORMER-GNN MODEL - TESTING & ANALYSIS")
    print("⚡ NEW FEATURES:")
    print("   📊 FLOPs calculation")
    print("   📈 Total model parameters")
    print("   🎯 Multi-threshold AP (AP@0.5, AP@0.7, AP@0.9)")
    print("   🔍 Comprehensive failure case analysis")
    print("=" * 80)
    
    try:
        # === 1. LOAD PRE-TRAINED MODEL ===
        print("\n📥 Loading pre-trained model...")
        
        if not os.path.exists(PATHS['model_path']):
            print(f"❌ Model file not found: {PATHS['model_path']}")
            print("💡 Please ensure the model file exists at the specified path")
            return
            
        # Load checkpoint
        checkpoint = torch.load(PATHS['model_path'], map_location=device)
        print(f"✅ Model checkpoint loaded from: {PATHS['model_path']}")
        
        # Extract scalers dari checkpoint
        scaler = checkpoint.get('scaler')
        bbox_scaler = checkpoint.get('bbox_scaler')
        
        if scaler is None or bbox_scaler is None:
            print("⚠️  Scalers not found in checkpoint, creating new ones...")
            temp_dataset = UnifiedHybridDataset(PATHS['test_json'], is_training=True)
            scaler = temp_dataset.scaler
            bbox_scaler = temp_dataset.bbox_scaler
        
        print("✅ Scalers loaded/created successfully")
        
        # === 2. LOAD TEST DATASET ===
        print("\n📂 Loading test dataset...")
        test_dataset = UnifiedHybridDataset(PATHS['test_json'], 
                                          scaler=scaler, 
                                          bbox_scaler=bbox_scaler, 
                                          is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        classes = ["cyclist", "car", "human", "wall", "tree", "cart"]
        
        # === 3. INITIALIZE MODEL ===
        print("\n🏗️  Initializing model architecture...")
        model = HybridTransformerGNNFCN(
            input_dim=7,
            hidden_dim=64,
            num_classes=6,
            num_heads=8,
            num_transformer_layers=3,
            num_gnn_layers=2,
            dropout=0.2
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Pre-trained model loaded and ready for testing")
        
        # === 4. NEW: MODEL COMPLEXITY ANALYSIS ===
        print("\n" + "=" * 60)
        print("📊 MODEL COMPLEXITY ANALYSIS")
        print("=" * 60)
        complexity = calculate_model_complexity(model, device, input_dim=7)
        
        # Print detailed complexity info
        print("\n📋 Parameter Breakdown by Component:")
        for component, params in complexity['parameters']['component_params'].items():
            print(f"   • {component}: {params:,} parameters")
        
        print(f"\n📈 FLOPs Breakdown:")
        if 'transformer_total' in complexity['flops']:
            print(f"   • Transformer: {complexity['flops']['transformer_total']:,} FLOPs")
            print(f"   • GNN: {complexity['flops']['gnn_total']:,} FLOPs")
            print(f"   • Fusion: {complexity['flops']['fusion_attention']:,} FLOPs")
        
        # === 5. TESTING PHASE ===
        print("\n" + "=" * 60)
        print("🧪 COMPREHENSIVE TESTING")
        print("=" * 60)
        testing_start_time = time.time()
        
        results = evaluate_hybrid_model_comprehensive(model, test_loader, device, bbox_scaler, classes)
        
        testing_time = time.time() - testing_start_time
        
        # === 6. PRINT COMPREHENSIVE RESULTS ===
        print("\n" + "=" * 60)
        print("📊 ENHANCED TESTING RESULTS")
        print("=" * 60)
        
        print(f"\n🎯 CLASSIFICATION METRICS:")
        print(f"   ✅ Accuracy: {results['classification']['accuracy']:.4f}")
        
        print(f"\n🎯 3D DETECTION METRICS:")
        print(f"   ✅ Mean 3D IoU: {results['detection_3d']['mean_iou']:.4f}")
        print(f"   ✅ Mean Translation Error: {results['detection_3d']['mean_translation_error']:.4f}m")
        
        print(f"\n🎯 MULTI-THRESHOLD AVERAGE PRECISION:")
        print(f"   ✅ mAP@0.5: {results['multi_threshold_ap']['AP@0.5']['mAP']:.4f}")
        print(f"   ✅ mAP@0.7: {results['multi_threshold_ap']['AP@0.7']['mAP']:.4f}")
        print(f"   ✅ mAP@0.9: {results['multi_threshold_ap']['AP@0.9']['mAP']:.4f}")
        print(f"   ✅ mAP (average): {results['multi_threshold_ap']['mAP_average']:.4f}")
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   ✅ Avg Time per Frame: {results['performance']['avg_time_per_frame_ms']:.2f} ms/frame")
        print(f"   ✅ Inference Speed: {results['performance']['inference_speed_fps']:.2f} FPS")
        print(f"   ⏱️  Testing Duration: {testing_time:.2f} seconds ({testing_time/60:.2f} minutes)")
        
        print(f"\n🎯 MODEL COMPLEXITY:")
        print(f"   ✅ Total Parameters: {complexity['parameters']['total_params']:,} ({complexity['parameters']['total_params_millions']:.3f}M)")
        print(f"   ✅ Trainable Parameters: {complexity['parameters']['trainable_params']:,}")
        print(f"   ✅ Estimated FLOPs: {complexity['flops']['total_mega']:.3f}M")
        print(f"   ✅ Memory: {complexity['memory']['param_memory_mb']:.2f} MB")
        
        print(f"\n🔍 FAILURE ANALYSIS SUMMARY:")
        failure_summary = results['failure_analysis']['summary']
        print(f"   ✅ Total Samples: {failure_summary['total_samples']}")
        print(f"   ⚠️  Misclassifications: {failure_summary['total_misclassifications']} ({failure_summary['misclassification_rate']:.2f}%)")
        print(f"   ⚠️  Low IoU Detections: {failure_summary['total_low_iou_detections']} ({failure_summary['low_iou_rate']:.2f}%)")
        print(f"   ⚠️  False Positives: {failure_summary['total_false_positives']} ({failure_summary['false_positive_rate']:.2f}%)")
        
        # Per-class summary
        print("\n📈 PER-CLASS SUMMARY:")
        print(f"   {'Class':<10} {'IoU':>8} {'AP@0.5':>8} {'AP@0.7':>8} {'AP@0.9':>8} {'Fail Rate':>10}")
        print("   " + "-" * 54)
        for class_name in classes:
            iou = results['detection_3d']['per_class_iou'][class_name]
            ap05 = results['multi_threshold_ap']['AP@0.5']['per_class_ap'][class_name]
            ap07 = results['multi_threshold_ap']['AP@0.7']['per_class_ap'][class_name]
            ap09 = results['multi_threshold_ap']['AP@0.9']['per_class_ap'][class_name]
            fail_rate = results['failure_analysis']['per_class_stats'][class_name]['low_iou_rate'] * 100
            print(f"   {class_name:<10} {iou:>8.3f} {ap05:>8.3f} {ap07:>8.3f} {ap09:>8.3f} {fail_rate:>9.2f}%")
        
        # Common misclassification pairs
        if results['failure_analysis']['common_misclassification_pairs']:
            print("\n📊 TOP MISCLASSIFICATION PAIRS:")
            for pair in results['failure_analysis']['common_misclassification_pairs'][:5]:
                print(f"   • {pair['true_class']} → {pair['predicted_as']}: {pair['count']} cases")
        
        # === 7. CREATE TESTING VISUALIZATIONS ===
        print("\n🎨 Creating enhanced visualizations...")
        timestamp = create_testing_visualizations(results, PATHS['images_save'])
        
        # === 8. SAVE TESTING RESULTS ===
        excel_path = save_testing_results(results, PATHS['results_save'], timestamp, testing_time, complexity)
        
        # === 9. FINAL SUMMARY ===
        total_time = time.time() - total_start_time
        
        print("\n" + "=" * 60)
        print("🎉 ENHANCED TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✅ Excel results: {excel_path}")
        print(f"✅ Visualizations: {PATHS['images_save']}")
        print(f"⏱️  Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        print("\n📋 SUMMARY OF NEW FEATURES:")
        print(f"   ✅ FLOPs: {complexity['flops']['total_mega']:.3f} MFLOPs")
        print(f"   ✅ Parameters: {complexity['parameters']['total_params_millions']:.3f}M")
        print(f"   ✅ AP@0.5: {results['multi_threshold_ap']['AP@0.5']['mAP']:.4f}")
        print(f"   ✅ AP@0.7: {results['multi_threshold_ap']['AP@0.7']['mAP']:.4f}")
        print(f"   ✅ AP@0.9: {results['multi_threshold_ap']['AP@0.9']['mAP']:.4f}")
        print(f"   ✅ Failure Cases Analyzed: {failure_summary['total_samples']} samples")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting tips:")
        print("   1. Verify model file exists at specified path")
        print("   2. Check test data paths are accessible")
        print("   3. Ensure GPU memory availability")
        print("   4. Verify data format consistency")
        print("   5. Install thop library for accurate FLOPs: pip install thop")

if __name__ == "__main__":
    main()