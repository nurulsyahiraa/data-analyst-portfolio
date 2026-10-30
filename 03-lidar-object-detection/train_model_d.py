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
from collections import Counter
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
    def __init__(self, json_folder, scaler=None, is_training=True):
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
        if is_training:
            self.bbox_scaler = StandardScaler() 
            self.bbox_targets = self.bbox_scaler.fit_transform(self.bbox_targets)
            print("✓ BBox normalization fitted")
        
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
# 4. HYBRID TRANSFORMER-GNN-FCN MODEL (REORDERED)
# ============================================================================

class HybridTransformerGNNFCN(nn.Module):
    """
    Model hybrid yang menggabungkan (URUTAN BARU):
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
        
        # === INTELLIGENT FUSION (REORDERED) ===
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
# 5. UNIFIED TRAINER WITH COST SENSITIVE LEARNING
# ============================================================================

class UnifiedHybridTrainer:
    """
    Trainer yang menggabungkan best practices dari kedua model:
    - Cost-sensitive learning dari kedua program
    - Comprehensive logging
    - Robust training loop dengan early stopping
    """
    def __init__(self, model, device='cuda', train_dataset=None):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # === COST-SENSITIVE LEARNING ===
        if train_dataset is not None:
            class_weights = self._calculate_class_weights(train_dataset)
            self.criterion_class = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print(f"✓ Cost-sensitive weights applied: {class_weights.cpu().numpy()}")
        else:
            self.criterion_class = nn.CrossEntropyLoss()
        
        # Regression loss (proven dari Transformer model)
        self.criterion_bbox = nn.L1Loss()
        
        # === OPTIMIZER & SCHEDULER ===
        # Use proven configuration dari Transformer model
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.3, patience=5, verbose=True
        )
        
        # Loss balancing (proven dari Transformer model)
        self.alpha = 1.0
        
        # === COMPREHENSIVE TRACKING ===
        self.training_history = {
            'epoch': [], 'train_loss': [], 'train_acc': [], 
            'train_class_loss': [], 'train_regression_loss': [],
            'val_loss': [], 'val_acc': [], 'val_class_loss': [], 
            'val_regression_loss': [], 'learning_rate': [], 'epoch_time': []
        }
        
        print("=== UNIFIED HYBRID TRAINER INITIALIZED ===")
        print("✓ Cost-sensitive learning enabled")
        print("✓ Comprehensive metrics tracking enabled")
    
    def _calculate_class_weights(self, dataset):
        """Calculate balanced class weights untuk cost-sensitive learning"""
        labels = [dataset.labels[i] for i in range(len(dataset))]
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
        
        weights_tensor = torch.ones(6)
        for i, weight in zip(unique_labels, class_weights):
            weights_tensor[i] = weight
            
        return weights_tensor
    
    def train_epoch(self, train_loader):
        """Training untuk satu epoch dengan detailed monitoring"""
        self.model.train()
        total_loss = 0
        total_class_loss = 0
        total_regression_loss = 0
        correct = 0
        total = 0
        
        epoch_start_time = time.time()
        
        for batch_X, batch_y, batch_bbox, _, _, _ in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device) 
            batch_bbox = batch_bbox.to(self.device)
            
            # Create batch indices untuk graph construction
            batch_indices = torch.arange(len(batch_X), device=self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            class_output, bbox_output = self.model(batch_X, batch_indices)
            
            # Calculate losses
            loss_class = self.criterion_class(class_output, batch_y)
            loss_bbox = self.criterion_bbox(bbox_output, batch_bbox)
            total_loss_batch = loss_class + self.alpha * loss_bbox
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping untuk stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += total_loss_batch.item()
            total_class_loss += loss_class.item()
            total_regression_loss += loss_bbox.item()
            
            predicted = torch.argmax(class_output, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        epoch_time = time.time() - epoch_start_time
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_regression_loss = total_regression_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, avg_class_loss, avg_regression_loss, epoch_time
    
    def validate_epoch(self, val_loader):
        """Validation untuk satu epoch"""
        self.model.eval()
        total_loss = 0
        total_class_loss = 0
        total_regression_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y, batch_bbox, _, _, _ in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_bbox = batch_bbox.to(self.device)
                
                batch_indices = torch.arange(len(batch_X), device=self.device)
                
                class_output, bbox_output = self.model(batch_X, batch_indices)
                
                loss_class = self.criterion_class(class_output, batch_y)
                loss_bbox = self.criterion_bbox(bbox_output, batch_bbox)
                total_loss_batch = loss_class + self.alpha * loss_bbox
                
                total_loss += total_loss_batch.item()
                total_class_loss += loss_class.item()
                total_regression_loss += loss_bbox.item()
                
                predicted = torch.argmax(class_output, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_class_loss = total_class_loss / len(val_loader)
        avg_regression_loss = total_regression_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, avg_class_loss, avg_regression_loss
    
    def train(self, train_loader, val_loader, epochs=100, save_path=None):
        """Full training loop dengan early stopping"""
        print("=== STARTING HYBRID TRAINING ===")
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15  # Reasonable patience
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc, train_class_loss, train_reg_loss, epoch_time = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc, val_class_loss, val_reg_loss = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record comprehensive history
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['train_class_loss'].append(train_class_loss)
            self.training_history['train_regression_loss'].append(train_reg_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_class_loss'].append(val_class_loss)
            self.training_history['val_regression_loss'].append(val_reg_loss)
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['epoch_time'].append(epoch_time)
            
            # Progress reporting
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%')
                print(f'           Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')
                print(f'           Class Loss: Train={train_class_loss:.4f}, Val={val_class_loss:.4f}')
                print(f'           Reg Loss: Train={train_reg_loss:.4f}, Val={val_reg_loss:.4f}')
                print(f'           Learning Rate: {current_lr:.2e}')
            
            # Early stopping dan model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'scaler': train_loader.dataset.scaler,
                        'bbox_scaler': train_loader.dataset.bbox_scaler,
                        'training_history': self.training_history,
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
            # Stop jika learning rate terlalu kecil
            if current_lr < 1e-7:
                print(f"Learning rate too small, stopping at epoch {epoch}")
                break
        
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return self.training_history

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
# 7. COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_hybrid_model_comprehensive(model, test_loader, device, bbox_scaler, classes):
    """
    Comprehensive evaluation dengan semua metrics yang dibutuhkan
    Menggunakan approach proven dari Transformer model
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []
    all_class_predictions = []
    all_class_targets = []
    all_ious = []
    all_translation_errors = []
    
    print("🔍 Starting comprehensive hybrid model evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            features, class_targets, bbox_targets, orig_coords, orig_dims, rotation_z = batch_data
            
            features = features.to(device)
            class_targets = class_targets.to(device)
            
            # Create batch indices untuk graph construction
            batch_indices = torch.arange(len(features), device=device)
            
            # Model prediction
            class_output, bbox_output = model(features, batch_indices)
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
    
    print("📊 Calculating comprehensive metrics...")
    
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
    
    # Per-class IoU dan AP
    per_class_iou = {}
    per_class_ap = {}
    
    for class_id, class_name in enumerate(classes):
        # IoU per class
        class_ious = [all_ious[i] for i, pred in enumerate(all_predictions) 
                     if all_ground_truths[i]['class_id'] == class_id]
        per_class_iou[class_name] = np.mean(class_ious) if class_ious else 0.0
        
        # AP per class (simplified: TP at IoU > 0.5 threshold)
        class_preds = [pred for pred in all_predictions if pred['class_id'] == class_id]
        class_gts = [gt for gt in all_ground_truths if gt['class_id'] == class_id]
        
        if len(class_gts) == 0:
            per_class_ap[class_name] = 0.0
        else:
            tp = sum(1 for pred in class_preds if pred['iou'] >= 0.5)
            per_class_ap[class_name] = tp / len(class_gts)
    
    mAP = np.mean(list(per_class_ap.values()))
    
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
        'predictions': all_predictions,
        'ground_truths': all_ground_truths
    }

# ============================================================================
# 8. EXCELLENT VISUALIZATION (FROM TRANSFORMER MODEL)
# ============================================================================

def create_comprehensive_visualizations(training_history, results, save_path):
    """
    Create comprehensive visualizations menggunakan proven approach dari Transformer model
    Tidak ada perubahan dari approach yang sudah proven effective
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Training curves (2x2 layout yang proven)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total loss curves
    axes[0, 0].plot(training_history['epoch'], training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(training_history['epoch'], training_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Hybrid GNN-Transformer: Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(training_history['epoch'], training_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(training_history['epoch'], training_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Hybrid GNN-Transformer: Classification Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Regression loss curves
    axes[1, 0].plot(training_history['epoch'], training_history['train_regression_loss'], 'b-', label='Train Reg Loss', linewidth=2)
    axes[1, 0].plot(training_history['epoch'], training_history['val_regression_loss'], 'r-', label='Val Reg Loss', linewidth=2)
    axes[1, 0].set_title('Hybrid GNN-Transformer: Regression Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Regression Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[1, 1].plot(training_history['epoch'], training_history['learning_rate'], 'g-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'hybrid_training_curves_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix (proven style)
    plt.figure(figsize=(10, 8))
    cm = results['classification']['confusion_matrix']
    classes = results['classification']['class_names']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Hybrid GNN-Transformer: Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'hybrid_confusion_matrix_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-class metrics (proven layout)
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
    axes[1].bar(x + 0.2, ap_values, 0.4, label='Average Precision', alpha=0.8, color='lightcoral')
    
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
    
    print(f"✅ Comprehensive visualizations saved with timestamp: {timestamp}")
    return timestamp

def save_comprehensive_results(training_history, results, save_path, timestamp):
    """
    Save comprehensive results ke multi-sheet Excel file
    Menggunakan proven approach dari Transformer model
    """
    excel_filename = f"hybrid_transformer_gnn_results_{timestamp}.xlsx"
    excel_path = os.path.join(save_path, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Training History
        training_df = pd.DataFrame(training_history)
        training_df.to_excel(writer, sheet_name='Training_History', index=False)
        
        # 2. Classification Metrics
        classification_df = pd.DataFrame(results['classification']['per_class_metrics'])
        classification_df.to_excel(writer, sheet_name='Classification_Metrics', index=False)
        
        # 3. Confusion Matrix
        cm_df = pd.DataFrame(
            results['classification']['confusion_matrix'],
            index=results['classification']['class_names'],
            columns=results['classification']['class_names']
        )
        cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
        
        # 4. 3D Detection Metrics
        detection_data = []
        for class_name in results['classification']['class_names']:
            detection_data.append({
                'Class': class_name,
                'IoU': results['detection_3d']['per_class_iou'][class_name],
                'AP': results['detection_3d']['per_class_ap'][class_name]
            })
        
        detection_df = pd.DataFrame(detection_data)
        detection_df.to_excel(writer, sheet_name='3D_Detection_Metrics', index=False)
        
        # 5. Sample-wise Results
        sample_results_df = pd.DataFrame({
            'Sample_ID': range(len(results['detection_3d']['all_translation_errors'])),
            'Translation_Error': results['detection_3d']['all_translation_errors'],
            'IoU': results['detection_3d']['all_ious']
        })
        sample_results_df.to_excel(writer, sheet_name='Sample_Results', index=False)
        
        # 6. Model Summary
        summary_data = {
            'Metric': [
                'Model_Architecture', 'Feature_Representation', 'Cost_Sensitive_Learning',
                'Overall_Accuracy', 'Mean_IoU', 'mAP', 'Mean_Translation_Error',
                'Training_Epochs', 'Best_Val_Loss', 'Final_Train_Acc', 'Final_Val_Acc'
            ],
            'Value': [
                'Hybrid Transformer-GNN-FCN',
                '7 features (x,y,z,w,h,d,rot_z)',
                'Enabled (Balanced Class Weights)',
                results['classification']['accuracy'],
                results['detection_3d']['mean_iou'],
                results['detection_3d']['mAP'],
                results['detection_3d']['mean_translation_error'],
                len(training_history['epoch']),
                min(training_history['val_loss']),
                training_history['train_acc'][-1],
                training_history['val_acc'][-1]
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)
    
    print(f"✅ Comprehensive results saved: {excel_filename}")
    return excel_path

# ============================================================================
# 9. INTERACTIVE 3D VISUALIZATION (FROM TRANSFORMER MODEL - PROVEN EFFECTIVE)
# ============================================================================

def create_interactive_3d_visualizations(results, pcd_base_path, json_base_path, num_files=5):
    """
    CLEAN & ELEGANT Interactive 3D visualizations dari Transformer model
    Tidak ada perubahan - proven effective implementation
    
    Features:
    - Point cloud sebagai background utama (prominent)
    - Solid colored bounding boxes (bukan wireframe)
    - Elegant labels dengan background box
    - No clutter (no axes, no help text, no unnecessary UI)
    - Smart label positioning untuk avoid overlap
    """
    if not VISUALIZATION_3D_AVAILABLE:
        print("⚠️  3D visualization libraries not available. Skipping 3D visualizations.")
        print("💡 Program akan tetap berjalan dengan visualization 2D yang comprehensive.")
        return
    
    print(f"🎨 Creating CLEAN Interactive 3D visualizations for {num_files} random files...")
    print(f"📁 PCD path: {pcd_base_path}")
    print(f"📁 JSON path: {json_base_path}")
    
    # Solid colors untuk bounding boxes (sesuai requirements)
    class_colors = {
        'cart': [2/255, 156/255, 122/255],      # #029C7A - Teal
        'tree': [7/255, 236/255, 123/255],      # #07EC7B - Bright Green
        'wall': [252/255, 191/255, 198/255],    # #FCBFC6 - Light Pink
        'human': [197/255, 175/255, 139/255],   # #C5AF8B - Tan
        'cyclist': [188/255, 35/255, 197/255],  # #BC23C5 - Purple
        'car': [110/255, 19/255, 255/255]       # #6E13FF - Blue
    }
    
    # Get available JSON files
    json_files = [f for f in os.listdir(json_base_path) if f.endswith('.pcd.json')]
    if len(json_files) == 0:
        json_files = [f for f in os.listdir(json_base_path) if f.endswith('.json')]
    
    if len(json_files) == 0:
        print("❌ No JSON files found!")
        return
    
    selected_files = random.sample(json_files, min(num_files, len(json_files)))
    print(f"🎯 Selected {len(selected_files)} files for visualization")
    
    successful_visualizations = 0
    
    for i, json_file in enumerate(selected_files):
        try:
            print(f"\n🔄 Processing file {i+1}/{len(selected_files)}: {json_file}")
            
            # Load JSON data
            json_path = os.path.join(json_base_path, json_file)
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Handle naming convention
            if json_file.endswith('.pcd.json'):
                pcd_file = json_file[:-5]
            else:
                pcd_file = json_file.replace('.json', '.pcd')
            
            pcd_path = os.path.join(pcd_base_path, pcd_file)
            
            if not os.path.exists(pcd_path):
                print(f"⚠️  PCD file not found: {pcd_path}")
                continue
            
            # Load point cloud
            try:
                pcd = o3d.io.read_point_cloud(pcd_path)
                points = np.asarray(pcd.points)
                
                if len(points) == 0:
                    print(f"⚠️  Empty point cloud: {pcd_file}")
                    continue
                    
                print(f"✅ Loaded point cloud: {len(points)} points")
                
            except Exception as e:
                print(f"❌ Error loading PCD: {e}")
                continue
            
            # Create CLEAN plotter (no title, minimal UI)
            plotter = pv.Plotter(window_size=[1800, 1200])
            
            # CRITICAL: Point cloud sebagai BACKGROUND dengan warna grey
            point_cloud = pv.PolyData(points)
            plotter.add_mesh(point_cloud, 
                           point_size=2, 
                           color='lightgrey', 
                           opacity=1.0,  # Full opacity untuk clear background
                           render_points_as_spheres=True)
            
            # Calculate centroid dan bounds untuk intelligent positioning
            pcd_centroid = np.mean(points, axis=0)
            pcd_min = np.min(points, axis=0)
            pcd_max = np.max(points, axis=0)
            scene_bounds = pcd_max - pcd_min
            
            print(f"📍 Point cloud centroid: {pcd_centroid}")
            
            # Extract ground truth dan predictions
            label_map = {"cyclist": 0, "car": 1, "human": 2, "wall": 3, "tree": 4, "cart": 5}
            classes = ["cyclist", "car", "human", "wall", "tree", "cart"]
            
            # Extract ground truth bounding boxes
            ground_truth_boxes = []
            for figure in json_data.get('figures', []):
                obj_id = figure.get('objectKey')
                class_name = None
                
                for obj in json_data.get('objects', []):
                    if obj.get('key') == obj_id:
                        class_name = obj.get('classTitle')
                        break
                
                if class_name and class_name in label_map:
                    geometry = figure.get('geometry', {})
                    position = geometry.get('position', {})
                    dimensions = geometry.get('size', geometry.get('dimensions', {}))
                    rotation = geometry.get('rotation', {})
                    
                    if isinstance(position, dict) and len(position) >= 3:
                        coords = [position.get('x', 0), position.get('y', 0), position.get('z', 0)]
                        
                        if isinstance(dimensions, dict):
                            dims = [dimensions.get('x', 1), dimensions.get('y', 1), dimensions.get('z', 1)]
                        else:
                            dims = [1.0, 1.0, 1.0]
                        
                        rot_z = rotation.get('z', 0) if isinstance(rotation, dict) else 0
                        
                        ground_truth_boxes.append({
                            'class_name': class_name,
                            'center': coords,
                            'dimensions': dims,
                            'rotation_z': rot_z,
                            'box': coords + dims + [rot_z]  # For IoU calculation
                        })
            
            print(f"📦 Found {len(ground_truth_boxes)} ground truth boxes")
            
            # Find actual model predictions untuk file ini berdasarkan ground truth
            actual_predictions = []
            
            for gt_box in ground_truth_boxes:
                # Find closest prediction dari results berdasarkan spatial proximity
                best_pred = None
                min_distance = float('inf')
                
                for pred in results['predictions']:
                    # Calculate distance antara ground truth center dan prediction center
                    gt_center = np.array(gt_box['center'])
                    pred_center = np.array(pred['box'][:3])
                    distance = np.linalg.norm(gt_center - pred_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_pred = pred
                
                if best_pred and min_distance < max(scene_bounds) * 0.3:  # Reasonable distance threshold
                    # Calculate real IoU antara prediction dan ground truth
                    real_iou = calculate_3d_iou_with_rotation(best_pred['box'], gt_box['box'])
                    
                    actual_predictions.append({
                        'predicted_class': best_pred['class_name'],
                        'true_class': gt_box['class_name'],
                        'center': gt_box['center'],  # Use GT center untuk positioning
                        'dimensions': gt_box['dimensions'],  # Use GT dimensions untuk clean display
                        'rotation_z': gt_box['rotation_z'],  # Use GT rotation untuk clean display
                        'confidence': best_pred['confidence'],
                        'real_iou': real_iou,
                        'prediction_box': best_pred['box'],
                        'ground_truth_box': gt_box['box']
                    })
                else:
                    # No matching prediction found - show as missed detection
                    actual_predictions.append({
                        'predicted_class': 'missed',
                        'true_class': gt_box['class_name'],
                        'center': gt_box['center'],
                        'dimensions': gt_box['dimensions'],
                        'rotation_z': gt_box['rotation_z'],
                        'confidence': 0.0,
                        'real_iou': 0.0,
                        'prediction_box': None,
                        'ground_truth_box': gt_box['box']
                    })
            
            print(f"🎯 Matched {len(actual_predictions)} predictions dengan ground truth")
            
            # Add bounding boxes dengan SOLID style dan ELEGANT labels
            for j, pred in enumerate(actual_predictions):
                try:
                    true_class = pred['true_class']
                    predicted_class = pred['predicted_class']
                    center = np.array(pred['center'])
                    dimensions = np.abs(np.array(pred['dimensions']))
                    rotation_z = pred['rotation_z']
                    confidence = pred['confidence']
                    real_iou = pred['real_iou']
                    
                    # Create oriented bounding box
                    box_mesh = pv.Cube(center=center,
                                       x_length=dimensions[0],
                                       y_length=dimensions[1], 
                                       z_length=dimensions[2])
                    
                    # Apply rotation
                    if abs(rotation_z) > 1e-6:
                        box_mesh.rotate_z(np.degrees(rotation_z), point=center, inplace=True)
                    
                    # Determine color berdasarkan prediction
                    if predicted_class == 'missed':
                        color = [0.5, 0.5, 0.5]  # Grey untuk missed detection
                        display_color = color
                    else:
                        # Use exact colors sesuai spesifikasi
                        color = class_colors.get(predicted_class, [0.5, 0.5, 0.5])
                        display_color = color
                    
                    # SOLID colored bounding box - FULL OPACITY (tidak transparan)
                    plotter.add_mesh(box_mesh, 
                                   color=display_color, 
                                   opacity=1.0,  # FULL SOLID - tidak transparan
                                   show_edges=False)  # Pure solid tanpa edges
                    
                    # Label HARUS TEPAT DI ATAS BOUNDING BOX dan mengikuti rotation
                    # Calculate exact top center dari rotated bounding box
                    if abs(rotation_z) > 1e-6:
                        # For rotated boxes, calculate actual top center
                        half_dims = dimensions / 2
                        top_center_local = [0, 0, half_dims[2]]  # Local top center
                        
                        # Apply same rotation ke label position
                        cos_r, sin_r = np.cos(rotation_z), np.sin(rotation_z)
                        rot_matrix = np.array([
                            [cos_r, -sin_r, 0],
                            [sin_r,  cos_r, 0], 
                            [0,      0,     1]
                        ])
                        rotated_top = np.dot(rot_matrix, top_center_local)
                        label_pos = center + rotated_top + [0, 0, 0.2]  # Small offset di atas
                    else:
                        # For non-rotated boxes, simple top center
                        label_pos = center + [0, 0, dimensions[2]/2 + 0.2]
                    
                    # Create ELEGANT label dengan SEMUA TEKS HITAM
                    if predicted_class == 'missed':
                        # Missed detection
                        label_text = f"MISSED\nReality: {true_class.title()}\nIoU: 0.00"
                    elif predicted_class == true_class:
                        # Correct prediction
                        label_text = f"{predicted_class.title()}\nConfidence: {confidence:.2f}\nIoU: {real_iou:.2f}"
                    else:
                        # Wrong prediction
                        label_text = f"{predicted_class.title()}\nReality: {true_class.title()}\nConfidence: {confidence:.2f}\nIoU: {real_iou:.2f}"
                    
                    # Add label yang TEPAT di atas bounding box dengan SEMUA TEKS HITAM
                    plotter.add_point_labels([label_pos], [label_text],
                                           point_size=0,  # Hide the point
                                           font_size=9,
                                           text_color='black',  # SEMUA TEKS HITAM
                                           shape_color='white',  # Background putih
                                           shape_opacity=0.9,   # Background highlight
                                           margin=4,
                                           show_points=False,
                                           always_visible=True,
                                           fill_shape=True)
                    
                except Exception as e:
                    print(f"⚠️  Error adding bbox {j}: {e}")
                    continue
            
            # CLEAN camera positioning (no clutter)
            camera_distance = np.linalg.norm(scene_bounds) * 1.2
            plotter.camera_position = [(pcd_centroid[0] + camera_distance, 
                                      pcd_centroid[1] + camera_distance, 
                                      pcd_centroid[2] + camera_distance/3),
                                     pcd_centroid,
                                     (0, 0, 1)]
            
            # NO axes, NO title, NO legend, NO help text - CLEAN interface
            plotter.show_axes = False
            plotter.show_grid = False
            
            # Show CLEAN interactive window
            print(f"🎮 Opening clean interactive window for {pcd_file}...")
            print("💡 Clean interface - focus pada point cloud dan predictions")
            print("   Mouse: rotate, wheel: zoom, right-click: pan, 'q': close")
            
            plotter.show(auto_close=False)
            
            print(f"✅ Clean visualization completed untuk {pcd_file}")
            successful_visualizations += 1
            
        except Exception as e:
            print(f"❌ Error creating visualization for {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Clean interactive visualizations completed!")
    print(f"✅ Successfully showed {successful_visualizations}/{len(selected_files)} visualizations")
    
    if successful_visualizations == 0:
        print("⚠️  No successful visualizations created. Please check file paths and data format.")

# ============================================================================
# 10. MAIN PROGRAM
# ============================================================================

def main():
    """
    Main program untuk Hybrid GNN-Transformer-FCN model
    Mengatasi semua inkonsistensi dan mismatch dari gabungan kedua program
    """
    # === CONFIGURATION PATHS ===
    PATHS = {
        'train_json': 'data/train',
        'val_json': 'data/val',
        'test_json': 'data/test',
        'pcd_base': 'data/pcd_test',
        'model_save': 'outputs/checkpoints',
        'results_save': 'outputs/results',
        'images_save': 'outputs/figures'
    }
    
    # Create directories jika belum ada
    for path in [PATHS['results_save'], PATHS['images_save']]:
        os.makedirs(path, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    print("=" * 80)
    print("🚀 HYBRID TRANSFORMER-GNN-FCN MODEL")
    print("🎯 Combining: Transformer attention + GNN spatial reasoning + FCN output")
    print("⚖️  Cost-sensitive learning + Comprehensive evaluation")
    print("🎨 Proven visualization approach from Transformer model")
    print("=" * 80)
    
    try:
        # === 1. LOAD DATASETS DENGAN UNIFIED APPROACH ===
        print("📂 Loading datasets dengan unified approach...")
        train_dataset = UnifiedHybridDataset(PATHS['train_json'], is_training=True)
        val_dataset = UnifiedHybridDataset(PATHS['val_json'], scaler=train_dataset.scaler, is_training=False)
        val_dataset.bbox_scaler = train_dataset.bbox_scaler
        
        # Data loaders dengan consistent batch size
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        classes = ["cyclist", "car", "human", "wall", "tree", "cart"]
        
        # === 2. INITIALIZE HYBRID MODEL (REORDERED) ===
        print("🏗️  Initializing Hybrid Transformer-GNN-FCN model...")
        model = HybridTransformerGNNFCN(
            input_dim=7,           # Consistent dengan proven Transformer approach
            hidden_dim=64,         # Proven dimension dari Transformer model
            num_classes=6,
            num_heads=8,           # Proven dari Transformer model
            num_transformer_layers=3,  # Proven dari Transformer model
            num_gnn_layers=2,      # Reasonable depth untuk GNN
            dropout=0.2            # Proven dropout rate
        )
        
        # === 3. INITIALIZE UNIFIED TRAINER ===
        trainer = UnifiedHybridTrainer(model, device, train_dataset)
        
        # === 4. TRAINING PHASE ===
        print("🎯 Starting hybrid training...")
        model_path = os.path.join(PATHS['model_save'], f"hybrid_transformer_gnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        training_history = trainer.train(train_loader, val_loader, epochs=100, save_path=model_path)
        
        # === 5. TESTING PHASE ===
        print("\n🧪 Starting comprehensive testing...")
        test_dataset = UnifiedHybridDataset(PATHS['test_json'], scaler=train_dataset.scaler, is_training=False)
        test_dataset.bbox_scaler = train_dataset.bbox_scaler
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        # Load best model untuk testing
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            print("✅ Best model loaded untuk testing")
        except Exception as e:
            print(f"⚠️  Using current model for testing: {e}")
        
        # === 6. COMPREHENSIVE EVALUATION ===
        results = evaluate_hybrid_model_comprehensive(model, test_loader, device, train_dataset.bbox_scaler, classes)
        
        # === 7. PRINT HASIL ===
        print("\n📊 HYBRID MODEL RESULTS:")
        print(f"✅ Classification Accuracy: {results['classification']['accuracy']:.4f}")
        print(f"✅ Mean 3D IoU: {results['detection_3d']['mean_iou']:.4f}")
        print(f"✅ Mean Average Precision (mAP): {results['detection_3d']['mAP']:.4f}")
        print(f"✅ Mean Translation Error: {results['detection_3d']['mean_translation_error']:.4f}")
        
        # Per-class results summary
        print("\n📈 PER-CLASS SUMMARY:")
        for class_name in classes:
            iou = results['detection_3d']['per_class_iou'][class_name]
            ap = results['detection_3d']['per_class_ap'][class_name]
            print(f"  {class_name:8s}: IoU={iou:.3f}, AP={ap:.3f}")
        
        # === 8. CREATE COMPREHENSIVE VISUALIZATIONS ===
        print("\n🎨 Creating comprehensive visualizations...")
        timestamp = create_comprehensive_visualizations(training_history, results, PATHS['images_save'])
        
        # === 9. SAVE COMPREHENSIVE RESULTS ===
        excel_path = save_comprehensive_results(training_history, results, PATHS['results_save'], timestamp)
        
        # === 10. INTERACTIVE 3D VISUALIZATIONS ===
        if VISUALIZATION_3D_AVAILABLE and os.path.exists(PATHS['pcd_base']):
            print("\n🎮 Starting INTERACTIVE 3D visualizations...")
            print("💡 Each visualization akan terbuka dalam window terpisah")
            print("   Anda dapat merotasi, zoom, dan explore setiap scene secara interaktif")
            print("   Tutup window untuk melanjutkan ke file berikutnya")
            create_interactive_3d_visualizations(results, PATHS['pcd_base'], PATHS['test_json'], num_files=5)
        else:
            print("⚠️  3D visualizations skipped (libraries not available atau path not found)")
            print(f"🔍 Checked path: {PATHS['pcd_base']}")
            print("💡 Please verify the PCD path is correct")
        
        # === 11. FINAL SUMMARY ===
        print("\n🎉 HYBRID PROGRAM COMPLETED SUCCESSFULLY!")
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Excel results: {excel_path}")
        print(f"✅ Visualizations: {PATHS['images_save']}")
        
        # Performance assessment
        if results['detection_3d']['mean_iou'] > 0.6:
            print("🏆 EXCELLENT! IoU > 0.6 achieved - hybrid approach working excellently!")
        elif results['detection_3d']['mean_iou'] > 0.5:
            print("🥈 VERY GOOD! IoU > 0.5 achieved - solid performance!")
        elif results['detection_3d']['mean_iou'] > 0.4:
            print("🥉 GOOD! Reasonable performance achieved!")
        else:
            print("📊 Performance dapat ditingkatkan dengan hyperparameter tuning")
        
        print("\n💡 Hybrid Model Architecture Summary:")
        print("   🎯 Transformer Branch: Attention-based feature learning (FIRST)")
        print("   🔗 GNN Branch: Spatial relationship modeling (SECOND)")
        print("   🧠 Smart Fusion: Cross-attention untuk combine representations")
        print("   📊 FCN Heads: Classification dan regression output")
        print("   ⚖️  Cost-Sensitive: Handle class imbalance effectively")
        
        print("\n🔧 Technical Specifications:")
        print(f"   📐 Input Features: 7D (x,y,z,w,h,d,rot_z)")
        print(f"   🏗️  Hidden Dimension: 64")
        print(f"   🎯 Transformer Layers: 3 dengan 8 attention heads (FIRST)")
        print(f"   🔗 GNN Layers: 2 dengan residual connections (SECOND)")
        print(f"   💾 Parameters: Proven dari successful models")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting tips:")
        print("   1. Verify all file paths exist dan accessible")
        print("   2. Check GPU memory availability")
        print("   3. Ensure data format consistency")
        print("   4. Consider reducing batch size if memory issues")
        print("   5. Check dependency installations (torch, sklearn, etc.)")

if __name__ == "__main__":
    main()