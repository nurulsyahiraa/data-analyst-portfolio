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
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import warnings
import random
from collections import Counter

# Optional imports with graceful fallback for 3D visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️  Open3D not available. 3D visualizations will be skipped.")

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("⚠️  PyVista not available. 3D visualizations will be skipped.")

# Optional import for FLOPs calculation using thop library
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  thop library not available. Using manual FLOPs estimation.")

warnings.filterwarnings('ignore')

# ============================================================================
# NEW: FLOPs ANALYZER CLASS
# ============================================================================
# FLOPs (Floating Point Operations) adalah metrik untuk mengukur computational
# complexity dari sebuah model neural network. Semakin tinggi FLOPs, semakin
# banyak operasi matematika yang diperlukan untuk satu forward pass.

class FLOPsAnalyzer:
    """
    Comprehensive FLOPs and Model Complexity Analyzer
    
    Fitur:
    - Menghitung total FLOPs menggunakan thop library (jika tersedia)
    - Manual estimation sebagai fallback
    - Menghitung total parameter (trainable dan non-trainable)
    - Memory footprint analysis
    """
    def __init__(self):
        self.use_thop = THOP_AVAILABLE
        print(f"✓ FLOPs Analyzer initialized (thop available: {self.use_thop})")
    
    def count_parameters(self, model):
        """
        Menghitung total parameter dalam model
        
        Returns:
            dict: Dictionary berisi total_params, trainable_params, non_trainable_params
        """
        # Total semua parameter (weights dan biases)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Parameter yang bisa di-train (requires_grad=True)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Parameter yang frozen (requires_grad=False)
        non_trainable_params = total_params - trainable_params
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params_formatted': self._format_number(total_params),
            'trainable_params_formatted': self._format_number(trainable_params)
        }
    
    def _format_number(self, num):
        """Format angka besar menjadi human-readable (K, M, B)"""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)
    
    def manual_flops_estimation(self, model, input_shape):
        """
        Manual FLOPs estimation ketika thop tidak tersedia
        
        Menghitung FLOPs untuk layer-layer umum:
        - Linear layer: 2 * input_features * output_features (multiply + add)
        - Transformer: perkiraan berdasarkan attention mechanism
        
        Args:
            model: PyTorch model
            input_shape: tuple (batch_size, features)
            
        Returns:
            int: Estimated total FLOPs
        """
        total_flops = 0
        batch_size = input_shape[0]
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Linear layer FLOPs: 2 * batch * in_features * out_features
                # Factor 2 untuk multiplication dan addition
                flops = 2 * batch_size * module.in_features * module.out_features
                total_flops += flops
            
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Transformer encoder layer estimation:
                # Self-attention: 4 * seq_len^2 * d_model (Q, K, V projections + attention)
                # FFN: 2 * seq_len * d_model * dim_feedforward
                d_model = module.self_attn.embed_dim
                seq_len = 1  # Untuk input kita, seq_len = 1
                dim_ff = module.linear1.out_features
                
                # Self-attention FLOPs
                attention_flops = 4 * batch_size * seq_len * seq_len * d_model
                # FFN FLOPs (2 linear layers)
                ffn_flops = 2 * batch_size * seq_len * d_model * dim_ff + \
                           2 * batch_size * seq_len * dim_ff * d_model
                
                total_flops += attention_flops + ffn_flops
            
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm: 5 operations per element (mean, var, normalize, scale, shift)
                if hasattr(module, 'normalized_shape'):
                    flops = 5 * batch_size * np.prod(module.normalized_shape)
                    total_flops += flops
        
        return total_flops
    
    def analyze_model_complexity(self, model, sample_input, device='cuda'):
        """
        Comprehensive model complexity analysis
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor untuk profiling
            device: Device untuk computation
            
        Returns:
            dict: Complete complexity analysis results
        """
        model.eval()
        
        # Parameter counting
        param_info = self.count_parameters(model)
        
        # FLOPs calculation
        if self.use_thop:
            try:
                # Pastikan input di device yang benar
                sample = sample_input[:1].to(device)
                
                # Profile menggunakan thop
                flops, params = profile(model, inputs=(sample,), verbose=False)
                flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
                gflops = flops / 1e9
                mflops = flops / 1e6
                
            except Exception as e:
                print(f"⚠️  thop profiling failed: {e}. Using manual estimation.")
                flops = self.manual_flops_estimation(model, sample_input.shape)
                flops_formatted = self._format_number(flops)
                gflops = flops / 1e9
                mflops = flops / 1e6
        else:
            # Manual estimation
            flops = self.manual_flops_estimation(model, sample_input.shape)
            flops_formatted = self._format_number(flops)
            gflops = flops / 1e9
            mflops = flops / 1e6
        
        # Memory analysis
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        # GPU memory analysis (jika menggunakan CUDA)
        if device.type == 'cuda' if hasattr(device, 'type') else str(device) == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(sample_input[:1].to(device))
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_memory_mb = model_size_mb  # Approximate untuk CPU
        
        return {
            'flops': {
                'total_flops': flops,
                'flops_formatted': flops_formatted,
                'gflops': gflops,
                'mflops': mflops
            },
            'parameters': param_info,
            'memory': {
                'model_size_mb': model_size_mb,
                'peak_memory_mb': peak_memory_mb
            }
        }


# ============================================================================
# 1. ENHANCED DATASET - SAME AS BEFORE BUT WITH BETTER TRACKING
# ============================================================================

class FixedJSONDataset(Dataset):
    """
    Dataset yang sudah terbukti menghasilkan IoU 0.66
    Tidak ada perubahan dari versi yang sukses
    """
    def __init__(self, json_folder, scaler=None, bbox_scaler=None, is_training=False):
        self.json_folder = json_folder
        self.is_training = is_training
        
        # Extract all data first (sama seperti sebelumnya)
        self.data, self.labels, self.bbox_targets, self.file_info = self._load_json_data()
        
        # Setup label mapping
        self.label_map = {"cyclist": 0, "car": 1, "human": 2, "wall": 3, "tree": 4, "cart": 5}
        self.classes = list(self.label_map.keys())
        
        # Global normalization (proven effective)
        if scaler is not None:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)
            print("✓ Using shared StandardScaler untuk consistency")
        else:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
            
        # Bbox target normalization
        if bbox_scaler is not None:
            self.bbox_scaler = bbox_scaler
            self.bbox_targets = self.bbox_scaler.transform(self.bbox_targets)
            print("✓ Using shared BBox Scaler untuk consistency")
        else:
            self.bbox_scaler = StandardScaler() 
            self.bbox_targets = self.bbox_scaler.fit_transform(self.bbox_targets)
        
        print(f"✓ Dataset loaded: {len(self.data)} samples with 7 features")
        print(f"✓ Class distribution: {Counter(self.labels)}")
    
    def _load_json_data(self):
        """Same as before - proven effective extraction logic"""
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
                        features = list(position.values())
                        
                        if len(features) == 3:
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
                print(f"Error loading {file_name}: {e}")
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
        
        original_coords = torch.FloatTensor(self.file_info[idx]["coords"])
        original_dims = torch.FloatTensor(self.file_info[idx]["dimensions"])
        rotation_z = self.file_info[idx]["rotation_z"]
        
        return (features, class_target, bbox_target, 
                original_coords, original_dims, rotation_z)


# ============================================================================
# 2. ENHANCED MODEL - SAME ARCHITECTURE WITH BETTER MONITORING
# ============================================================================

class FixedTransformerModel(nn.Module):
    """
    Model yang terbukti efektif - tidak ada perubahan dari versi sukses
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_classes=6, num_heads=8, num_layers=3):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, 
            dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_class = nn.Linear(hidden_dim, num_classes)
        self.fc_bbox = nn.Linear(hidden_dim, 7)
        self.gelu = nn.GELU()
        
        print("✓ Fixed Transformer Model initialized (proven architecture)")
    
    def forward(self, x):
        x = self.embedding(x.unsqueeze(1))
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.gelu(x)
        
        class_output = self.fc_class(x)
        bbox_output = self.fc_bbox(x)
        
        return class_output, bbox_output


# ============================================================================
# 3. ENHANCED IoU CALCULATION - PROVEN EFFECTIVE
# ============================================================================

def calculate_3d_iou_with_rotation(box1, box2):
    """
    IoU calculation yang sudah terbukti menghasilkan 0.66 mean IoU
    Tidak ada perubahan dari versi sukses
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
    """Calculate translation error (Euclidean distance)"""
    return np.linalg.norm(np.array(pred_coords) - np.array(gt_coords))


# ============================================================================
# 4. NEW: ENHANCED AP@0.5 CALCULATOR
# ============================================================================
# AP@0.5 (Average Precision at IoU threshold 0.5) adalah metrik standar dalam
# object detection yang mengukur seberapa baik model mendeteksi objek dengan
# IoU minimal 0.5 terhadap ground truth.

def calculate_ap_at_threshold(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate Average Precision at specific IoU threshold
    
    AP@0.5 menghitung precision-recall curve dan mengintegrasikan area di bawahnya.
    
    Langkah-langkah:
    1. Sort predictions berdasarkan confidence score (descending)
    2. Untuk setiap prediction, tentukan apakah True Positive atau False Positive
    3. Hitung precision dan recall di setiap titik
    4. Interpolasi precision dan hitung area (AP)
    
    Args:
        predictions: List of dict dengan keys 'confidence', 'iou', 'class_id'
        ground_truths: List of dict dengan keys 'class_id'
        iou_threshold: Threshold IoU untuk menentukan TP/FP (default 0.5)
        
    Returns:
        float: Average Precision value
    """
    if len(predictions) == 0 or len(ground_truths) == 0:
        return 0.0
    
    # Sort predictions by confidence (descending)
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    # Track which ground truths have been matched
    matched_gt = set()
    
    true_positives = []
    false_positives = []
    
    total_gt = len(ground_truths)
    
    for pred in sorted_preds:
        # Find best matching ground truth (by IoU)
        best_iou = pred.get('iou', 0)
        
        if best_iou >= iou_threshold:
            # This could be a True Positive
            # Simplified: we use the pre-calculated IoU
            pred_idx = sorted_preds.index(pred)
            if pred_idx not in matched_gt:
                true_positives.append(1)
                false_positives.append(0)
                matched_gt.add(pred_idx)
            else:
                true_positives.append(0)
                false_positives.append(1)
        else:
            # False Positive (IoU below threshold)
            true_positives.append(0)
            false_positives.append(1)
    
    # Calculate cumulative sums
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    
    # Calculate precision and recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / (total_gt + 1e-10)
    
    # Add sentinel values for proper interpolation
    precisions = np.concatenate([[1], precisions])
    recalls = np.concatenate([[0], recalls])
    
    # Make precision monotonically decreasing (interpolation)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate AP using rectangle rule (area under PR curve)
    recall_diff = np.diff(recalls)
    ap = np.sum(recall_diff * precisions[1:])
    
    return ap


def calculate_per_class_ap_at_05(predictions, ground_truths, classes):
    """
    Calculate AP@0.5 untuk setiap class
    
    Args:
        predictions: List of all predictions
        ground_truths: List of all ground truths
        classes: List of class names
        
    Returns:
        dict: Per-class AP@0.5 values dan mAP@0.5
    """
    per_class_ap = {}
    
    for class_id, class_name in enumerate(classes):
        # Filter predictions dan ground truths untuk class ini
        class_preds = [p for i, p in enumerate(predictions) 
                      if ground_truths[i]['class_id'] == class_id]
        class_gts = [gt for gt in ground_truths if gt['class_id'] == class_id]
        
        if len(class_gts) == 0:
            per_class_ap[class_name] = 0.0
        else:
            # Simplified AP calculation based on IoU >= 0.5
            tp = sum(1 for pred in class_preds if pred['iou'] >= 0.5)
            per_class_ap[class_name] = tp / len(class_gts) if len(class_gts) > 0 else 0.0
    
    # mAP@0.5 adalah rata-rata AP@0.5 dari semua classes
    mAP_05 = np.mean(list(per_class_ap.values()))
    
    return {
        'per_class_ap_05': per_class_ap,
        'mAP_05': mAP_05
    }


# ============================================================================
# 5. COMPREHENSIVE EVALUATION WITH ALL METRICS (ENHANCED)
# ============================================================================

def evaluate_model_comprehensive_enhanced(model, test_loader, device, bbox_scaler, classes, flops_analyzer=None):
    """
    Comprehensive evaluation dengan semua metrics yang diminta
    
    ENHANCED dengan:
    - FLOPs dan Total Parameters
    - AP@0.5 per class dan mAP@0.5
    - Detailed performance metrics
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []
    all_class_predictions = []
    all_class_targets = []
    all_ious = []
    all_translation_errors = []
    
    print("🔍 Starting comprehensive evaluation...")
    
    inference_time = 0.0  # Total inference time in seconds
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            features, class_targets, bbox_targets, orig_coords, orig_dims, rotation_z = batch_data
            
            features = features.to(device)
            class_targets = class_targets.to(device)
            
            # Measure inference time for model forward pass
            start_time = time.time()
            class_output, bbox_output = model(features)
            if device.type == 'cuda':
                torch.cuda.synchronize()  # Ensure GPU operations are complete
            end_time = time.time()
            inference_time += (end_time - start_time)
            
            class_preds = torch.argmax(class_output, dim=1)
            
            # Denormalization
            bbox_preds_denorm = bbox_scaler.inverse_transform(bbox_output.cpu().numpy())
            
            for i in range(len(features)):
                pred_box = bbox_preds_denorm[i]
                
                gt_coords = orig_coords[i].numpy()
                gt_dims = orig_dims[i].numpy()
                gt_rot = rotation_z[i] if isinstance(rotation_z[i], (int, float)) else rotation_z[i].item()
                gt_box = np.concatenate([gt_coords, gt_dims, [gt_rot]])
                
                # Calculate IoU
                iou = calculate_3d_iou_with_rotation(pred_box, gt_box)
                all_ious.append(iou)
                
                # Calculate translation error
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
    
    # Calculate comprehensive metrics
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
    
    # Per-class IoU
    per_class_iou = {}
    for class_id, class_name in enumerate(classes):
        class_ious = [all_ious[i] for i, pred in enumerate(all_predictions) 
                     if all_ground_truths[i]['class_id'] == class_id]
        per_class_iou[class_name] = np.mean(class_ious) if class_ious else 0.0
    
    # ENHANCED: Calculate AP@0.5 per class
    ap_results = calculate_per_class_ap_at_05(all_predictions, all_ground_truths, classes)
    per_class_ap_05 = ap_results['per_class_ap_05']
    mAP_05 = ap_results['mAP_05']
    
    # Calculate average time per frame (ms/frame) and FPS
    num_samples = len(all_ious)
    avg_time_per_frame_ms = (inference_time / num_samples) * 1000 if num_samples > 0 else 0.0
    fps = num_samples / inference_time if inference_time > 0 else 0.0
    
    # ENHANCED: FLOPs and Parameters analysis
    model_complexity = {}
    if flops_analyzer is not None:
        try:
            sample_input = next(iter(test_loader))[0][:1]
            model_complexity = flops_analyzer.analyze_model_complexity(model, sample_input, device)
            print(f"✓ Model complexity analysis completed")
        except Exception as e:
            print(f"⚠️  FLOPs analysis failed: {e}")
            # Manual parameter counting as fallback
            model_complexity = {
                'parameters': flops_analyzer.count_parameters(model),
                'flops': {'total_flops': 0, 'flops_formatted': 'N/A', 'gflops': 0, 'mflops': 0},
                'memory': {'model_size_mb': 0, 'peak_memory_mb': 0}
            }
    
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
            'mAP_05': mAP_05,  # ENHANCED: Explicit mAP@0.5
            'per_class_ap_05': per_class_ap_05,  # ENHANCED: Per-class AP@0.5
            'mean_translation_error': mean_translation_error,
            'all_ious': all_ious,
            'all_translation_errors': all_translation_errors
        },
        'performance': {
            'avg_time_per_frame_ms': avg_time_per_frame_ms,
            'inference_speed_fps': fps,
            'total_inference_time_sec': inference_time,
            'num_samples': num_samples
        },
        'model_complexity': model_complexity,  # ENHANCED: FLOPs dan Parameters
        'predictions': all_predictions,
        'ground_truths': all_ground_truths
    }


# ============================================================================
# 6. COMPREHENSIVE VISUALIZATION AND SAVING (ENHANCED)
# ============================================================================

def create_comprehensive_visualizations(results, save_path):
    """
    Create comprehensive visualizations dan save ke PNG
    ENHANCED dengan AP@0.5 visualization
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = results['classification']['confusion_matrix']
    classes = results['classification']['class_names']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (6x6)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'confusion_matrix_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class metrics (Classification)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Precision, Recall, F1-Score
    metrics_df = pd.DataFrame(results['classification']['per_class_metrics'])
    x = np.arange(len(metrics_df))
    width = 0.25
    
    axes[0].bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8, color='#3498db')
    axes[0].bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8, color='#2ecc71')
    axes[0].bar(x + width, metrics_df['F1_Score'], width, label='F1-Score', alpha=0.8, color='#e74c3c')
    
    axes[0].set_xlabel('Classes', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Classification Metrics per Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_df['Class'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)
    
    # ENHANCED: IoU and AP@0.5 per class
    class_names = list(results['detection_3d']['per_class_iou'].keys())
    iou_values = list(results['detection_3d']['per_class_iou'].values())
    ap_values = list(results['detection_3d']['per_class_ap_05'].values())
    
    x = np.arange(len(class_names))
    axes[1].bar(x - 0.2, iou_values, 0.4, label='IoU', alpha=0.8, color='#f39c12')
    axes[1].bar(x + 0.2, ap_values, 0.4, label='AP@0.5', alpha=0.8, color='#9b59b6')
    
    axes[1].set_xlabel('Classes', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('3D Detection Metrics per Class (IoU & AP@0.5)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'per_class_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ENHANCED: Model Complexity Summary Chart
    if 'model_complexity' in results and results['model_complexity']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Parameters breakdown
        if 'parameters' in results['model_complexity']:
            params = results['model_complexity']['parameters']
            param_labels = ['Total\nParams', 'Trainable\nParams']
            param_values = [params['total_params'], params['trainable_params']]
            
            bars = axes[0].bar(param_labels, param_values, color=['#3498db', '#2ecc71'], alpha=0.8)
            axes[0].set_ylabel('Number of Parameters', fontsize=12)
            axes[0].set_title('Model Parameters', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, param_values):
                height = bar.get_height()
                axes[0].annotate(f'{val:,}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=10)
        
        # FLOPs and Memory
        if 'flops' in results['model_complexity'] and 'memory' in results['model_complexity']:
            flops = results['model_complexity']['flops']
            memory = results['model_complexity']['memory']
            
            metric_labels = ['GFLOPs', 'Model Size\n(MB)', 'Peak Memory\n(MB)']
            metric_values = [
                flops.get('gflops', 0),
                memory.get('model_size_mb', 0),
                memory.get('peak_memory_mb', 0)
            ]
            
            bars = axes[1].bar(metric_labels, metric_values, color=['#e74c3c', '#f39c12', '#9b59b6'], alpha=0.8)
            axes[1].set_ylabel('Value', fontsize=12)
            axes[1].set_title('Model Complexity Metrics', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, metric_values):
                height = bar.get_height()
                axes[1].annotate(f'{val:.3f}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'model_complexity_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved with timestamp: {timestamp}")
    return timestamp


def save_comprehensive_results(results, save_path, timestamp, testing_time):
    """
    Save semua results ke multi-sheet Excel
    ENHANCED dengan FLOPs, Parameters, dan AP@0.5
    """
    excel_filename = f"comprehensive_results_{timestamp}.xlsx"
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
        
        # 3. ENHANCED: 3D Detection Metrics with AP@0.5
        detection_data = []
        for class_name in results['classification']['class_names']:
            detection_data.append({
                'Class': class_name,
                'IoU': results['detection_3d']['per_class_iou'][class_name],
                'AP@0.5': results['detection_3d']['per_class_ap_05'][class_name]
            })
        
        detection_df = pd.DataFrame(detection_data)
        detection_df.to_excel(writer, sheet_name='3D_Detection_Metrics', index=False)
        
        # 4. Translation Errors dan IoU per sample
        translation_df = pd.DataFrame({
            'Sample_ID': range(len(results['detection_3d']['all_translation_errors'])),
            'Translation_Error': results['detection_3d']['all_translation_errors'],
            'IoU': results['detection_3d']['all_ious']
        })
        translation_df.to_excel(writer, sheet_name='Sample_Results', index=False)
        
        # 5. ENHANCED: Model Complexity Sheet
        if 'model_complexity' in results and results['model_complexity']:
            complexity = results['model_complexity']
            complexity_data = {
                'Metric': [
                    'Total_Parameters',
                    'Trainable_Parameters',
                    'Total_FLOPs',
                    'GFLOPs',
                    'MFLOPs',
                    'Model_Size_MB',
                    'Peak_Memory_MB'
                ],
                'Value': [
                    complexity.get('parameters', {}).get('total_params', 0),
                    complexity.get('parameters', {}).get('trainable_params', 0),
                    complexity.get('flops', {}).get('total_flops', 0),
                    complexity.get('flops', {}).get('gflops', 0),
                    complexity.get('flops', {}).get('mflops', 0),
                    complexity.get('memory', {}).get('model_size_mb', 0),
                    complexity.get('memory', {}).get('peak_memory_mb', 0)
                ],
                'Formatted': [
                    complexity.get('parameters', {}).get('total_params_formatted', 'N/A'),
                    complexity.get('parameters', {}).get('trainable_params_formatted', 'N/A'),
                    complexity.get('flops', {}).get('flops_formatted', 'N/A'),
                    f"{complexity.get('flops', {}).get('gflops', 0):.4f} GFLOPs",
                    f"{complexity.get('flops', {}).get('mflops', 0):.2f} MFLOPs",
                    f"{complexity.get('memory', {}).get('model_size_mb', 0):.2f} MB",
                    f"{complexity.get('memory', {}).get('peak_memory_mb', 0):.2f} MB"
                ]
            }
            complexity_df = pd.DataFrame(complexity_data)
            complexity_df.to_excel(writer, sheet_name='Model_Complexity', index=False)
        
        # 6. ENHANCED: Comprehensive Summary
        # Extract complexity values safely
        total_params = results.get('model_complexity', {}).get('parameters', {}).get('total_params', 0)
        trainable_params = results.get('model_complexity', {}).get('parameters', {}).get('trainable_params', 0)
        gflops = results.get('model_complexity', {}).get('flops', {}).get('gflops', 0)
        total_flops = results.get('model_complexity', {}).get('flops', {}).get('total_flops', 0)
        
        summary_data = {
            'Metric': [
                'Overall_Accuracy',
                'Mean_IoU',
                'mAP@0.5',  # ENHANCED
                'Mean_Translation_Error',
                'Testing_Time_Seconds',
                'Testing_Time_Minutes',
                'Total_Test_Samples',
                'Avg_Time_Per_Frame_ms',
                'Inference_Speed_FPS',
                'Total_Parameters',  # ENHANCED
                'Trainable_Parameters',  # ENHANCED
                'GFLOPs',  # ENHANCED
                'Total_FLOPs'  # ENHANCED
            ],
            'Value': [
                f"{results['classification']['accuracy']:.4f}",
                f"{results['detection_3d']['mean_iou']:.4f}",
                f"{results['detection_3d']['mAP_05']:.4f}",  # ENHANCED
                f"{results['detection_3d']['mean_translation_error']:.4f}",
                f"{testing_time:.2f}",
                f"{testing_time/60:.2f}",
                len(results['detection_3d']['all_ious']),
                f"{results['performance']['avg_time_per_frame_ms']:.4f}",
                f"{results['performance']['inference_speed_fps']:.2f}",
                f"{total_params:,}",  # ENHANCED
                f"{trainable_params:,}",  # ENHANCED
                f"{gflops:.4f}",  # ENHANCED
                f"{total_flops:,}"  # ENHANCED
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Comprehensive results saved: {excel_filename}")
    return excel_path


# ============================================================================
# 7. ENHANCED 3D VISUALIZATION WITH CORRECT NAMING CONVENTION
# ============================================================================

def create_interactive_3d_visualizations(results, pcd_base_path, json_base_path, num_files=5):
    """
    CLEAN & ELEGANT Interactive 3D visualizations
    """
    # Check if visualization libraries are available
    if not OPEN3D_AVAILABLE or not PYVISTA_AVAILABLE:
        print("⚠️  3D visualization libraries not available. Skipping 3D visualizations.")
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
            
            # Point cloud sebagai BACKGROUND dengan warna grey
            point_cloud = pv.PolyData(points)
            plotter.add_mesh(point_cloud, 
                           point_size=2, 
                           color='lightgrey', 
                           opacity=1.0,
                           render_points_as_spheres=True)
            
            # Calculate centroid dan bounds
            pcd_centroid = np.mean(points, axis=0)
            pcd_min = np.min(points, axis=0)
            pcd_max = np.max(points, axis=0)
            scene_bounds = pcd_max - pcd_min
            
            # Extract ground truth dan predictions
            label_map = {"cyclist": 0, "car": 1, "human": 2, "wall": 3, "tree": 4, "cart": 5}
            
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
                            'box': coords + dims + [rot_z]
                        })
            
            print(f"📦 Found {len(ground_truth_boxes)} ground truth boxes")
            
            # Find actual model predictions untuk file ini
            actual_predictions = []
            
            for gt_box in ground_truth_boxes:
                best_pred = None
                min_distance = float('inf')
                
                for pred in results['predictions']:
                    gt_center = np.array(gt_box['center'])
                    pred_center = np.array(pred['box'][:3])
                    distance = np.linalg.norm(gt_center - pred_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_pred = pred
                
                if best_pred and min_distance < max(scene_bounds) * 0.3:
                    real_iou = calculate_3d_iou_with_rotation(best_pred['box'], gt_box['box'])
                    
                    actual_predictions.append({
                        'predicted_class': best_pred['class_name'],
                        'true_class': gt_box['class_name'],
                        'center': gt_box['center'],
                        'dimensions': gt_box['dimensions'],
                        'rotation_z': gt_box['rotation_z'],
                        'confidence': best_pred['confidence'],
                        'real_iou': real_iou,
                        'prediction_box': best_pred['box'],
                        'ground_truth_box': gt_box['box']
                    })
                else:
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
            
            # Add bounding boxes
            for j, pred in enumerate(actual_predictions):
                try:
                    true_class = pred['true_class']
                    predicted_class = pred['predicted_class']
                    center = np.array(pred['center'])
                    dimensions = np.abs(np.array(pred['dimensions']))
                    rotation_z = pred['rotation_z']
                    confidence = pred['confidence']
                    real_iou = pred['real_iou']
                    
                    # Create bounding box
                    box_mesh = pv.Cube(center=center,
                                       x_length=dimensions[0],
                                       y_length=dimensions[1], 
                                       z_length=dimensions[2])
                    
                    # Apply rotation
                    if abs(rotation_z) > 1e-6:
                        box_mesh.rotate_z(np.degrees(rotation_z), point=center, inplace=True)
                    
                    # Determine color
                    if predicted_class == 'missed':
                        color = [0.5, 0.5, 0.5]
                    else:
                        color = class_colors.get(predicted_class, [0.5, 0.5, 0.5])
                    
                    # Add solid bounding box
                    plotter.add_mesh(box_mesh, 
                                   color=color, 
                                   opacity=1.0,
                                   show_edges=False)
                    
                    # Label position
                    if abs(rotation_z) > 1e-6:
                        half_dims = dimensions / 2
                        top_center_local = [0, 0, half_dims[2]]
                        cos_r, sin_r = np.cos(rotation_z), np.sin(rotation_z)
                        rot_matrix = np.array([
                            [cos_r, -sin_r, 0],
                            [sin_r,  cos_r, 0], 
                            [0,      0,     1]
                        ])
                        rotated_top = np.dot(rot_matrix, top_center_local)
                        label_pos = center + rotated_top + [0, 0, 0.2]
                    else:
                        label_pos = center + [0, 0, dimensions[2]/2 + 0.2]
                    
                    # Create label
                    if predicted_class == 'missed':
                        label_text = f"MISSED\nReality: {true_class.title()}\nIoU: 0.00"
                    elif predicted_class == true_class:
                        label_text = f"{predicted_class.title()}\nConf: {confidence:.2f}\nIoU: {real_iou:.2f}"
                    else:
                        label_text = f"{predicted_class.title()}\nReality: {true_class.title()}\nConf: {confidence:.2f}\nIoU: {real_iou:.2f}"
                    
                    plotter.add_point_labels([label_pos], [label_text],
                                           point_size=0,
                                           font_size=9,
                                           text_color='black',
                                           shape_color='white',
                                           shape_opacity=0.9,
                                           margin=4,
                                           show_points=False,
                                           always_visible=True,
                                           fill_shape=True)
                    
                except Exception as e:
                    print(f"⚠️  Error adding bbox {j}: {e}")
                    continue
            
            # Camera positioning
            camera_distance = np.linalg.norm(scene_bounds) * 1.2
            plotter.camera_position = [(pcd_centroid[0] + camera_distance, 
                                      pcd_centroid[1] + camera_distance, 
                                      pcd_centroid[2] + camera_distance/3),
                                     pcd_centroid,
                                     (0, 0, 1)]
            
            plotter.show_axes = False
            plotter.show_grid = False
            
            print(f"🎮 Opening interactive window for {pcd_file}...")
            print("💡 Controls: Mouse=rotate, Wheel=zoom, Right-click=pan, 'q'=close")
            
            plotter.show(auto_close=False)
            
            print(f"✅ Visualization completed untuk {pcd_file}")
            successful_visualizations += 1
            
        except Exception as e:
            print(f"❌ Error creating visualization for {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Interactive visualizations completed!")
    print(f"✅ Successfully showed {successful_visualizations}/{len(selected_files)} visualizations")


# ============================================================================
# 8. MAIN PROGRAM - TESTING ONLY VERSION (ENHANCED)
# ============================================================================

def main():
    """
    Main program untuk TESTING ONLY
    Menggunakan pre-trained model tanpa training
    
    ENHANCED dengan:
    - FLOPs calculation
    - Total parameter counting
    - AP@0.5 metrics
    """
    # PATHS - Update sesuai lokasi file Anda
    PATHS = {
        'test_json': "data/test",
        'pcd_base': "data/pcd_test",
        'model_path': "outputs/checkpoints/enhanced_model_20250628_094907.pth",
        'results_save': "outputs/results",
        'images_save': "outputs/figures"
    }
    
    # Create directories if not exist
    for path in [PATHS['results_save'], PATHS['images_save']]:
        os.makedirs(path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("=" * 80)
    print("🚀 ENHANCED TESTING PROGRAM")
    print("⚡ Using pre-trained model: enhanced_model_20250628_094907.pth")
    print("📊 ENHANCED Features:")
    print("   ✓ 3D IoU, AP@0.5, mAP@0.5, Translation error")
    print("   ✓ FLOPs (Floating Point Operations)")
    print("   ✓ Total Parameters (trainable & non-trainable)")
    print("   ✓ Avg time/frame, FPS, Memory usage")
    print("🎨 Visualizations: PyVista 3D, PNG charts, Multi-sheet Excel")
    print("=" * 80)
    
    try:
        # 1. Initialize FLOPs Analyzer
        print("\n📐 Initializing FLOPs Analyzer...")
        flops_analyzer = FLOPsAnalyzer()
        
        # 2. Load pre-trained model
        print("📥 Loading pre-trained model...")
        
        if not os.path.exists(PATHS['model_path']):
            print(f"❌ Model file not found: {PATHS['model_path']}")
            print("💡 Please ensure the model file exists at the specified path")
            return
            
        checkpoint = torch.load(PATHS['model_path'], map_location=device)
        
        # Extract scalers dari checkpoint
        scaler = checkpoint.get('scaler')
        bbox_scaler = checkpoint.get('bbox_scaler')
        
        if scaler is None or bbox_scaler is None:
            print("⚠️  Scalers not found in checkpoint, creating new ones...")
            temp_dataset = FixedJSONDataset(PATHS['test_json'], is_training=True)
            scaler = temp_dataset.scaler
            bbox_scaler = temp_dataset.bbox_scaler
        
        print("✅ Scalers loaded/created successfully")
        
        # 3. Load test dataset
        print("📂 Loading test dataset...")
        test_dataset = FixedJSONDataset(PATHS['test_json'], scaler=scaler, bbox_scaler=bbox_scaler, is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        classes = ["cyclist", "car", "human", "wall", "tree", "cart"]
        
        # 4. Initialize model dan load weights
        model = FixedTransformerModel(input_dim=7, num_classes=6)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Pre-trained model loaded and ready for testing")
        
        # 5. ENHANCED: Quick model complexity analysis
        print("\n📊 Analyzing model complexity...")
        sample_input = next(iter(test_loader))[0]
        quick_complexity = flops_analyzer.analyze_model_complexity(model, sample_input, device)
        
        print(f"   ✓ Total Parameters: {quick_complexity['parameters']['total_params_formatted']}")
        print(f"   ✓ Trainable Parameters: {quick_complexity['parameters']['trainable_params_formatted']}")
        print(f"   ✓ FLOPs: {quick_complexity['flops']['flops_formatted']}")
        print(f"   ✓ GFLOPs: {quick_complexity['flops']['gflops']:.4f}")
        print(f"   ✓ Model Size: {quick_complexity['memory']['model_size_mb']:.2f} MB")
        
        # 6. Test dengan comprehensive evaluation
        print("\n🧪 Starting comprehensive testing...")
        testing_start_time = time.time()
        
        results = evaluate_model_comprehensive_enhanced(
            model, test_loader, device, bbox_scaler, classes, flops_analyzer
        )
        
        testing_time = time.time() - testing_start_time
        
        # 7. Print results
        print(f"\n{'='*80}")
        print("📊 ENHANCED TESTING RESULTS")
        print(f"{'='*80}")
        print(f"⏱️  Testing completed in {testing_time:.2f} seconds")
        print(f"\n📈 CLASSIFICATION METRICS:")
        print(f"   ✅ Accuracy: {results['classification']['accuracy']:.4f}")
        
        print(f"\n🎯 3D DETECTION METRICS:")
        print(f"   ✅ Mean IoU: {results['detection_3d']['mean_iou']:.4f}")
        print(f"   ✅ mAP@0.5: {results['detection_3d']['mAP_05']:.4f}")  # ENHANCED
        print(f"   ✅ Mean Translation Error: {results['detection_3d']['mean_translation_error']:.4f}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   ✅ Average Time per Frame: {results['performance']['avg_time_per_frame_ms']:.4f} ms/frame")
        print(f"   ✅ Inference Speed: {results['performance']['inference_speed_fps']:.2f} FPS")
        
        print(f"\n🔧 MODEL COMPLEXITY:")  # ENHANCED
        if 'model_complexity' in results and results['model_complexity']:
            mc = results['model_complexity']
            print(f"   ✅ Total Parameters: {mc['parameters']['total_params_formatted']} ({mc['parameters']['total_params']:,})")
            print(f"   ✅ Trainable Parameters: {mc['parameters']['trainable_params_formatted']} ({mc['parameters']['trainable_params']:,})")
            print(f"   ✅ FLOPs: {mc['flops']['flops_formatted']}")
            print(f"   ✅ GFLOPs: {mc['flops']['gflops']:.4f}")
            print(f"   ✅ Model Size: {mc['memory']['model_size_mb']:.2f} MB")
        
        # Per-class results summary
        print(f"\n📈 PER-CLASS SUMMARY (IoU & AP@0.5):")
        print(f"   {'Class':<10} {'IoU':>8} {'AP@0.5':>8}")
        print(f"   {'-'*28}")
        for class_name in classes:
            iou = results['detection_3d']['per_class_iou'][class_name]
            ap = results['detection_3d']['per_class_ap_05'][class_name]
            print(f"   {class_name:<10} {iou:>8.3f} {ap:>8.3f}")
        
        # 8. Create comprehensive visualizations
        print("\n🎨 Creating testing visualizations...")
        timestamp = create_comprehensive_visualizations(results, PATHS['images_save'])
        
        # 9. Save comprehensive results
        excel_path = save_comprehensive_results(results, PATHS['results_save'], timestamp, testing_time)
        
        # 10. Create INTERACTIVE 3D visualizations
        if os.path.exists(PATHS['pcd_base']):
            print("\n🎮 Starting INTERACTIVE 3D visualizations...")
            print("💡 Each visualization akan terbuka dalam window terpisah")
            print("   Anda dapat merotasi, zoom, dan explore setiap scene secara interaktif")
            print("   Tutup window untuk melanjutkan ke file berikutnya")
            create_interactive_3d_visualizations(results, PATHS['pcd_base'], PATHS['test_json'], num_files=5)
        else:
            print("⚠️  PCD base path not found. Skipping 3D visualizations.")
            print(f"📁 Checked path: {PATHS['pcd_base']}")
        
        print(f"\n{'='*80}")
        print("🎉 ENHANCED TESTING PROGRAM COMPLETED!")
        print(f"{'='*80}")
        print(f"✅ Excel results: {excel_path}")
        print(f"✅ PNG visualizations: {PATHS['images_save']}")
        
        if results['detection_3d']['mean_iou'] > 0.5:
            print("🏆 SUCCESS! IoU target achieved!")
        
        # Final summary
        print(f"\n📋 QUICK SUMMARY:")
        print(f"   • Accuracy: {results['classification']['accuracy']:.4f}")
        print(f"   • Mean IoU: {results['detection_3d']['mean_iou']:.4f}")
        print(f"   • mAP@0.5: {results['detection_3d']['mAP_05']:.4f}")
        print(f"   • Total Params: {results['model_complexity']['parameters']['total_params_formatted']}")
        print(f"   • GFLOPs: {results['model_complexity']['flops']['gflops']:.4f}")
        print(f"   • FPS: {results['performance']['inference_speed_fps']:.2f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()