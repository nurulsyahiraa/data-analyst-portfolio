"""
Enhanced Testing Model 4 - Hybrid GNN-Transformer FCN
=====================================================
Enhancements:
1. FLOPs Calculation - Model computational complexity analysis
2. Total Model Parameters - Detailed parameter counting
3. IoU Threshold 0.5 - Standard detection evaluation metrics
4. Failure Case Analysis - Comprehensive error analysis

Author: Enhanced version based on original testing_model_4.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import time
import warnings
from datetime import datetime

# ============================================================================
# OPTIONAL IMPORTS WITH GRACEFUL FALLBACK
# ============================================================================

# Check for torch_geometric availability
try:
    from torch_geometric.nn import GCNConv
    TORCH_GEOMETRIC_AVAILABLE = True
    print("✓ PyTorch Geometric available - using advanced GCNConv")
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠ PyTorch Geometric not available - using simplified GNN")

# Check for thop (FLOPs calculation library)
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
    print("✓ THOP available - using advanced FLOPs calculation")
except ImportError:
    THOP_AVAILABLE = False
    print("⚠ THOP not available - using manual FLOPs estimation")

warnings.filterwarnings('ignore')

# ============================================================================
# 1. FLOPs AND MODEL COMPLEXITY ANALYZER
# ============================================================================

class ModelComplexityAnalyzer:
    """
    Comprehensive model complexity analyzer that calculates:
    - FLOPs (Floating Point Operations)
    - Total and trainable parameters
    - Memory footprint
    - Layer-wise breakdown
    
    This analyzer works with or without the THOP library, providing
    manual estimation as a fallback for accurate complexity assessment.
    """
    
    def __init__(self):
        self.use_thop = THOP_AVAILABLE
        print(f"✓ ModelComplexityAnalyzer initialized (THOP: {self.use_thop})")
    
    def count_parameters(self, model):
        """
        Count total and trainable parameters in the model.
        
        Returns a detailed breakdown including:
        - Total parameters across all layers
        - Trainable parameters (requires_grad=True)
        - Non-trainable/frozen parameters
        - Layer-wise parameter distribution
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        layer_params = {}
        
        # Iterate through all named parameters for detailed breakdown
        for name, param in model.named_parameters():
            num_params = param.numel()  # Number of elements in tensor
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
            else:
                non_trainable_params += num_params
            
            # Group parameters by layer type for analysis
            layer_type = name.split('.')[0]  # Get top-level module name
            if layer_type not in layer_params:
                layer_params[layer_type] = {'total': 0, 'trainable': 0}
            layer_params[layer_type]['total'] += num_params
            if param.requires_grad:
                layer_params[layer_type]['trainable'] += num_params
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'layer_breakdown': layer_params,
            'total_params_millions': total_params / 1e6,
            'trainable_params_millions': trainable_params / 1e6
        }
    
    def estimate_flops_manual(self, model, input_size=7):
        """
        Manual FLOPs estimation when THOP is not available.
        
        This method estimates FLOPs by analyzing each layer type:
        - Linear layers: 2 * in_features * out_features (multiply-add)
        - Conv layers: 2 * kernel_size * in_channels * out_channels * output_size
        - BatchNorm: 2 * features (for normalization operations)
        - Attention: O(n^2 * d) for self-attention mechanisms
        """
        total_flops = 0
        flops_breakdown = {}
        
        for name, module in model.named_modules():
            module_flops = 0
            
            if isinstance(module, nn.Linear):
                # FLOPs for linear layer: 2 * in * out (multiply + add)
                module_flops = 2 * module.in_features * module.out_features
                if module.bias is not None:
                    module_flops += module.out_features  # Add bias
                    
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm: mean, variance, normalize, scale, shift
                # Approximately 5 operations per element
                module_flops = 5 * module.normalized_shape[0]
                
            elif isinstance(module, nn.MultiheadAttention):
                # Attention FLOPs: Q*K^T + softmax + attention*V
                # Simplified: 4 * embed_dim^2 for each attention head
                embed_dim = module.embed_dim
                num_heads = module.num_heads
                module_flops = 4 * embed_dim * embed_dim
                
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Transformer layer includes self-attention + FFN
                d_model = module.self_attn.embed_dim
                dim_feedforward = module.linear1.out_features
                # Self-attention: 4 * d^2, FFN: 2 * d * ff * 2
                module_flops = 4 * d_model * d_model + 4 * d_model * dim_feedforward
            
            if module_flops > 0:
                layer_type = type(module).__name__
                if layer_type not in flops_breakdown:
                    flops_breakdown[layer_type] = 0
                flops_breakdown[layer_type] += module_flops
                total_flops += module_flops
        
        return {
            'total_flops': total_flops,
            'gflops': total_flops / 1e9,
            'mflops': total_flops / 1e6,
            'flops_breakdown': flops_breakdown,
            'flops_formatted': f"{total_flops:,}"
        }
    
    def calculate_flops_thop(self, model, sample_input, device):
        """
        Calculate FLOPs using THOP library for accurate measurement.
        
        THOP provides more accurate FLOPs calculation by actually
        tracing the forward pass and counting operations.
        """
        model.eval()
        try:
            # Create dummy batch indices for the model
            batch_size = sample_input.size(0)
            batch_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Profile the model - need custom input handling for this model
            # Since the model takes (x, batch), we need manual profiling
            flops, params = profile(model, inputs=(sample_input, batch_indices), verbose=False)
            flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
            
            return {
                'total_flops': flops,
                'gflops': flops / 1e9,
                'mflops': flops / 1e6,
                'flops_formatted': flops_formatted,
                'params_formatted': params_formatted
            }
        except Exception as e:
            print(f"⚠ THOP profiling failed: {e}, falling back to manual estimation")
            return self.estimate_flops_manual(model)
    
    def analyze_model(self, model, sample_input, device):
        """
        Comprehensive model analysis combining all metrics.
        
        This is the main entry point that returns complete complexity analysis
        including parameters, FLOPs, memory, and detailed breakdowns.
        """
        # Parameter analysis
        param_analysis = self.count_parameters(model)
        
        # FLOPs analysis - try THOP first, fallback to manual
        if self.use_thop:
            try:
                flops_analysis = self.calculate_flops_thop(model, sample_input, device)
            except:
                flops_analysis = self.estimate_flops_manual(model)
        else:
            flops_analysis = self.estimate_flops_manual(model)
        
        # Memory analysis
        model_memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        total_memory_mb = (model_memory_bytes + buffer_memory_bytes) / (1024 ** 2)
        
        # GPU memory if available
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            model.eval()
            with torch.no_grad():
                batch_indices = torch.zeros(sample_input.size(0), dtype=torch.long, device=device)
                _ = model(sample_input, batch_indices)
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_gpu_memory_mb = 0.0
        
        return {
            'parameters': param_analysis,
            'flops': flops_analysis,
            'memory': {
                'model_size_mb': total_memory_mb,
                'peak_gpu_memory_mb': peak_gpu_memory_mb,
                'model_memory_bytes': model_memory_bytes,
                'buffer_memory_bytes': buffer_memory_bytes
            }
        }


# ============================================================================
# 2. FAILURE CASE ANALYZER
# ============================================================================

class FailureCaseAnalyzer:
    """
    Comprehensive failure case analysis for 3D object detection.
    
    This analyzer identifies and categorizes different types of detection
    failures to help understand model weaknesses and guide improvements.
    
    Failure categories include:
    - Classification errors (wrong class prediction)
    - Localization errors (correct class, poor IoU)
    - False positives (detected non-existent objects)
    - False negatives (missed objects)
    - Confidence calibration issues
    """
    
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.5):
        """
        Initialize the failure analyzer with detection thresholds.
        
        Args:
            iou_threshold: Minimum IoU to consider a detection as correct (default 0.5)
            confidence_threshold: Minimum confidence for valid detection (default 0.5)
        """
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        print(f"✓ FailureCaseAnalyzer initialized (IoU threshold: {iou_threshold})")
    
    def analyze_failures(self, predictions, ground_truths, classes):
        """
        Perform comprehensive failure analysis on detection results.
        
        This method categorizes all detections into success/failure types
        and provides detailed statistics for each failure category.
        """
        # Initialize counters and storage
        failure_cases = {
            'classification_errors': [],      # Wrong class predicted
            'localization_errors': [],        # Right class, low IoU
            'low_confidence_correct': [],     # Correct but low confidence
            'high_confidence_wrong': [],      # Wrong with high confidence
            'severe_localization': [],        # IoU < 0.25 (very poor)
            'boundary_cases': []              # IoU near threshold (0.45-0.55)
        }
        
        # Per-class failure statistics
        per_class_failures = {cls: {
            'total': 0,
            'classification_errors': 0,
            'localization_errors': 0,
            'successful': 0,
            'ious': [],
            'confidence_scores': []
        } for cls in classes}
        
        # Confusion analysis for misclassifications
        confusion_pairs = defaultdict(int)  # (true_class, pred_class) -> count
        
        # Analyze each prediction
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            pred_class = pred['class_name']
            true_class = gt['class_name']
            iou = pred['iou']
            confidence = pred['confidence']
            
            # Update per-class statistics
            per_class_failures[true_class]['total'] += 1
            per_class_failures[true_class]['ious'].append(iou)
            per_class_failures[true_class]['confidence_scores'].append(confidence)
            
            # Case 1: Classification Error (wrong class)
            if pred_class != true_class:
                failure_cases['classification_errors'].append({
                    'index': i,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'iou': iou,
                    'confidence': confidence,
                    'pred_box': pred['box'],
                    'gt_box': gt['box'],
                    'translation_error': pred['translation_error']
                })
                per_class_failures[true_class]['classification_errors'] += 1
                confusion_pairs[(true_class, pred_class)] += 1
                
                # High confidence wrong prediction (concerning)
                if confidence >= self.confidence_threshold:
                    failure_cases['high_confidence_wrong'].append({
                        'index': i,
                        'true_class': true_class,
                        'predicted_class': pred_class,
                        'confidence': confidence,
                        'iou': iou
                    })
            
            # Case 2: Correct class but localization issues
            elif pred_class == true_class:
                if iou < self.iou_threshold:
                    # Localization error
                    failure_cases['localization_errors'].append({
                        'index': i,
                        'class': true_class,
                        'iou': iou,
                        'confidence': confidence,
                        'pred_box': pred['box'],
                        'gt_box': gt['box'],
                        'translation_error': pred['translation_error'],
                        'position_error': np.linalg.norm(pred['box'][:3] - gt['box'][:3]),
                        'size_error': np.linalg.norm(pred['box'][3:6] - gt['box'][3:6])
                    })
                    per_class_failures[true_class]['localization_errors'] += 1
                    
                    # Severe localization error
                    if iou < 0.25:
                        failure_cases['severe_localization'].append({
                            'index': i,
                            'class': true_class,
                            'iou': iou,
                            'translation_error': pred['translation_error']
                        })
                else:
                    # Successful detection
                    per_class_failures[true_class]['successful'] += 1
                    
                    # Low confidence correct (underconfident)
                    if confidence < self.confidence_threshold:
                        failure_cases['low_confidence_correct'].append({
                            'index': i,
                            'class': true_class,
                            'iou': iou,
                            'confidence': confidence
                        })
            
            # Case 3: Boundary cases (near IoU threshold)
            if 0.45 <= iou <= 0.55 and pred_class == true_class:
                failure_cases['boundary_cases'].append({
                    'index': i,
                    'class': true_class,
                    'iou': iou,
                    'confidence': confidence
                })
        
        # Calculate summary statistics
        summary = self._calculate_failure_summary(failure_cases, per_class_failures, 
                                                   confusion_pairs, len(predictions))
        
        return {
            'failure_cases': failure_cases,
            'per_class_failures': per_class_failures,
            'confusion_pairs': dict(confusion_pairs),
            'summary': summary
        }
    
    def _calculate_failure_summary(self, failure_cases, per_class_failures, 
                                    confusion_pairs, total_predictions):
        """
        Calculate summary statistics from failure analysis.
        """
        # Overall failure rates
        total_classification_errors = len(failure_cases['classification_errors'])
        total_localization_errors = len(failure_cases['localization_errors'])
        total_failures = total_classification_errors + total_localization_errors
        
        # Find most common confusion pairs
        sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        top_confusions = sorted_confusions[:5] if sorted_confusions else []
        
        # Per-class failure rates
        class_failure_rates = {}
        for cls, stats in per_class_failures.items():
            if stats['total'] > 0:
                class_failure_rates[cls] = {
                    'classification_error_rate': stats['classification_errors'] / stats['total'],
                    'localization_error_rate': stats['localization_errors'] / stats['total'],
                    'success_rate': stats['successful'] / stats['total'],
                    'mean_iou': np.mean(stats['ious']) if stats['ious'] else 0.0,
                    'mean_confidence': np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0.0,
                    'total_samples': stats['total']
                }
        
        # Identify hardest classes (highest failure rate)
        hardest_classes = sorted(
            [(cls, rates['success_rate']) for cls, rates in class_failure_rates.items()],
            key=lambda x: x[1]
        )[:3]
        
        return {
            'total_predictions': total_predictions,
            'total_failures': total_failures,
            'overall_failure_rate': total_failures / total_predictions if total_predictions > 0 else 0,
            'classification_error_count': total_classification_errors,
            'classification_error_rate': total_classification_errors / total_predictions if total_predictions > 0 else 0,
            'localization_error_count': total_localization_errors,
            'localization_error_rate': total_localization_errors / total_predictions if total_predictions > 0 else 0,
            'severe_localization_count': len(failure_cases['severe_localization']),
            'high_confidence_wrong_count': len(failure_cases['high_confidence_wrong']),
            'low_confidence_correct_count': len(failure_cases['low_confidence_correct']),
            'boundary_case_count': len(failure_cases['boundary_cases']),
            'top_confusion_pairs': top_confusions,
            'class_failure_rates': class_failure_rates,
            'hardest_classes': hardest_classes
        }
    
    def generate_failure_report(self, failure_analysis, output_path=None):
        """
        Generate a detailed failure analysis report.
        """
        summary = failure_analysis['summary']
        
        report_lines = [
            "=" * 70,
            "FAILURE CASE ANALYSIS REPORT",
            f"IoU Threshold: {self.iou_threshold}",
            "=" * 70,
            "",
            "OVERALL STATISTICS:",
            f"  Total Predictions: {summary['total_predictions']}",
            f"  Total Failures: {summary['total_failures']} ({summary['overall_failure_rate']*100:.2f}%)",
            f"  Classification Errors: {summary['classification_error_count']} ({summary['classification_error_rate']*100:.2f}%)",
            f"  Localization Errors: {summary['localization_error_count']} ({summary['localization_error_rate']*100:.2f}%)",
            "",
            "CRITICAL ISSUES:",
            f"  Severe Localization (IoU<0.25): {summary['severe_localization_count']}",
            f"  High Confidence Wrong: {summary['high_confidence_wrong_count']}",
            f"  Boundary Cases (IoU 0.45-0.55): {summary['boundary_case_count']}",
            "",
            "TOP CONFUSION PAIRS (True -> Predicted):"
        ]
        
        for (true_cls, pred_cls), count in summary['top_confusion_pairs']:
            report_lines.append(f"  {true_cls} -> {pred_cls}: {count} cases")
        
        report_lines.extend([
            "",
            "PER-CLASS ANALYSIS:"
        ])
        
        for cls, rates in summary['class_failure_rates'].items():
            report_lines.extend([
                f"  {cls.upper()}:",
                f"    Samples: {rates['total_samples']}",
                f"    Success Rate: {rates['success_rate']*100:.2f}%",
                f"    Classification Error Rate: {rates['classification_error_rate']*100:.2f}%",
                f"    Localization Error Rate: {rates['localization_error_rate']*100:.2f}%",
                f"    Mean IoU: {rates['mean_iou']:.4f}",
                f"    Mean Confidence: {rates['mean_confidence']:.4f}"
            ])
        
        report_lines.extend([
            "",
            "HARDEST CLASSES (lowest success rate):"
        ])
        for cls, success_rate in summary['hardest_classes']:
            report_lines.append(f"  {cls}: {success_rate*100:.2f}% success rate")
        
        report_lines.append("=" * 70)
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"✓ Failure report saved to: {output_path}")
        
        return report


# ============================================================================
# 3. UNIFIED DATASET (UNCHANGED FROM ORIGINAL)
# ============================================================================

class UnifiedHybridDataset(Dataset):
    """
    Dataset class for loading and preprocessing JSON annotation data.
    Maintains consistency with the original implementation.
    """
    def __init__(self, json_folder, scaler=None, bbox_scaler=None):
        self.json_folder = json_folder
        self.data, self.labels, self.bbox_targets, self.file_info = self._load_json_data()
        self.label_map = {"cyclist": 0, "car": 1, "human": 2, "wall": 3, "tree": 4, "cart": 5}
        self.classes = list(self.label_map.keys())
        
        if scaler is not None:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)
        else:
            raise ValueError("Scaler harus disediakan untuk testing!")
        
        if bbox_scaler is not None:
            self.bbox_scaler = bbox_scaler
            self.bbox_targets = self.bbox_scaler.transform(self.bbox_targets)
        else:
            raise ValueError("BBox scaler harus disediakan untuk testing!")
        
        print(f"✓ Test Dataset: {len(self.data)} samples with 7 features")
        print(f"✓ Class distribution: {Counter(self.labels)}")
    
    def _load_json_data(self):
        """Load and parse JSON annotation files."""
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
        original_coords = torch.FloatTensor(self.file_info[idx]["coords"])
        original_dims = torch.FloatTensor(self.file_info[idx]["dimensions"])
        rotation_z = self.file_info[idx]["rotation_z"]
        
        return (features, class_target, bbox_target, 
                original_coords, original_dims, rotation_z)


# ============================================================================
# 4. GRAPH CONSTRUCTOR (UNCHANGED)
# ============================================================================

class SmartGraphConstructor:
    """
    Spatial graph constructor for GNN-based feature learning.
    Builds graphs based on spatial proximity of objects.
    """
    def __init__(self, connection_radius=5.0, max_neighbors=8):
        self.connection_radius = connection_radius
        self.max_neighbors = max_neighbors
    
    def build_spatial_graph(self, batch_features, batch_indices):
        """Build spatial graph from batch features."""
        device = batch_features.device
        positions = batch_features[:, :3]
        edge_indices = []
        edge_attrs = []
        
        unique_batches = torch.unique(batch_indices)
        
        for batch_id in unique_batches:
            mask = batch_indices == batch_id
            batch_positions = positions[mask]
            batch_size = batch_positions.size(0)
            
            if batch_size < 2:
                if batch_size == 1:
                    global_idx = torch.where(mask)[0][0].item()
                    edge_indices.append([global_idx, global_idx])
                    edge_attrs.append(torch.zeros(4, device=device))
                continue
            
            distances = torch.cdist(batch_positions, batch_positions, p=2)
            
            for i in range(batch_size):
                neighbor_mask = (distances[i] <= self.connection_radius) & (distances[i] > 0)
                neighbors = torch.where(neighbor_mask)[0]
                
                if len(neighbors) > self.max_neighbors:
                    neighbor_distances = distances[i][neighbors]
                    _, keep_indices = torch.topk(neighbor_distances, self.max_neighbors, largest=False)
                    neighbors = neighbors[keep_indices]
                
                global_i = torch.where(mask)[0][i].item()
                for j in neighbors:
                    global_j = torch.where(mask)[0][j].item()
                    rel_pos = batch_positions[j] - batch_positions[i]
                    distance = distances[i][j]
                    edge_feat = torch.cat([rel_pos, distance.unsqueeze(0)])
                    edge_indices.extend([[global_i, global_j], [global_j, global_i]])
                    edge_attrs.extend([edge_feat, -edge_feat])
        
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
# 5. FLEXIBLE GNN LAYER (UNCHANGED)
# ============================================================================

class FlexibleGNNLayer(nn.Module):
    """
    Flexible GNN layer that works with or without PyTorch Geometric.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.gnn_layer = GCNConv(in_dim, out_dim)
            self.use_advanced = True
        else:
            self.node_transform = nn.Linear(in_dim, out_dim)
            self.message_transform = nn.Linear(in_dim * 2, out_dim)
            self.use_advanced = False
        
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        if self.use_advanced:
            out = self.gnn_layer(x, edge_index)
        else:
            out = self.node_transform(x)
            
            if edge_index.size(1) > 0:
                row, col = edge_index
                neighbor_features = x[col]
                source_features = x[row]
                messages = self.message_transform(torch.cat([source_features, neighbor_features], dim=1))
                out = out.index_add(0, row, messages) / 2
        
        out = self.activation(out)
        out = self.layer_norm(out)
        return out


# ============================================================================
# 6. HYBRID GNN-TRANSFORMER MODEL (UNCHANGED)
# ============================================================================

class HybridGNNTransformerFCN(nn.Module):
    """
    Hybrid model combining GNN and Transformer architectures.
    
    Architecture flow:
    1. Input embedding
    2. GNN branch for spatial relationship learning
    3. Transformer branch for attention-based feature learning
    4. Cross-attention fusion
    5. Classification and regression heads
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_classes=6, 
                 num_heads=8, num_transformer_layers=3, num_gnn_layers=2, dropout=0.2):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.graph_constructor = SmartGraphConstructor()
        self.gnn_layers = nn.ModuleList([FlexibleGNNLayer(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, 
            batch_first=True, dropout=dropout
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fc_class = nn.Linear(hidden_dim, num_classes)
        self.fc_bbox = nn.Linear(hidden_dim, 7)
        self.gelu = nn.GELU()
    
    def forward(self, x, batch=None):
        batch_size = x.size(0)
        device = x.device
        if batch is None:
            batch = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        embedded = self.input_embedding(x)
        edge_index, edge_attr = self.graph_constructor.build_spatial_graph(x, batch)
        
        # GNN branch
        gnn_features = embedded
        for i, gnn_layer in enumerate(self.gnn_layers):
            residual = gnn_features
            gnn_features = gnn_layer(gnn_features, edge_index, edge_attr)
            if i > 0:
                gnn_features = gnn_features + residual * 0.5
        
        # Transformer branch
        transformer_input = embedded.unsqueeze(1)
        transformer_features = self.transformer(transformer_input).squeeze(1)
        
        # Cross-attention fusion
        gnn_query = gnn_features.unsqueeze(1)
        transformer_kv = transformer_features.unsqueeze(1)
        fused_features, _ = self.fusion_attention(gnn_query, transformer_kv, transformer_kv)
        fused_features = fused_features.squeeze(1)
        fused_features = self.fusion_norm(fused_features + embedded)
        
        activated = self.gelu(fused_features)
        class_output = self.fc_class(activated)
        bbox_output = self.fc_bbox(activated)
        
        return class_output, bbox_output


# ============================================================================
# 7. IoU AND ERROR CALCULATION (UNCHANGED)
# ============================================================================

def calculate_3d_iou_with_rotation(box1, box2):
    """
    Calculate 3D Intersection over Union with rotation support.
    
    Args:
        box1, box2: [x, y, z, dx, dy, dz, rotation_z]
    
    Returns:
        IoU value between 0 and 1
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
    """Calculate Euclidean translation error between predicted and ground truth."""
    return np.linalg.norm(np.array(pred_coords) - np.array(gt_coords))


# ============================================================================
# 8. ENHANCED COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_hybrid_model_comprehensive(model, test_loader, device, bbox_scaler, classes, 
                                        complexity_analyzer, failure_analyzer, iou_threshold=0.5):
    """
    Enhanced comprehensive evaluation with:
    - Standard classification metrics
    - 3D detection metrics with IoU threshold 0.5
    - Performance metrics (timing, FPS)
    - Model complexity analysis (FLOPs, parameters)
    - Detailed failure case analysis
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []
    all_class_predictions = []
    all_class_targets = []
    all_ious = []
    all_translation_errors = []
    
    total_inference_time = 0.0
    num_samples = 0
    
    print("🔍 Starting comprehensive hybrid model evaluation...")
    print(f"   IoU Threshold: {iou_threshold}")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            features, class_targets, bbox_targets, orig_coords, orig_dims, rotation_z = batch_data
            
            features = features.to(device)
            class_targets = class_targets.to(device)
            
            batch_indices = torch.arange(len(features), device=device)
            
            # Time the inference
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            class_output, bbox_output = model(features, batch_indices)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            total_inference_time += batch_time
            num_samples += len(features)
            
            class_preds = torch.argmax(class_output, dim=1)
            bbox_preds_denorm = bbox_scaler.inverse_transform(bbox_output.cpu().numpy())
            
            for i in range(len(features)):
                pred_box = bbox_preds_denorm[i]
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
    
    print("📊 Calculating comprehensive metrics...")
    
    # ==================== CLASSIFICATION METRICS ====================
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
    
    # ==================== 3D DETECTION METRICS (IoU >= 0.5) ====================
    mean_iou = np.mean(all_ious)
    mean_translation_error = np.mean(all_translation_errors)
    
    # True Positives, False Positives, False Negatives at IoU threshold 0.5
    tp_at_threshold = sum(1 for iou in all_ious if iou >= iou_threshold)
    fp_at_threshold = sum(1 for i, pred in enumerate(all_predictions) 
                          if pred['class_id'] != all_ground_truths[i]['class_id'] or all_ious[i] < iou_threshold)
    
    precision_at_threshold = tp_at_threshold / len(all_predictions) if len(all_predictions) > 0 else 0
    recall_at_threshold = tp_at_threshold / len(all_ground_truths) if len(all_ground_truths) > 0 else 0
    
    # Per-class IoU and AP at threshold 0.5
    per_class_iou = {}
    per_class_ap = {}
    per_class_tp = {}
    per_class_fp = {}
    
    for class_id, class_name in enumerate(classes):
        class_mask = [i for i, gt in enumerate(all_ground_truths) if gt['class_id'] == class_id]
        class_ious = [all_ious[i] for i in class_mask]
        per_class_iou[class_name] = np.mean(class_ious) if class_ious else 0.0
        
        # Calculate TP and FP at threshold for this class
        class_tp = sum(1 for i in class_mask if all_ious[i] >= iou_threshold and 
                       all_predictions[i]['class_id'] == class_id)
        class_fp = sum(1 for i in class_mask if all_ious[i] < iou_threshold or 
                       all_predictions[i]['class_id'] != class_id)
        
        per_class_tp[class_name] = class_tp
        per_class_fp[class_name] = class_fp
        
        # AP calculation
        class_gts_count = len(class_mask)
        per_class_ap[class_name] = class_tp / class_gts_count if class_gts_count > 0 else 0.0
    
    mAP = np.mean(list(per_class_ap.values()))
    
    # ==================== PERFORMANCE METRICS ====================
    avg_time_per_frame_sec = total_inference_time / num_samples if num_samples > 0 else 0.0
    avg_time_per_frame_ms = avg_time_per_frame_sec * 1000
    inference_speed_fps = num_samples / total_inference_time if total_inference_time > 0 else 0.0
    
    # ==================== MODEL COMPLEXITY ANALYSIS ====================
    print("🔧 Analyzing model complexity (FLOPs & Parameters)...")
    sample_input = next(iter(test_loader))[0][:1].to(device)
    complexity_results = complexity_analyzer.analyze_model(model, sample_input, device)
    
    # ==================== FAILURE CASE ANALYSIS ====================
    print("🔍 Performing failure case analysis...")
    failure_results = failure_analyzer.analyze_failures(all_predictions, all_ground_truths, classes)
    
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
            'all_translation_errors': all_translation_errors,
            'iou_threshold': iou_threshold,
            'tp_at_threshold': tp_at_threshold,
            'fp_at_threshold': fp_at_threshold,
            'precision_at_threshold': precision_at_threshold,
            'recall_at_threshold': recall_at_threshold,
            'per_class_tp': per_class_tp,
            'per_class_fp': per_class_fp
        },
        'performance': {
            'avg_time_per_frame_ms': avg_time_per_frame_ms,
            'inference_speed_fps': inference_speed_fps,
            'total_inference_time_sec': total_inference_time,
            'total_samples': num_samples
        },
        'model_complexity': complexity_results,
        'failure_analysis': failure_results,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths
    }


# ============================================================================
# 9. RESULTS SAVING
# ============================================================================

def save_comprehensive_results(results, save_path, timestamp):
    """
    Save all results to a comprehensive Excel file with multiple sheets.
    """
    excel_filename = f"enhanced_testing_results_{timestamp}.xlsx"
    excel_path = os.path.join(save_path, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Classification Metrics
        pd.DataFrame(results['classification']['per_class_metrics']).to_excel(
            writer, sheet_name='Classification_Metrics', index=False)
        
        # Sheet 2: Confusion Matrix
        cm_df = pd.DataFrame(
            results['classification']['confusion_matrix'],
            index=results['classification']['class_names'],
            columns=results['classification']['class_names']
        )
        cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
        
        # Sheet 3: Per-Class 3D Detection Metrics
        detection_data = []
        for class_name in results['classification']['class_names']:
            detection_data.append({
                'Class': class_name,
                'Mean_IoU': results['detection_3d']['per_class_iou'][class_name],
                'AP@0.5': results['detection_3d']['per_class_ap'][class_name],
                'TP@0.5': results['detection_3d']['per_class_tp'][class_name],
                'FP@0.5': results['detection_3d']['per_class_fp'][class_name]
            })
        pd.DataFrame(detection_data).to_excel(writer, sheet_name='3D_Detection_Metrics', index=False)
        
        # Sheet 4: Performance Metrics
        performance_data = {
            'Metric': [
                'Average_Time_per_Frame_ms',
                'Inference_Speed_FPS',
                'Total_Inference_Time_sec',
                'Total_Samples'
            ],
            'Value': [
                results['performance']['avg_time_per_frame_ms'],
                results['performance']['inference_speed_fps'],
                results['performance']['total_inference_time_sec'],
                results['performance']['total_samples']
            ]
        }
        pd.DataFrame(performance_data).to_excel(writer, sheet_name='Performance', index=False)
        
        # Sheet 5: Model Complexity (FLOPs & Parameters)
        complexity = results['model_complexity']
        complexity_data = {
            'Metric': [
                'Total_Parameters',
                'Trainable_Parameters',
                'Total_Parameters_Millions',
                'FLOPs',
                'GFLOPs',
                'Model_Size_MB',
                'Peak_GPU_Memory_MB'
            ],
            'Value': [
                complexity['parameters']['total_params'],
                complexity['parameters']['trainable_params'],
                f"{complexity['parameters']['total_params_millions']:.4f}",
                complexity['flops']['flops_formatted'],
                f"{complexity['flops']['gflops']:.6f}",
                f"{complexity['memory']['model_size_mb']:.4f}",
                f"{complexity['memory']['peak_gpu_memory_mb']:.4f}"
            ]
        }
        pd.DataFrame(complexity_data).to_excel(writer, sheet_name='Model_Complexity', index=False)
        
        # Sheet 6: Failure Analysis Summary
        failure_summary = results['failure_analysis']['summary']
        failure_data = {
            'Metric': [
                'Total_Predictions',
                'Total_Failures',
                'Overall_Failure_Rate',
                'Classification_Errors',
                'Classification_Error_Rate',
                'Localization_Errors',
                'Localization_Error_Rate',
                'Severe_Localization_Count',
                'High_Confidence_Wrong_Count',
                'Boundary_Case_Count'
            ],
            'Value': [
                failure_summary['total_predictions'],
                failure_summary['total_failures'],
                f"{failure_summary['overall_failure_rate']*100:.2f}%",
                failure_summary['classification_error_count'],
                f"{failure_summary['classification_error_rate']*100:.2f}%",
                failure_summary['localization_error_count'],
                f"{failure_summary['localization_error_rate']*100:.2f}%",
                failure_summary['severe_localization_count'],
                failure_summary['high_confidence_wrong_count'],
                failure_summary['boundary_case_count']
            ]
        }
        pd.DataFrame(failure_data).to_excel(writer, sheet_name='Failure_Summary', index=False)
        
        # Sheet 7: Per-Class Failure Rates
        class_failure_data = []
        for cls, rates in failure_summary['class_failure_rates'].items():
            class_failure_data.append({
                'Class': cls,
                'Total_Samples': rates['total_samples'],
                'Success_Rate': f"{rates['success_rate']*100:.2f}%",
                'Classification_Error_Rate': f"{rates['classification_error_rate']*100:.2f}%",
                'Localization_Error_Rate': f"{rates['localization_error_rate']*100:.2f}%",
                'Mean_IoU': f"{rates['mean_iou']:.4f}",
                'Mean_Confidence': f"{rates['mean_confidence']:.4f}"
            })
        pd.DataFrame(class_failure_data).to_excel(writer, sheet_name='Per_Class_Failures', index=False)
        
        # Sheet 8: Top Confusion Pairs
        confusion_pairs_data = []
        for (true_cls, pred_cls), count in failure_summary['top_confusion_pairs']:
            confusion_pairs_data.append({
                'True_Class': true_cls,
                'Predicted_Class': pred_cls,
                'Count': count
            })
        pd.DataFrame(confusion_pairs_data).to_excel(writer, sheet_name='Top_Confusion_Pairs', index=False)
        
        # Sheet 9: Overall Summary
        summary_data = {
            'Metric': [
                'Model_Architecture',
                'Overall_Accuracy',
                'Mean_IoU',
                'mAP@0.5',
                'Precision@0.5',
                'Recall@0.5',
                'Mean_Translation_Error',
                'Average_Time_per_Frame_ms',
                'Inference_Speed_FPS',
                'Total_Parameters',
                'GFLOPs',
                'Overall_Failure_Rate'
            ],
            'Value': [
                'Hybrid GNN-Transformer FCN',
                f"{results['classification']['accuracy']:.4f}",
                f"{results['detection_3d']['mean_iou']:.4f}",
                f"{results['detection_3d']['mAP']:.4f}",
                f"{results['detection_3d']['precision_at_threshold']:.4f}",
                f"{results['detection_3d']['recall_at_threshold']:.4f}",
                f"{results['detection_3d']['mean_translation_error']:.4f}",
                f"{results['performance']['avg_time_per_frame_ms']:.4f}",
                f"{results['performance']['inference_speed_fps']:.2f}",
                f"{complexity['parameters']['total_params']:,}",
                f"{complexity['flops']['gflops']:.6f}",
                f"{failure_summary['overall_failure_rate']*100:.2f}%"
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Overall_Summary', index=False)
    
    print(f"✅ Comprehensive results saved to: {excel_path}")
    return excel_path


# ============================================================================
# 10. MAIN FUNCTION
# ============================================================================

def main():
    """
    Main testing function with enhanced analysis capabilities.
    """
    # Configuration paths (adjust as needed)
    test_json = "data/test"
    model_path = "outputs/checkpoints/hybrid_gnn_transformer_20250702_174803.pth"
    results_save = "outputs/results"
    
    # Create output directory if needed
    os.makedirs(results_save, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    print("=" * 70)
    print("ENHANCED HYBRID GNN-TRANSFORMER TESTING")
    print("=" * 70)
    print("Features:")
    print("  ✓ FLOPs Calculation")
    print("  ✓ Total Model Parameters")
    print("  ✓ IoU Threshold 0.5 Evaluation")
    print("  ✓ Comprehensive Failure Case Analysis")
    print("=" * 70)
    
    # Initialize analyzers
    complexity_analyzer = ModelComplexityAnalyzer()
    failure_analyzer = FailureCaseAnalyzer(iou_threshold=0.5)
    
    # Load checkpoint
    print("\n📂 Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    scaler = checkpoint['scaler']
    bbox_scaler = checkpoint['bbox_scaler']
    print("✓ Checkpoint loaded successfully")
    
    # Load test dataset
    print("\n📂 Loading test dataset...")
    test_dataset = UnifiedHybridDataset(test_json, scaler=scaler, bbox_scaler=bbox_scaler)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    classes = ["cyclist", "car", "human", "wall", "tree", "cart"]
    
    # Initialize and load model
    print("\n🧠 Initializing model...")
    model = HybridGNNTransformerFCN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded and ready")
    
    # Run comprehensive evaluation
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("=" * 70)
    
    results = evaluate_hybrid_model_comprehensive(
        model, test_loader, device, bbox_scaler, classes,
        complexity_analyzer, failure_analyzer, iou_threshold=0.5
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 ENHANCED TESTING RESULTS")
    print("=" * 70)
    
    print("\n--- CLASSIFICATION METRICS ---")
    print(f"✅ Overall Accuracy: {results['classification']['accuracy']:.4f}")
    
    print("\n--- 3D DETECTION METRICS (IoU Threshold: 0.5) ---")
    print(f"✅ Mean 3D IoU: {results['detection_3d']['mean_iou']:.4f}")
    print(f"✅ mAP@0.5: {results['detection_3d']['mAP']:.4f}")
    print(f"✅ Precision@0.5: {results['detection_3d']['precision_at_threshold']:.4f}")
    print(f"✅ Recall@0.5: {results['detection_3d']['recall_at_threshold']:.4f}")
    print(f"✅ Mean Translation Error: {results['detection_3d']['mean_translation_error']:.4f}")
    print(f"✅ TP@0.5: {results['detection_3d']['tp_at_threshold']}")
    print(f"✅ FP@0.5: {results['detection_3d']['fp_at_threshold']}")
    
    print("\n--- PER-CLASS IoU and AP@0.5 ---")
    for class_name in classes:
        iou = results['detection_3d']['per_class_iou'][class_name]
        ap = results['detection_3d']['per_class_ap'][class_name]
        print(f"  {class_name:8s}: IoU={iou:.4f}, AP@0.5={ap:.4f}")
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"✅ Average Time per Frame: {results['performance']['avg_time_per_frame_ms']:.4f} ms/frame")
    print(f"✅ Inference Speed: {results['performance']['inference_speed_fps']:.2f} FPS")
    
    print("\n--- MODEL COMPLEXITY ---")
    complexity = results['model_complexity']
    print(f"✅ Total Parameters: {complexity['parameters']['total_params']:,}")
    print(f"✅ Trainable Parameters: {complexity['parameters']['trainable_params']:,}")
    print(f"✅ Parameters (Millions): {complexity['parameters']['total_params_millions']:.4f} M")
    print(f"✅ FLOPs: {complexity['flops']['flops_formatted']}")
    print(f"✅ GFLOPs: {complexity['flops']['gflops']:.6f}")
    print(f"✅ Model Size: {complexity['memory']['model_size_mb']:.4f} MB")
    
    print("\n--- FAILURE CASE ANALYSIS ---")
    failure_summary = results['failure_analysis']['summary']
    print(f"✅ Total Predictions: {failure_summary['total_predictions']}")
    print(f"✅ Total Failures: {failure_summary['total_failures']} ({failure_summary['overall_failure_rate']*100:.2f}%)")
    print(f"✅ Classification Errors: {failure_summary['classification_error_count']} ({failure_summary['classification_error_rate']*100:.2f}%)")
    print(f"✅ Localization Errors: {failure_summary['localization_error_count']} ({failure_summary['localization_error_rate']*100:.2f}%)")
    print(f"✅ Severe Localization (IoU<0.25): {failure_summary['severe_localization_count']}")
    print(f"✅ High Confidence Wrong: {failure_summary['high_confidence_wrong_count']}")
    
    print("\n--- TOP CONFUSION PAIRS ---")
    for (true_cls, pred_cls), count in failure_summary['top_confusion_pairs'][:5]:
        print(f"  {true_cls} → {pred_cls}: {count} cases")
    
    print("\n--- HARDEST CLASSES ---")
    for cls, success_rate in failure_summary['hardest_classes']:
        print(f"  {cls}: {success_rate*100:.2f}% success rate")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = save_comprehensive_results(results, results_save, timestamp)
    
    # Generate and save failure report
    report_path = os.path.join(results_save, f"failure_report_{timestamp}.txt")
    failure_report = failure_analyzer.generate_failure_report(results['failure_analysis'], report_path)
    
    print("\n" + "=" * 70)
    print("🎉 ENHANCED TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📁 Excel Results: {excel_path}")
    print(f"📁 Failure Report: {report_path}")


if __name__ == "__main__":
    main()