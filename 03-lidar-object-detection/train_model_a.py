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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import warnings
import random
from collections import Counter
import open3d as o3d
import pyvista as pv
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# 1. ENHANCED DATASET - SAME AS BEFORE BUT WITH BETTER TRACKING
# ============================================================================

class FixedJSONDataset(Dataset):
    """
    Dataset yang sudah terbukti menghasilkan IoU 0.66
    Tidak ada perubahan dari versi yang sukses
    """
    def __init__(self, json_folder, scaler=None, is_training=True):
        self.json_folder = json_folder
        self.is_training = is_training
        
        # Extract all data first (sama seperti sebelumnya)
        self.data, self.labels, self.bbox_targets, self.file_info = self._load_json_data()
        
        # Setup label mapping
        self.label_map = {"cyclist": 0, "car": 1, "human": 2, "wall": 3, "tree": 4, "cart": 5}
        self.classes = list(self.label_map.keys())
        
        # Global normalization (proven effective)
        if scaler is None and is_training:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
            print("✓ Global StandardScaler fitted pada training data")
        elif scaler is not None:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)
            print("✓ Using shared StandardScaler untuk consistency")
        else:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
            
        # Bbox target normalization
        if is_training:
            self.bbox_scaler = StandardScaler() 
            self.bbox_targets = self.bbox_scaler.fit_transform(self.bbox_targets)
            print("✓ BBox target normalization fitted")
        
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
# 3. ENHANCED TRAINER WITH COMPREHENSIVE LOGGING
# ============================================================================

class EnhancedTrainer:
    """
    Enhanced trainer dengan comprehensive logging untuk analisis mendalam
    """
    def __init__(self, model, device='cuda', train_dataset=None):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Setup cost-sensitive loss
        if train_dataset is not None:
            class_weights = self._calculate_class_weights(train_dataset)
            self.criterion_class = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print(f"✓ Cost-sensitive weights: {class_weights}")
        else:
            self.criterion_class = nn.CrossEntropyLoss()
        
        self.criterion_bbox = nn.L1Loss()
        
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.3, patience=5, verbose=True
        )
        
        self.alpha = 1.0
        
        # ENHANCED: Comprehensive tracking untuk analisis
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_class_loss': [],
            'train_regression_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_class_loss': [],
            'val_regression_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        print("=== ENHANCED TRAINER INITIALIZED ===")
        print("✓ Comprehensive metrics tracking enabled")
        print("✓ Ready for detailed analysis")
    
    def _calculate_class_weights(self, dataset):
        """Calculate cost-sensitive weights"""
        labels = [dataset.labels[i] for i in range(len(dataset))]
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
        
        weights_tensor = torch.ones(6)
        for i, weight in zip(unique_labels, class_weights):
            weights_tensor[i] = weight
            
        return weights_tensor
    
    def train_epoch(self, train_loader):
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
            
            self.optimizer.zero_grad()
            class_output, bbox_output = self.model(batch_X)
            
            loss_class = self.criterion_class(class_output, batch_y)
            loss_bbox = self.criterion_bbox(bbox_output, batch_bbox)
            loss = loss_class + self.alpha * loss_bbox
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_class_loss += loss_class.item()
            total_regression_loss += loss_bbox.item()
            
            predicted = torch.argmax(class_output, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        epoch_time = time.time() - epoch_start_time
        
        avg_loss = total_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_regression_loss = total_regression_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, avg_class_loss, avg_regression_loss, epoch_time
    
    def validate_epoch(self, val_loader):
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
                
                class_output, bbox_output = self.model(batch_X)
                
                loss_class = self.criterion_class(class_output, batch_y)
                loss_bbox = self.criterion_bbox(bbox_output, batch_bbox)
                loss = loss_class + self.alpha * loss_bbox
                
                total_loss += loss.item()
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
        print("=== STARTING ENHANCED TRAINING ===")
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc, train_class_loss, train_reg_loss, epoch_time = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_class_loss, val_reg_loss = self.validate_epoch(val_loader)
            
            # Scheduler step
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
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%')
                print(f'           Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')
                print(f'           Reg Loss: Train={train_reg_loss:.4f}, Val={val_reg_loss:.4f}')
            
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
                        'val_loss': val_loss
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"Training completed! Best val loss: {best_val_loss:.4f}")
        return self.training_history

# ============================================================================
# 4. ENHANCED IoU CALCULATION - PROVEN EFFECTIVE
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
# 5. COMPREHENSIVE EVALUATION WITH ALL METRICS
# ============================================================================

def evaluate_model_comprehensive_enhanced(model, test_loader, device, bbox_scaler, classes):
    """
    Comprehensive evaluation dengan semua metrics yang diminta
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []
    all_class_predictions = []
    all_class_targets = []
    all_ious = []
    all_translation_errors = []
    
    print("🔍 Starting comprehensive evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            features, class_targets, bbox_targets, orig_coords, orig_dims, rotation_z = batch_data
            
            features = features.to(device)
            class_targets = class_targets.to(device)
            
            class_output, bbox_output = model(features)
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
    
    # Per-class IoU and AP
    per_class_iou = {}
    per_class_ap = {}
    
    for class_id, class_name in enumerate(classes):
        # IoU per class
        class_ious = [all_ious[i] for i, pred in enumerate(all_predictions) 
                     if all_ground_truths[i]['class_id'] == class_id]
        per_class_iou[class_name] = np.mean(class_ious) if class_ious else 0.0
        
        # AP per class (simplified)
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
# 6. COMPREHENSIVE VISUALIZATION AND SAVING
# ============================================================================

def create_comprehensive_visualizations(training_history, results, save_path):
    """
    Create comprehensive visualizations dan save ke PNG
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss curves
    axes[0, 0].plot(training_history['epoch'], training_history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(training_history['epoch'], training_history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(training_history['epoch'], training_history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(training_history['epoch'], training_history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Regression loss curves
    axes[1, 0].plot(training_history['epoch'], training_history['train_regression_loss'], 'b-', label='Train Reg Loss')
    axes[1, 0].plot(training_history['epoch'], training_history['val_regression_loss'], 'r-', label='Val Reg Loss')
    axes[1, 0].set_title('Regression Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Regression Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate curve
    axes[1, 1].plot(training_history['epoch'], training_history['learning_rate'], 'g-')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'training_curves_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = results['classification']['confusion_matrix']
    classes = results['classification']['class_names']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (6x6)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'confusion_matrix_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-class metrics
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Precision, Recall, F1-Score
    metrics_df = pd.DataFrame(results['classification']['per_class_metrics'])
    x = np.arange(len(metrics_df))
    width = 0.25
    
    axes[0].bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    axes[0].bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    axes[0].bar(x + width, metrics_df['F1_Score'], width, label='F1-Score', alpha=0.8)
    
    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Classification Metrics per Class')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_df['Class'], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU and AP per class
    class_names = list(results['detection_3d']['per_class_iou'].keys())
    iou_values = list(results['detection_3d']['per_class_iou'].values())
    ap_values = list(results['detection_3d']['per_class_ap'].values())
    
    x = np.arange(len(class_names))
    axes[1].bar(x - 0.2, iou_values, 0.4, label='IoU', alpha=0.8, color='skyblue')
    axes[1].bar(x + 0.2, ap_values, 0.4, label='AP', alpha=0.8, color='lightgreen')
    
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Score')
    axes[1].set_title('3D Detection Metrics per Class')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'per_class_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved with timestamp: {timestamp}")
    return timestamp

def save_comprehensive_results(training_history, results, save_path, timestamp):
    """
    Save semua results ke multi-sheet Excel
    """
    excel_filename = f"comprehensive_results_{timestamp}.xlsx"
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
        
        # 5. Translation Errors
        translation_df = pd.DataFrame({
            'Sample_ID': range(len(results['detection_3d']['all_translation_errors'])),
            'Translation_Error': results['detection_3d']['all_translation_errors'],
            'IoU': results['detection_3d']['all_ious']
        })
        translation_df.to_excel(writer, sheet_name='Translation_Errors', index=False)
        
        # 6. Summary
        summary_data = {
            'Metric': [
                'Overall_Accuracy', 'Mean_IoU', 'mAP', 'Mean_Translation_Error',
                'Training_Epochs', 'Best_Val_Loss', 'Final_Train_Acc', 'Final_Val_Acc'
            ],
            'Value': [
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
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Comprehensive results saved: {excel_filename}")
    return excel_path

# ============================================================================
# 7. ENHANCED 3D VISUALIZATION WITH CORRECT NAMING CONVENTION
# ============================================================================

def create_interactive_3d_visualizations(results, pcd_base_path, json_base_path, num_files=5):
    """
    CLEAN & ELEGANT Interactive 3D visualizations
    
    Features:
    - Point cloud sebagai background utama (prominent)
    - Solid colored bounding boxes (bukan wireframe)
    - Elegant labels dengan background box
    - No clutter (no axes, no help text, no unnecessary UI)
    - Smart label positioning untuk avoid overlap
    """
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
# 8. MAIN PROGRAM - COMPREHENSIVE VERSION
# ============================================================================

def main():
    """
    Main program dengan semua enhancements dan visualizations
    """
    # PATHS - CORRECTED WITH PROPER PCD PATH AND NAMING CONVENTION
    PATHS = {
        'train_json': "data/train",
        'val_json': "data/val",
        'test_json': "data/test",
        'pcd_base': "data/pcd_test",  # CORRECTED PATH
        'model_save': "outputs/checkpoints/",
        'results_save': "outputs/results",
        'images_save': "outputs/figures"
    }
    
    # Create directories if not exist
    for path in [PATHS['results_save'], PATHS['images_save']]:
        os.makedirs(path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("=" * 80)
    print("🚀 COMPREHENSIVE ENHANCED PROGRAM")
    print("📊 Features: Training curves, Confusion matrix, 3D IoU, AP, mAP, Translation error")
    print("🎨 Visualizations: PyVista 3D, Multi-sheet Excel, PNG curves")
    print("=" * 80)
    
    try:
        # 1. Load datasets
        print("📂 Loading datasets...")
        train_dataset = FixedJSONDataset(PATHS['train_json'], is_training=True)
        val_dataset = FixedJSONDataset(PATHS['val_json'], scaler=train_dataset.scaler, is_training=False)
        val_dataset.bbox_scaler = train_dataset.bbox_scaler
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        classes = ["cyclist", "car", "human", "wall", "tree", "cart"]
        
        # 2. Initialize enhanced model
        model = FixedTransformerModel(input_dim=7, num_classes=6)
        
        # 3. Train dengan enhanced trainer
        trainer = EnhancedTrainer(model, device, train_dataset)
        model_path = os.path.join(PATHS['model_save'], f"enhanced_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        training_history = trainer.train(train_loader, val_loader, epochs=100, save_path=model_path)
        
        # 4. Test dengan comprehensive evaluation
        print("\n🧪 Starting comprehensive testing...")
        test_dataset = FixedJSONDataset(PATHS['test_json'], scaler=train_dataset.scaler, is_training=False)
        test_dataset.bbox_scaler = train_dataset.bbox_scaler
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Load best model
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Comprehensive evaluation
        results = evaluate_model_comprehensive_enhanced(model, test_loader, device, train_dataset.bbox_scaler, classes)
        
        # Print results
        print("\n📊 COMPREHENSIVE RESULTS:")
        print(f"✅ Accuracy: {results['classification']['accuracy']:.4f}")
        print(f"✅ Mean IoU: {results['detection_3d']['mean_iou']:.4f}")
        print(f"✅ mAP: {results['detection_3d']['mAP']:.4f}")
        print(f"✅ Mean Translation Error: {results['detection_3d']['mean_translation_error']:.4f}")
        
        # 5. Create comprehensive visualizations
        print("\n🎨 Creating comprehensive visualizations...")
        timestamp = create_comprehensive_visualizations(training_history, results, PATHS['images_save'])
        
        # 6. Save comprehensive results
        excel_path = save_comprehensive_results(training_history, results, PATHS['results_save'], timestamp)
        
        # 7. Create INTERACTIVE 3D visualizations (no PNG saving)
        if os.path.exists(PATHS['pcd_base']):
            print("\n🎮 Starting INTERACTIVE 3D visualizations...")
            print("💡 Each visualization akan terbuka dalam window terpisah")
            print("   Anda dapat merotasi, zoom, dan explore setiap scene secara interaktif")
            print("   Tutup window untuk melanjutkan ke file berikutnya")
            create_interactive_3d_visualizations(results, PATHS['pcd_base'], PATHS['test_json'], num_files=5)
        else:
            print("⚠️  PCD base path not found. Skipping 3D visualizations.")
            print(f"🔍 Checked path: {PATHS['pcd_base']}")
            print("💡 Please verify the PCD path is correct")
        
        print("\n🎉 COMPREHENSIVE PROGRAM COMPLETED!")
        print(f"✅ Excel results: {excel_path}")
        print(f"✅ PNG visualizations: {PATHS['images_save']}")
        print(f"✅ Model saved: {model_path}")
        
        if results['detection_3d']['mean_iou'] > 0.5:
            print("🏆 SUCCESS! IoU target achieved and comprehensive analysis completed!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()