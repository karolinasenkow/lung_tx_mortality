import os
import re
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Resized, ToTensord
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import DenseNet121  

BASE_LOGDIR = "./logs/DenseNet-2"

class Modified3DDenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.densenet = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes
        )

    def forward(self, x):
        return self.densenet(x)

def safe_save_json(data, filepath):
    temp_file = filepath + '.tmp'
    try:
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=4)
        os.replace(temp_file, filepath)
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

def save_checkpoint(state, is_best, logdir, fold, epoch):
    os.makedirs(os.path.join(logdir, f"fold_{fold}"), exist_ok=True)
    checkpoint_state = {
        'epoch': epoch,
        'model_state_dict': state['model'].state_dict(),
        'optimizer_state_dict': state['optimizer'].state_dict(),
        'best_val_loss': state['best_val_loss'],
        'train_metrics_history': state['train_metrics_history'],
        'val_metrics_history': state['val_metrics_history'],
        'early_stop_counter': state['early_stop_counter']
    }
    checkpoint_path = os.path.join(logdir, f"fold_{fold}", f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint_state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(logdir, f"fold_{fold}_best_model.pth")
        torch.save(checkpoint_state, best_model_path)

def calculate_metrics(outputs, labels):
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    labels_np = labels.cpu().numpy()
    try:
        accuracy = np.mean(predictions == labels_np)
        auroc = roc_auc_score(labels_np, probabilities[:, 1])
        conf_mat = confusion_matrix(labels_np, predictions)
        class_report = classification_report(labels_np, predictions, output_dict=True)
        precision, recall, _ = precision_recall_curve(labels_np, probabilities[:, 1])
        avg_precision = average_precision_score(labels_np, probabilities[:, 1])
        return {
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'conf_matrix': conf_mat,
            'precision': float(class_report['weighted avg']['precision']),
            'recall': float(class_report['weighted avg']['recall']),
            'f1': float(class_report['weighted avg']['f1-score']),
            'avg_precision': float(avg_precision),
            'loss': 0.0
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def load_data(labels_file, data_path, task):
    print(f"Reading labels file for task: {task}")
    df = pd.read_csv(labels_file, index_col=0)
    df = df[df[task].notna()]
    df['path'] = f'{data_path}/' + df['Patient'].astype(str) + '/' + df['filename'].astype(str) + '.nii.gz'
    df = df[['path', task, 'Patient']]
    file_list = df['path'].values
    labels = df[task].astype(int).values
    patient_ids = df['Patient'].values
    print(f"Total matched files for {task}: {len(file_list)}")
    return file_list, labels, patient_ids

def prepare_data(files, labels):
    return [{"image": file_path, "label": label} for file_path, label in zip(files, labels)]

def evaluate_model(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(data_loader)
    return metrics

def train_model(task, data_path, labels_file, batch_size, lr, patience, max_epochs, resume_checkpoint=None, resume_fold=None):
    # if not resuming, create a new log directory
    if resume_checkpoint is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = os.path.join(BASE_LOGDIR, f"densenet_{task}_{timestamp}")
        os.makedirs(logdir, exist_ok=True)
        config = {
            'task': task,
            'learning_rate': lr,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'patience': patience,
            'timestamp': timestamp
        }
        safe_save_json(config, os.path.join(logdir, 'config.json'))
    else:
        logdir = os.path.dirname(os.path.dirname(os.path.abspath(resume_checkpoint)))
        print(f"Resuming run using log directory: {logdir}")

    # standardization
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(128, 128, 128)),
        ToTensord(keys=["image"]),
    ])

    file_list, labels, patient_ids = load_data(labels_file, data_path, task)
    df_patients = pd.DataFrame({'patient': patient_ids, 'label': labels})
    patient_labels = df_patients.groupby('patient')['label'].max()
    unique_patients = patient_labels.index.to_numpy()
    unique_labels = patient_labels.values

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results = []

    for fold, (train_val_patient_idx, test_patient_idx) in enumerate(skf.split(unique_patients, unique_labels)):
        print(f"\nProcessing Fold {fold + 1}/{k_folds}")
        train_val_patients = unique_patients[train_val_patient_idx]
        test_patients = unique_patients[test_patient_idx]
        train_val_mask = np.isin(patient_ids, train_val_patients)
        test_mask = np.isin(patient_ids, test_patients)
        train_val_files = file_list[train_val_mask]
        train_val_labels = labels[train_val_mask]
        test_files = file_list[test_mask]
        test_labels = labels[test_mask]

        df_train_val = pd.DataFrame({'patient': patient_ids[train_val_mask], 'label': labels[train_val_mask]})
        train_val_patient_labels = df_train_val.groupby('patient')['label'].max()
        unique_train_val_patients = train_val_patient_labels.index.to_numpy()
        unique_train_val_labels = train_val_patient_labels.values

        train_patients, val_patients = train_test_split(
            unique_train_val_patients, test_size=0.2, 
            stratify=unique_train_val_labels, random_state=42
        )
        train_mask = np.isin(patient_ids[train_val_mask], train_patients)
        val_mask = np.isin(patient_ids[train_val_mask], val_patients)
        train_files_final = train_val_files[train_mask]
        train_labels_final = train_val_labels[train_mask]
        val_files_final = train_val_files[val_mask]
        val_labels_final = train_val_labels[val_mask]

        train_ds = CacheDataset(data=prepare_data(train_files_final, train_labels_final), transform=transforms, cache_rate=0.0)
        val_ds = CacheDataset(data=prepare_data(val_files_final, val_labels_final), transform=transforms, cache_rate=0.0)
        test_ds = CacheDataset(data=prepare_data(test_files, test_labels), transform=transforms, cache_rate=0.0)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        train_class_counts = Counter(train_labels_final)
        total_train_samples = sum(train_class_counts.values())
        class_weights_dict = {cls: total_train_samples / train_class_counts[cls] for cls in sorted(train_class_counts.keys())}
        class_weights_tensor = torch.tensor([class_weights_dict[i] for i in sorted(class_weights_dict.keys())], dtype=torch.float32)
        loss_function = CrossEntropyLoss(weight=class_weights_tensor.to(device))

        # instantiate DenseNet instead of ResNet
        model = Modified3DDenseNet(num_classes=2).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        # training loop for this fold
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_metrics_history = []
        val_metrics_history = []
        for epoch in range(1, max_epochs + 1):
            print(f"\nEpoch {epoch}/{max_epochs} for Fold {fold + 1}")
            model.train()
            train_loss = 0
            train_outputs = []
            train_labels_list = []
            for batch in tqdm(train_loader, desc="Training", leave=False):
                inputs, batch_labels = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_outputs.append(outputs.detach())
                train_labels_list.append(batch_labels)
            train_outputs = torch.cat(train_outputs)
            train_labels_tensor = torch.cat(train_labels_list)
            train_metrics = calculate_metrics(train_outputs, train_labels_tensor)
            train_metrics['loss'] = train_loss / len(train_loader)
            train_metrics_history.append(train_metrics)

            val_metrics = evaluate_model(model, val_loader, loss_function, device)
            val_metrics_history.append(val_metrics)
            print(f"Fold {fold + 1} Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, Train AUC: {train_metrics['auroc']:.4f}")
            print(f"Fold {fold + 1} Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auroc']:.4f}")

            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            state = {
                'model': model,
                'optimizer': optimizer,
                'best_val_loss': best_val_loss,
                'train_metrics_history': train_metrics_history,
                'val_metrics_history': val_metrics_history,
                'early_stop_counter': early_stop_counter
            }
            save_checkpoint(state, is_best, logdir, fold, epoch)
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs for Fold {fold + 1}")
                break

        # testing: load best checkpoint and evaluate
        best_model_path = os.path.join(logdir, f"fold_{fold}_best_model.pth")
        best_checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        test_metrics = evaluate_model(model, test_loader, loss_function, device)
        fold_summary = {
            'best_val_loss': float(best_checkpoint['best_val_loss']),
            'best_epoch': best_checkpoint['epoch'],
            'test_metrics': {
                'loss': float(test_metrics['loss']),
                'accuracy': float(test_metrics['accuracy']),
                'auroc': float(test_metrics['auroc']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1']),
                'avg_precision': float(test_metrics['avg_precision']),
                'conf_matrix': test_metrics['conf_matrix'].tolist()
            }
        }
        safe_save_json(fold_summary, os.path.join(logdir, f"fold_{fold}_summary.json"))
        fold_results.append(fold_summary)

    overall_results = {
        'task': str(task),
        'number_of_folds': len(fold_results),
        'average_best_val_loss': float(np.mean([f['best_val_loss'] for f in fold_results])),
        'average_test_auc': float(np.mean([f['test_metrics']['auroc'] for f in fold_results])),
        'std_test_auc': float(np.std([f['test_metrics']['auroc'] for f in fold_results])),
        'average_test_accuracy': float(np.mean([f['test_metrics']['accuracy'] for f in fold_results])),
        'average_test_f1': float(np.mean([f['test_metrics']['f1'] for f in fold_results])),
        'fold_summaries': fold_results
    }
    safe_save_json(overall_results, os.path.join(logdir, 'overall_results.json'))
    print("\nTraining and testing completed for all folds!")
    print(f"Average best validation loss: {overall_results['average_best_val_loss']:.4f}")
    print(f"Average test AUC: {overall_results['average_test_auc']:.4f} (±{overall_results['std_test_auc']:.4f})")
    print(f"Average test F1: {overall_results['average_test_f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Modified 3D DenseNet model for death prediction')
    parser.add_argument('--task', type=str, required=True, help='Task to train on (e.g., mortality_12m)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--data_path', type=str, default='/path/to/data', help='Path to the data directory')
    parser.add_argument('--labels_file', type=str, default='/path/to/labels.csv', help='Path to the labels file')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume testing')
    parser.add_argument('--resume_fold', type=int, default=None, help='Fold number to resume testing (0-indexed)')
    
    args = parser.parse_args()
    
    print(f"\nStarting run for task: {args.task}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Maximum epochs: {args.max_epochs}\n")
    
    try:
        train_model(args.task, args.data_path, args.labels_file,
                    args.batch_size, args.lr, args.patience, args.max_epochs,
                    resume_checkpoint=args.resume_checkpoint, resume_fold=args.resume_fold)
    except Exception as e:
        print(f"Error during training/testing: {str(e)}")
        raise
