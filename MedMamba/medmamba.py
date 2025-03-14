import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score  
import sys
from tqdm import tqdm  

sys.path.append("/projects/b1038/Pulmonary/ksenkow/CLAD_serial_CT/code/mamba_env/MedMamba")
from MedMamba3D import VSSM3D as MedMamba3D  

logdir = "./logs/medmamba3d-v2"
os.makedirs(logdir, exist_ok=True)

data_path = "/projects/b1038/Pulmonary/ksenkow/CLAD_serial_CT/data/6multiplied"
data = pd.read_csv('/projects/b1038/Pulmonary/ksenkow/CLAD_serial_CT/data/v2_analysis/01gather_data/mortality_metadata.csv', index_col=0)
data['path'] = f'{data_path}/' + data['Patient'] + '/' + data['filename'] + '.nii.gz'
data = data[['path', 'mortality_12m', 'Patient']]

def prepare_data(data):
    return [{'image': row['path'], 'label': row['mortality_12m']} for _, row in data.iterrows()]

patient_mortality = data.groupby('Patient')['mortality_12m'].max()

# stratified group k fold
k_folds = 5
stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# define Dataset class
class CTScanDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path, label = sample['image'], sample['label']
        
        image = nib.load(image_path).get_fdata()
        
        image = np.clip(image, -175, 250)
        image = (image + 175) / 425.0

        # ensure shape is (1, H, W, D) for CNN input
        image = np.expand_dims(image, axis=0)  # (1, H, W, D)
        
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float32)

        # resize to (96, 96, 96)
        image = F.interpolate(image.unsqueeze(0), size=(96, 96, 96), mode='trilinear', align_corners=False).squeeze(0)

        return image, torch.tensor(label, dtype=torch.long)

# define the model
class MortalityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MedMamba3D(in_chans=1, num_classes=2)
    
    def forward(self, x):
        return self.model(x)

# training loop
def train_model(train_loader, val_loader, fold):
    model = MortalityModel().cuda()
    
    # computing class weights for imbalanced data
    class_counts = Counter([int(sample[1].item()) for sample in train_loader.dataset])
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    weights = torch.tensor([class_weights[label] for label in sorted(class_weights.keys())],
                           dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    criterion = nn.CrossEntropyLoss(weight=weights)

    # use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    patience = 10  
    early_stop_counter = 0

    # training Loop
    for epoch in range(100):  
        model.train()
        epoch_loss = 0

        # lists for AUC calculation
        train_labels_epoch = []
        train_preds_epoch = []
        
        for images, labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Train", leave=False):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # probabilities for positive class 
            probs = F.softmax(outputs, dim=1)[:, 1]
            train_preds_epoch.extend(probs.detach().cpu().numpy())
            train_labels_epoch.extend(labels.detach().cpu().numpy())
        
        # compute train AUC 
        if len(np.unique(train_labels_epoch)) > 1:
            train_auc = roc_auc_score(train_labels_epoch, train_preds_epoch)
        else:
            train_auc = None

        # validation Step
        model.eval()
        val_loss = 0
        val_labels_epoch = []
        val_preds_epoch = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Val", leave=False):
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = F.softmax(outputs, dim=1)[:, 1]
                val_preds_epoch.extend(probs.detach().cpu().numpy())
                val_labels_epoch.extend(labels.detach().cpu().numpy())
        
        if len(np.unique(val_labels_epoch)) > 1:
            val_auc = roc_auc_score(val_labels_epoch, val_preds_epoch)
        else:
            val_auc = None

        train_auc_str = f"{train_auc:.4f}" if train_auc is not None else "N/A"
        val_auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
        print(f'Fold {fold+1} Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Train AUC: {train_auc_str}, Val AUC: {val_auc_str}')
        
        # save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(logdir, f'best_model_fold_{fold}.pth'))
            print(f'Best model saved at epoch {epoch+1}')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(logdir, f'checkpoint_fold_{fold}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch}')
        
        # early stopping
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return model

# cross-Validation Loop (running on all folds)
for fold, (train_val_patient_idx, test_patient_idx) in enumerate(stratified_kfold.split(np.zeros(len(patient_mortality)), patient_mortality)):
    print(f'Fold {fold+1}/{k_folds}')
    
    train_val_patients = patient_mortality.index[train_val_patient_idx]
    test_patients = patient_mortality.index[test_patient_idx]
    
    train_val_data = data[data['Patient'].isin(train_val_patients)]
    test_data = data[data['Patient'].isin(test_patients)]
    
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.2, stratify=patient_mortality.loc[train_val_patients], random_state=42
    )
    
    train_data = train_val_data[train_val_data['Patient'].isin(train_patients)]
    val_data = train_val_data[train_val_data['Patient'].isin(val_patients)]
    
    train_loader = DataLoader(CTScanDataset(prepare_data(train_data)), batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(CTScanDataset(prepare_data(val_data)), batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    
    train_model(train_loader, val_loader, fold)
