import os
import pandas as pd
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, Resized
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, CacheDataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import torch.nn as nn
from collections import OrderedDict
import numpy as np

data_path = "/projects/b1038/Pulmonary/ksenkow/CLAD_serial_CT/data/6multiplied"
logdir = "./logs/swin_unetr_unfrozen"
os.makedirs(logdir, exist_ok=True)

lr = 1e-4
max_epochs = 100
batch_size = 5
num_classes = 2
patience = 10 


data = pd.read_csv('/projects/b1038/Pulmonary/ksenkow/CLAD_serial_CT/data/v2_analysis/01gather_data/mortality_metadata.csv', index_col=0)
data['path'] = f'{data_path}/' + data['Patient'] + '/' + data['filename'] + '.nii.gz'
data = data[['path', 'mortality_12m', 'Patient']]


def prepare_data(data):
    return [{"image": row["path"], "label": row["mortality_12m"]} for _, row in data.iterrows()]

# stratified group k fold
k_folds = 4
group_labels = data['Patient'].values  
targets = data.groupby("Patient")["mortality_12m"].max().values  

stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# map patients to indices
patient_to_idx = defaultdict(list)
for idx, patient in enumerate(data["Patient"].values):
    patient_to_idx[patient].append(idx)

# k-fold cross-val
for fold, (train_val_patient_idx, test_patient_idx) in enumerate(stratified_kfold.split(np.zeros(len(targets)), targets)):

    print(f"Fold {fold + 1}/{k_folds}")

    train_val_patients = [list(patient_to_idx.keys())[i] for i in train_val_patient_idx]
    test_patients = [list(patient_to_idx.keys())[i] for i in test_patient_idx]

    # create train/test/val splits
    train_val_data = data[data['Patient'].isin(train_val_patients)]
    test_data = data[data['Patient'].isin(test_patients)]

    unique_train_val_patients = train_val_data['Patient'].unique()
    
    # patient only needs one CT positive for 12 month mortality to be placed in that group for train/test split
    patient_mortality_labels = train_val_data.groupby("Patient")["mortality_12m"].max()

    train_patients, val_patients = train_test_split(
        unique_train_val_patients, 
        test_size=0.2, 
        stratify=patient_mortality_labels.loc[unique_train_val_patients],  
        random_state=42
    )


    train_data = train_val_data[train_val_data['Patient'].isin(train_patients)]
    val_data = train_val_data[train_val_data['Patient'].isin(val_patients)]

    # computing class weights for imbalanced data
    class_counts = Counter(train_data['mortality_12m'])
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    weights = torch.tensor([class_weights[label] for label in sorted(class_weights.keys())], dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    loss_function = CrossEntropyLoss(weight=weights)


    train_files = prepare_data(train_data)
    val_files = prepare_data(val_data)
    test_files = prepare_data(test_data)

    # standardization 
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        ToTensord(keys=["image"]),
    ])

    # load data
    train_ds = CacheDataset(data=train_files, transform=transform, cache_rate=0.05, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=transform, cache_rate=0.05, num_workers=4)
    test_ds = CacheDataset(data=test_files, transform=transform, cache_rate=0.05, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # define the model
    model = SwinUNETR(img_size=(96, 96, 96), in_channels=1, out_channels=48, feature_size=48, use_checkpoint=True)

    # classification Head
    model.classification_head = nn.Sequential(
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(48, num_classes)
    )

    # load pretrained weights
    pretrained_path = "/projects/b1038/Pulmonary/ksenkow/CLAD_serial_CT/code/v2_analysis/05mortality_analysis/SWIN_UNETR/ssl_pretrained_weights.pth"
    if not os.path.exists(pretrained_path):
        from monai.apps import download_url
        download_url("https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth", pretrained_path)

    ssl_dict = torch.load(pretrained_path)
    ssl_weights = ssl_dict["model"]

    keys_to_remove = ["encoder.mask_token", "encoder.norm.weight", "encoder.norm.bias", "out.conv.conv.weight", "out.conv.conv.bias"]
    for key in keys_to_remove:
        ssl_weights.pop(key, None)

    monai_loadable_state_dict = OrderedDict()
    for key, value in ssl_weights.items():
        if key.startswith("encoder."):
            new_key = "swinViT." + key[8:] if "patch_embed" in key[8:19] else "swinViT." + key[8:18] + key[20:]
            monai_loadable_state_dict[new_key] = value
        else:
            monai_loadable_state_dict[key] = value

    model.load_state_dict(monai_loadable_state_dict, strict=False)

    # unfreeze Transformer Backbone
    for name, param in model.named_parameters():
#         if "swinViT" in name:
#             param.requires_grad = False
        if "swinViT.layers.3" in name:
            param.requires_grad = True
        elif "swinViT" in name:
            param.requires_grad = False
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

#     optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if "swinViT" in n and p.requires_grad], 'lr': lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if "classification_head" in n], 'lr': lr}
    ], weight_decay=1e-5)


    # training Loop
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model.classification_head(model(inputs))
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = model.classification_head(model(inputs))
                val_loss += loss_function(outputs, labels).item()

        print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(logdir, f"best_model_fold_{fold}.pth"))
            print(f"Best model for fold {fold + 1} saved at epoch {epoch}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(logdir, f"checkpoint_fold_{fold}_epoch_{epoch}.pth"))
            print(f"Checkpoint saved at epoch {epoch} for fold {fold + 1}")

        # early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs for fold {fold + 1}.")
            break
