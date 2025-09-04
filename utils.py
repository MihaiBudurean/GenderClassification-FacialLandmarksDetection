import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import seaborn as sns
import copy
import math


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Seed: {seed}')

def print_lr(optimizer):
    lrs = [group['lr'] for group in optimizer.param_groups]
    print("Current LR(s):", ", ".join(f"{lr:.6f}" for lr in lrs))
def compute_nme(pred_landmarks, true_landmarks, image_size=224):
    """
    Normalized Mean Error (NME) for landmark prediction.
    pred_landmarks, true_landmarks: tensors of shape (batch_size, 6)
    Returns: scalar NME
    """
    pred = pred_landmarks.view(-1, 3, 2)
    target = true_landmarks.view(-1, 3, 2)
    error = torch.norm(pred - target, dim=2).mean(dim=1)  # (batch,)
    norm_factor = image_size  # could be inter-ocular distance if needed
    nme = (error / norm_factor).mean().item()
    return nme


def visualize_landmarks(img_tensor, landmarks, title="Landmarks", color='red', ax=None):
    """
    Visualize landmarks on a given image tensor (C,H,W) on a provided matplotlib axis.
    """
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    landmarks = landmarks.view(3, 2).detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c=color)
    ax.set_title(title)
    ax.axis('off')


import matplotlib.pyplot as plt

def plot_training_history(history, title="Training History", ylim=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def train_epoch_gender(model, dataloader, loss_fn, optimizer, device, train=True):
    model.train() if train else model.eval()
    running_loss = 0.0

    for inputs, targets in dataloader:
        gender = targets[0].float().unsqueeze(1).to(device)
        inputs = inputs.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            loss = loss_fn(outputs, gender)
            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def train_epoch_landmark(model, dataloader, loss_fn, optimizer, device, train=True):
    model.train() if train else model.eval()
    running_loss = 0.0

    for inputs, targets in dataloader:
        landmarks = targets[1][:, :6].float().to(device)
        inputs = inputs.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            loss = loss_fn(outputs, landmarks)
            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def train_epoch_multitask(model, dataloader, loss_fn_cls, loss_fn_lmk, optimizer, device, loss_weights=None, train=True):
    model.train() if train else model.eval()
    running_loss = 0.0

    w_cls = loss_weights.get('cls', 1.0) if loss_weights else 1.0
    w_lmk = loss_weights.get('lmk', 1.0) if loss_weights else 1.0

    for inputs, targets in dataloader:
        gender = targets[0].float().unsqueeze(1).to(device)
        landmarks = targets[1][:, :6].float().to(device)
        inputs = inputs.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out_gender, out_landmarks = model(inputs)
            loss_cls = loss_fn_cls(out_gender, gender)
            loss_lmk = loss_fn_lmk(out_landmarks, landmarks)
            loss = w_cls * loss_cls + w_lmk * loss_lmk
            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def train_epoch_multitask_improved(
    model, dataloader, loss_fn_cls, loss_fn_lmk, optimizer, device, 
    loss_weights=None, scheduler=None, train=True
):
    model.train() if train else model.eval()
    running_loss = 0.0

    w_cls = loss_weights.get('cls', 1.0) if loss_weights else 1.0
    w_lmk = loss_weights.get('lmk', 1.0) if loss_weights else 1.0

    for inputs, targets in dataloader:
        gender = targets[0].float().unsqueeze(1).to(device)
        landmarks = targets[1][:, :6].float().to(device)
        inputs = inputs.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out_gender, out_landmarks = model(inputs)
            loss_cls = loss_fn_cls(out_gender, gender)
            loss_lmk = loss_fn_lmk(out_landmarks, landmarks)
            loss = w_cls * loss_cls + w_lmk * loss_lmk
            if train:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def train_model(
    model,
    dataloaders,
    task_type,
    loss_fn_cls=None,
    loss_fn_lmk=None,
    optimizer=None,
    scheduler=None,
    num_epochs=50,
    device='cuda',
    loss_weights=None,
    save_path='best_model.pth',
    early_stopping_patience=10,
    verbose=True,
    improved_multitask=False  # <-- NEW ARGUMENT
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': []}

    # Choose function handles
    if task_type == 'gender':
        train_fn = train_epoch_gender

    elif task_type == 'landmark':
        train_fn = train_epoch_landmark

    elif task_type == 'multitask':
        if improved_multitask:
            def train_fn(model, dataloader, loss_fn, optimizer, device, train=True, scheduler=None):
                loss_fn_cls, loss_fn_lmk = loss_fn
                return train_epoch_multitask_improved(
                    model=model,
                    dataloader=dataloader,
                    loss_fn_cls=loss_fn_cls,
                    loss_fn_lmk=loss_fn_lmk,
                    optimizer=optimizer,
                    device=device,
                    loss_weights=loss_weights,
                    scheduler=scheduler,
                    train=train
                )
        else:
            def train_fn(model, dataloader, loss_fn, optimizer, device, train=True, scheduler=None):
                loss_fn_cls, loss_fn_lmk = loss_fn
                return train_epoch_multitask(
                    model=model,
                    dataloader=dataloader,
                    loss_fn_cls=loss_fn_cls,
                    loss_fn_lmk=loss_fn_lmk,
                    optimizer=optimizer,
                    device=device,
                    loss_weights=loss_weights,
                    train=train
                )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print_lr(optimizer)

        # Pass scheduler to train_fn only for improved multitask train
        if task_type == 'multitask' and improved_multitask:
            train_loss = train_fn(model, dataloaders['train'],
                                  loss_fn_cls if task_type != 'landmark' else loss_fn_lmk,
                                  optimizer, device, train=True, scheduler=scheduler)
            val_loss = train_fn(model, dataloaders['val'],
                                loss_fn_cls if task_type != 'landmark' else loss_fn_lmk,
                                optimizer, device, train=False, scheduler=None)
        else:
            train_loss = train_fn(model, dataloaders['train'],
                                  loss_fn_cls if task_type != 'landmark' else loss_fn_lmk,
                                  optimizer, device, train=True)
            val_loss = train_fn(model, dataloaders['val'],
                                loss_fn_cls if task_type != 'landmark' else loss_fn_lmk,
                                optimizer, device, train=False)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if verbose:
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Scheduler logic stays the same
        if scheduler and not (task_type == 'multitask' and improved_multitask):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            save_model(model, save_path)
            if verbose:
                print("Model improved — saved.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model, history


# Shared backbone without final FC
class BaseResNet18Encoder(nn.Module):
    def __init__(self):
        super(BaseResNet18Encoder, self).__init__()
        resnet = resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Drop final FC
        self.out_features = resnet.fc.in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x


class BaseResNet18EncoderPretrained(nn.Module):
    def __init__(self):
        super(BaseResNet18EncoderPretrained, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_features = resnet.fc.in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


# Gender classification head
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.encoder = BaseResNet18Encoder()
        self.classifier = nn.Linear(self.encoder.out_features, 1)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


# Landmark regression head (3 keypoints: 6 coords)
class LandmarkRegressor(nn.Module):
    def __init__(self):
        super(LandmarkRegressor, self).__init__()
        self.encoder = BaseResNet18Encoder()
        self.regressor = nn.Linear(self.encoder.out_features, 6)

    def forward(self, x):
        x = self.encoder(x)
        return self.regressor(x)


# Multi-task: one encoder, two heads
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.encoder = BaseResNet18Encoder()
        self.classifier = nn.Linear(self.encoder.out_features, 1)
        self.regressor = nn.Linear(self.encoder.out_features, 6)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x), self.regressor(x)


class MultiTaskModelImproved(nn.Module):
    def __init__(self):
        super(MultiTaskModelImproved, self).__init__()
        self.encoder = BaseResNet18EncoderPretrained()
        self.classifier = nn.Linear(self.encoder.out_features, 1)
        self.regressor = nn.Linear(self.encoder.out_features, 6)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x), self.regressor(x)


class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0, reduction='mean'):
        """
        Wing loss for landmark regression.
        - w: width of the nonlinear part
        - epsilon: transition point
        """
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        x = pred - target
        abs_x = torch.abs(x)
        loss = torch.where(
            abs_x < self.w,
            self.w * torch.log(1 + abs_x / self.epsilon),
            abs_x - self.c()
        )
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def c(self):
        return self.w - self.w * math.log(1 + self.w / self.epsilon)
    

def evaluate_gender_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            gender = targets[0].to(device).float().unsqueeze(1)
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(gender.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    acc = accuracy_score(all_labels, all_preds >= 0.5)
    auc_score = roc_auc_score(all_labels, all_preds)

    print(f"Gender Classification — Accuracy: {acc:.4f} | ROC AUC: {auc_score:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds >= 0.5)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Gender Classification")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()



def evaluate_landmark_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    sample_img = None
    sample_pred = None
    sample_gt = None

    with torch.no_grad():
        for inputs, targets in dataloader:
            landmarks = targets[1][:, :6].float().to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)

            all_preds.append(outputs.cpu())
            all_targets.append(landmarks.cpu())

            # Save first image for visualization
            sample_img = inputs[0].cpu()
            sample_pred = outputs[0].cpu()
            sample_gt = landmarks[0].cpu()
            break  # Just take one batch

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = nn.functional.mse_loss(preds, targets).item()
    nme = compute_nme(preds, targets)

    print(f"Landmark Detection — MSE: {mse:.4f} | NME: {nme:.4f}")

    # Visualize prediction vs ground truth side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    visualize_landmarks(sample_img, sample_pred, title="Predicted Landmarks", color='red', ax=axs[0])
    visualize_landmarks(sample_img, sample_gt, title="Ground Truth Landmarks", color='green', ax=axs[1])
    plt.tight_layout()
    plt.show()


def evaluate_multitask_model(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_preds_lmk = []
    all_targets_lmk = []

    # For visualization
    sample_img = None
    sample_pred_lmk = None
    sample_true_lmk = None

    with torch.no_grad():
        for inputs, targets in dataloader:
            gender = targets[0].float().unsqueeze(1).to(device)
            landmarks = targets[1][:, :6].float().to(device)
            inputs = inputs.to(device)

            out_gender, out_landmarks = model(inputs)
            probs = torch.sigmoid(out_gender)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(gender.cpu().numpy())
            all_preds_lmk.append(out_landmarks.cpu())
            all_targets_lmk.append(landmarks.cpu())

            # Save 1 sample for visualization
            if sample_img is None:
                sample_img = inputs[0].cpu()
                sample_pred_lmk = out_landmarks[0].cpu()
                sample_true_lmk = landmarks[0].cpu()

    # GENDER METRICS
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    acc = accuracy_score(all_labels, all_probs >= 0.5)
    auc_score = roc_auc_score(all_labels, all_probs)
    print(f"[GENDER] Accuracy: {acc:.4f} | ROC AUC: {auc_score:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_probs >= 0.5)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Gender Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Gender Classification")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # LANDMARK METRICS
    preds_lmk = torch.cat(all_preds_lmk, dim=0)
    targets_lmk = torch.cat(all_targets_lmk, dim=0)
    mse = nn.functional.mse_loss(preds_lmk, targets_lmk).item()
    nme = compute_nme(preds_lmk, targets_lmk)
    print(f"[LANDMARK] MSE: {mse:.4f} | NME: {nme:.4f}")

    # Landmark visualization
    if sample_img is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        visualize_landmarks(sample_img, sample_pred_lmk, title="Predicted Landmarks", color='red', ax=axs[0])
        visualize_landmarks(sample_img, sample_true_lmk, title="Ground Truth Landmarks", color='green', ax=axs[1])
        plt.tight_layout()
        plt.show()
