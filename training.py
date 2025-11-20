import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import UNet
import wandb
from src.config import *
from src.data_loader import train_dataset, val_dataset

wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

wandb.init(
    project="unet_gang",
    config={
        "model_save_path": MODEL_SAVE_PATH,
        "label_dir": LABEL_PATH,
        "pet_images_dir": PET_IMAGE_PATH,
        "ct_images_dir": CT_IMAGE_PATH,
        "slice_axis": SLICE_AXIS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "validation_split": VALIDATION_SPLIT,
        "num_epochs": NUM_EPOCHS,
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
    }
)

def dice_coefficient(preds, targets, epsilon = 1e-6):
    preds = (preds > 0.5).float()
    targets = targets.float()
    preds_flat = preds.contiguous().view(preds.shape[0], -1)
    targets_flat = targets.contiguous().view(targets.shape[0], -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.mean()

def dice_loss(preds, targets, epsilon=1e-6):
    """Dice loss for training (uses soft predictions)"""
    preds = torch.sigmoid(preds)  # Convert logits to probabilities
    preds_flat = preds.contiguous().view(preds.shape[0], -1)
    targets_flat = targets.contiguous().view(targets.shape[0], -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return 1.0 - dice.mean()  # Return loss (1 - dice)

class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss with weighting"""
    def __init__(self, pos_weight=None, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, preds, targets):
        bce = self.bce_loss(preds, targets)
        dice = dice_loss(preds, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def train_epoch(
    model, loader, optimizer, criterion, device, epoch_num, num_epochs
):
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs_logits = model(inputs)
        loss = criterion(outputs_logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Calculate metrics for monitoring
        with torch.no_grad():
            outputs_probs = torch.sigmoid(outputs_logits)
            mean_pred = outputs_probs.mean().item()
            mean_target = targets.mean().item()
            max_pred = outputs_probs.max().item()
            
        wandb.log({
            "batch_loss": loss.item(),
            "epoch_num": epoch_num,
            "num_epochs": num_epochs,
            "mean_prediction": mean_pred,
            "mean_target": mean_target,
            "max_prediction": max_pred
        })
        print(
            f"Epoch {epoch_num}/{num_epochs} - Training Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f} - MeanPred: {mean_pred:.4f} - MeanTarget: {mean_target:.4f}"
        )
    avg_epoch_loss = running_loss / num_batches
    print(
        f"Epoch {epoch_num}/{num_epochs} - Training Completed - Avg Loss: {avg_epoch_loss:.4f}{' '*20}"
    )
    return avg_epoch_loss


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = len(loader)
    
    # Diagnostic counters
    total_target_pixels = 0
    total_positive_targets = 0
    total_positive_preds = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_logits = model(inputs)
            loss = criterion(outputs_logits, targets)
            running_loss += loss.item()
            outputs_probs = torch.sigmoid(outputs_logits)
            dice = dice_coefficient(outputs_probs, targets)
            running_dice += dice.item()
            
            # Collect diagnostic stats
            total_target_pixels += targets.numel()
            total_positive_targets += targets.sum().item()
            total_positive_preds += (outputs_probs > 0.5).sum().item()

    avg_val_loss = running_loss / num_batches
    avg_val_dice = running_dice / num_batches
    
    # Print diagnostics
    target_pos_ratio = total_positive_targets / total_target_pixels * 100
    pred_pos_ratio = total_positive_preds / total_target_pixels * 100
    print(f"Validation - Avg Loss: {avg_val_loss:.4f}, Avg Dice: {avg_val_dice:.4f}")
    print(f"  → Target positives: {target_pos_ratio:.4f}% | Pred positives: {pred_pos_ratio:.4f}%")
    print(f"  → Total positive pixels in targets: {int(total_positive_targets)}/{total_target_pixels}")
    
    return avg_val_loss, avg_val_dice


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if DEVICE.type == "cuda" else False,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if DEVICE.type == "cuda" else False,
)

model = UNet(n_channels=2, n_classes=1).to(DEVICE)

# Calculate positive weight to handle class imbalance
# This will be computed from the training data
print("Calculating class weights from training data...")
total_pixels = 0
positive_pixels = 0
sample_batches = min(50, len(train_loader))  # Sample first 50 batches
for i, (_, targets) in enumerate(train_loader):
    if i >= sample_batches:
        break
    total_pixels += targets.numel()
    positive_pixels += targets.sum().item()

pos_ratio = positive_pixels / total_pixels
neg_ratio = 1.0 - pos_ratio
pos_weight = torch.tensor([neg_ratio / pos_ratio]).to(DEVICE)
print(f"Class distribution: {pos_ratio*100:.4f}% positive pixels")
print(f"Using pos_weight: {pos_weight.item():.2f}")

# Use combined loss
criterion = CombinedLoss(pos_weight=pos_weight, bce_weight=0.5, dice_weight=0.5)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

train_losses_history = []
val_losses_history = []
val_dice_scores_history = []
best_val_dice = -1.0
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch(
        model, train_loader, optimizer, criterion, DEVICE, epoch, NUM_EPOCHS
    )
    print(f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}")
    val_loss, val_dice = validate_epoch(
        model, val_loader, criterion, DEVICE
    )
    print(f"Epoch {epoch}/{NUM_EPOCHS} - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    train_losses_history.append(train_loss)
    val_losses_history.append(val_loss)
    val_dice_scores_history.append(val_dice)
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_dice": val_dice,
        "best_val_dice": best_val_dice
    })

    # Save the model if validation Dice improves
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        wandb.save(MODEL_SAVE_PATH)
        print(
            f"Epoch {epoch}: New best validation Dice: {val_dice:.4f}. Model saved to {MODEL_SAVE_PATH}"
        )
    else:
        print(
            f"Epoch {epoch}: Validation Dice: {val_dice:.4f} (Best: {best_val_dice:.4f})"
        )
print("Training completed. Model saved.")
wandb.finish()