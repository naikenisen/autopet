import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data_loader import NiftDataset, get_patients
from src.model import UNet
import wandb
wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")

def dice_coefficient(
    preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Calculates the Dice coefficient for binary segmentation.
    """
    preds = (preds > 0.5).float()
    targets = targets.float()
    preds_flat = preds.contiguous().view(preds.shape[0], -1)
    targets_flat = targets.contiguous().view(targets.shape[0], -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.mean()


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
        wandb.log({
        "batch_loss": loss.item(),
        "epoch_num": epoch_num,
        "num_epochs": num_epochs
        })
        print(
            f"Epoch {epoch_num}/{num_epochs} - Training Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}"
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

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_logits = model(inputs)
            loss = criterion(outputs_logits, targets)
            running_loss += loss.item()
            outputs_probs = torch.sigmoid(outputs_logits)
            dice = dice_coefficient(outputs_probs, targets)
            running_dice += dice.item()

    avg_val_loss = running_loss / num_batches
    avg_val_dice = running_dice / num_batches
    print(f"Validation - Avg Loss: {avg_val_loss:.4f}, Avg Dice: {avg_val_dice:.4f}")
    return avg_val_loss, avg_val_dice


def test_epoch(model, loader, device):
    model.eval()
    running_dice = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_logits = model(inputs)
            outputs_probs = torch.sigmoid(outputs_logits)
            dice = dice_coefficient(outputs_probs, targets)
            running_dice += dice.item()

    avg_test_dice = running_dice / num_batches
    print(f"Test - Avg Dice: {avg_test_dice:.4f}")
    return avg_test_dice

DATASET_PATH = "/work/imvia/in156281/data_unet"
MODEL_SAVE_PATH = "output/models/unet_model.pth"
INPUT_FILENAMES = ["PET.nii.gz", "SEG.nii.gz"]
SLICE_AXIS = 2  # 0: x, 1: y, 2: z
BATCH_SIZE = 20
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.3
NUM_EPOCHS = 20
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_OUTPUT_CHANNELS_MODEL = 1  # Do not change unless we want to detect more labels in the future
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# --- WandB Init ---
wandb.init(
    project="unet_gang",
    config={
        "model_save_path": MODEL_SAVE_PATH,
        "input_filenames": INPUT_FILENAMES,
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

# --- Load Patients ---
patients = get_patients(data_folder_path=DATASET_PATH)

# Sélectionner un quart des patients au hasard
random.seed(RANDOM_SEED)
patients = random.sample(patients, len(patients) // 5)

# --- Dataset Init ---
train_patients, val_patients = train_test_split(
    patients, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
)
train_dataset = NiftDataset(
    patients=train_patients,
    filenames=INPUT_FILENAMES,
    slice_axis=SLICE_AXIS,
)
val_dataset = NiftDataset(
    patients=val_patients,
    filenames=INPUT_FILENAMES,
    slice_axis=SLICE_AXIS,
)

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

# --- Model Init ---
model = UNet(
    n_channels=len(INPUT_FILENAMES), n_classes=NUM_OUTPUT_CHANNELS_MODEL
).to(DEVICE)
print(f"U-Net instantiated.")

# --- Loss Function Init ---
criterion = nn.BCEWithLogitsLoss()
print(f"Loss function: BCEWithLogitsLoss")

# --- Optimizer Init ---
optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
)
print(f"Optimizer: AdamW with lr={LEARNING_RATE}")

# --- Training Loop ---
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
    # Store metrics
    train_losses_history.append(train_loss)
    val_losses_history.append(val_loss)
    val_dice_scores_history.append(val_dice)

    # --- WandB log ---
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