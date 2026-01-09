# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


# ==========================================
# 1. DATA PREPROCESSING
# ==========================================

# Load dataset
df = pd.read_csv('../dataset/hmnist_28_28_RGB.csv')

# Split the DataFrame into features (X) and labels (Y)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

# Initialize RandomOverSampler (We keep this, it is excellent for balancing classes!)
oversample = RandomOverSampler(random_state=42)
X, Y = oversample.fit_resample(X, Y)

# Convert to numpy arrays
X = X.to_numpy()
Y = Y.to_numpy()

# --- IMPORTANT CHANGE FOR PYTORCH ---
# Keras uses: (Height, Width, Channels) -> (28, 28, 3)
# PyTorch uses: (Channels, Height, Width) -> (3, 28, 28)

# 1. Reshape to original image dimensions (N, 28, 28, 3)
X = X.reshape(-1, 28, 28, 3)

# 2. Transpose axes for PyTorch format (N, 3, 28, 28)
# We move the last axis (channels) to the second position
X = np.transpose(X, (0, 3, 1, 2))

# 3. Normalize pixel values to range [0, 1]
X = X / 255.0

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(Y, dtype=torch.long) # In PyTorch, labels must be Long type

# Simple Split (80% train, 20% test) instead of K-Fold
# We use a single split to train one final model for verification
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoaders (equivalent to Keras batch_size handling)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
# We removed the 128 and 256 filter blocks.
# This makes mathematical verification feasible (seconds vs days).

class SkinCancerCNN(nn.Module):
    def __init__(self):
        super(SkinCancerCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First Convolutional Block
            # Equivalent to: Convolution2D(32, (3,3), padding='same')
            # In PyTorch: kernel_size=3 + padding=1 achieves 'same' padding
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),              # Equivalent to activation='relu'
            nn.MaxPool2d(2, 2),     # Reduces size to 14x14
            # Note: We removed BatchNormalization as it complicates verification tools
            
            # Second Convolutional Block
            # Equivalent to: Convolution2D(64, (3,3), padding='same')
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # Reduces size to 7x7
            
            # We stop adding Conv layers here. The original network went deeper.
            # Stopping here makes the network "Verification Friendly".
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            # Reduced Dense layers
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),        # Keeping our original Dropout
            nn.Linear(128, 7)       # 7 Output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model  
model = SkinCancerCNN()


# ==========================================
# 3. TRAINING LOOP
# ==========================================

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler (ReduceLROnPlateau implementation)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

EPOCHS = 15 # Reduced slightly as convergence is faster on this simplified net

print("Starting training...")

for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Validation phase (at the end of each epoch)
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    
    # Update Learning Rate if necessary based on Validation Accuracy
    scheduler.step(val_acc)
    
    # Print epoch stats
    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")


# ==========================================
# 4. SAVE RESULTS (Crucial for SVAI Project)
# ==========================================

# Save the trained model weights
torch.save(model.state_dict(), 'skin_model.pth')
print("\n[SUCCESS] Model saved as 'skin_model.pth'")

# Save data for verification 
# (We only save 20 images that the network classifies CORRECTLY)
model.eval()
images_to_verify = []
labels_to_verify = []
count = 0

print("Extracting images for verification...")
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        # Filter: keep only if prediction matches label
        correct_mask = (predicted == labels)
        
        if correct_mask.any():
            # Convert to NumPy for saving
            valid_imgs = inputs[correct_mask].numpy()
            valid_lbls = labels[correct_mask].numpy()
            
            for img, lbl in zip(valid_imgs, valid_lbls):
                images_to_verify.append(img)
                labels_to_verify.append(lbl)
                count += 1
                if count >= 20: # We only need 20 examples
                    break
        if count >= 20:
            break

# Save as .npy files for Alpha-Beta-CROWN and Auto-LiRPA
X_verify = np.array(images_to_verify)
Y_verify = np.array(labels_to_verify)

np.save('data_X.npy', X_verify)
np.save('data_Y.npy', Y_verify)

print(f"[SUCCESS] Verification dataset ({count} images) saved as 'data_X.npy' and 'data_Y.npy'")