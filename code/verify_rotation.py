import torch
import numpy as np
import os
import kornia.geometry.transform as K
from model_data_defs import load_model

def run_rotation_verification():
    print("\n=== EXPERIMENT 3: GEOMETRIC VERIFICATION (ROTATION) ===")
    
    # 1. Load Data and Model
    if not os.path.exists("data_X.npy"):
        print("[ERROR] Cannot find data_X.npy.")
        return

    X = np.load("data_X.npy")
    y = np.load("data_Y.npy")
    
    # Convert to tensors (Float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Load model (using the same definition we fixed)
    model = load_model("skin_model.pth")
    
    # 2. Experiment Configuration
    ROTATION_LIMIT = 15.0  # Test rotations up to +/- 15 degrees
    STEP_SIZE = 0.5        # Test every 0.5 degrees
    
    # Generate list of angles: -15, -14.5, ..., 0, ..., 14.5, 15
    steps = int((ROTATION_LIMIT * 2) / STEP_SIZE) + 1
    angles_to_test = torch.linspace(-ROTATION_LIMIT, ROTATION_LIMIT, steps)
    
    print(f"Property: Rotation Invariance (+/- {ROTATION_LIMIT}º)")
    print(f"Resolution: Verifying {len(angles_to_test)} angles per image.")
    
    # Prepare results file
    results_to_save = []
    results_to_save.append("ImageID,Class,Status,FailAngle\n")
    
    safe_count = 0
    total = len(X)
    
    print("-" * 60)
    
    for i in range(total):
        # Base image: (1, 3, 32, 32)
        image = X_tensor[i:i+1] 
        original_label = y_tensor[i].item()
        
        # Validate original prediction (no rotation)
        with torch.no_grad():
            base_pred = model(image).argmax(dim=1).item()
        
        # If model already fails on original, it doesn't count as rotation failure, but base failure
        if base_pred != original_label:
            print(f"Img {i}: [SKIP] Model already fails on original image.")
            results_to_save.append(f"{i},{original_label},SKIP,0\n")
            continue

        is_stable = True
        failed_angle = 0.0
        
        # Rotation loop
        for angle in angles_to_test:
            # Create angle tensor
            angle_t = torch.tensor([angle], dtype=torch.float32)
            
            # Rotate image (Bilinear is standard for photos)
            rotated_img = K.rotate(image, angle_t, mode='bilinear')
            
            with torch.no_grad():
                output = model(rotated_img)
                pred = output.argmax(dim=1).item()
            
            # Verification
            if pred != original_label:
                is_stable = False
                failed_angle = angle.item()
                # Break at first failure found
                break 
        
        # Report and Save
        if is_stable:
            print(f"Img {i} (Class {original_label}): [SAFE] - Stable between -{ROTATION_LIMIT}º and +{ROTATION_LIMIT}º")
            safe_count += 1
            results_to_save.append(f"{i},{original_label},SAFE,0\n")
        else:
            print(f"Img {i} (Class {original_label}): [UNSAFE]  - Fails at rotation {failed_angle:.1f}º")
            results_to_save.append(f"{i},{original_label},UNSAFE,{failed_angle:.1f}\n")

    # Save file
    with open("../results/results_rotation.txt", "w") as f:
        f.writelines(results_to_save)

    print("-" * 60)
    print(f"FINAL RESULT: {safe_count}/{total} robust images.")
    print(f"[INFO] Data saved in 'results_rotation.txt'")

if __name__ == "__main__":
    run_rotation_verification()