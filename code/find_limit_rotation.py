import torch
import numpy as np
import os
import kornia.geometry.transform as K
import matplotlib.pyplot as plt
import seaborn as sns
from model_data_defs import load_model

"""
=============================================================================
THEORY: GEOMETRIC ROBUSTNESS ANALYSIS (ROTATION INVARIANCE)
=============================================================================
Context: 
   Convolutional Neural Networks (CNNs) are not inherently invariant to 
   rotation unless specifically trained for it (e.g., via data augmentation 
   or Spatial Transformer Networks).

OBJECTIVE:
   To determine the maximum angle of rotation (θ_max) that an input image 
   can withstand before the model's prediction becomes incorrect. This creates 
   a "Robustness Curve" for the dataset.

MATHEMATICAL FORMULATION:
   A rotation transforms the pixel coordinates (x, y) of the input image to 
   new coordinates (x', y') using the Rotation Matrix R(θ):

       [x']     [ cos θ   -sin θ ]   [x]
       [  ]  =  [                ] * [ ]
       [y']     [ sin θ    cos θ ]   [y]

   Since (x', y') rarely fall on integer grid coordinates, the pixel intensity 
   must be estimated using Interpolation (typically Bilinear).

   Bilinear Interpolation Formula:
       I(x', y') ≈ w1*I11 + w2*I12 + w3*I21 + w4*I22
   
   * Note: The interpolation process itself introduces "noise" (aliasing/blur) 
     which can act as an adversarial perturbation, confusing the CNN.

METHODOLOGY: ITERATIVE SEARCH (GRID SEARCH)
   Unlike the LiRPA experiment (which proves bounds), geometric verification 
   is empirically tested because rotating the grid changes the structural 
   dependencies of the pixels, making formal bounding extremely computationally 
   expensive.
   
   Algorithm:
   1. Start with angle θ = 0.
   2. Expand the search range: ±0.5º, ±1.0º, ±1.5º...
   3. At each step, apply the Affine Transformation to the tensor.
   4. Stop when Prediction(Rotated_Image) ≠ Ground_Truth.
   5. Record the last safe angle as the "Robustness Limit".

IMPLEMENTATION:
   - Uses `kornia.geometry.transform.rotate` for differentiable affine transforms.
   - mode='bilinear': Standard interpolation for photographic images.
=============================================================================
"""

def find_robustness_limit():
    print("\n=== EXPERIMENT 3 (ADVANCED): ROTATION LIMIT SEARCH ===")
    
    # Load Data and Model
    if not os.path.exists("data_X.npy"):
        print("[ERROR] Cannot find data_X.npy")
        return

    X = np.load("data_X.npy")
    y = np.load("data_Y.npy")
    
    # Convert to tensors (Float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    model = load_model("skin_model.pth")
    
    #  Search Configuration
    MAX_ANGLE = 45.0   # Maximum angle to test (if passed, it's considered ultra-robust)
    STEP = 0.5         # Search precision (step size in degrees)
    
    # List to store the maximum safe angle for each image
    max_safe_angles = []
    
    print(f"Searching breaking points for {len(X)} images...")
    print(f"Search range: 0º to {MAX_ANGLE}º (Step: {STEP}º)")
    print("-" * 60)
    
    for i in range(len(X)):
        image = X_tensor[i:i+1]
        label = y_tensor[i].item()
        
        # Validate base image (0 degrees)
        with torch.no_grad():
            base_pred = model(image).argmax(dim=1).item()
        
        if base_pred != label:
            print(f"Img {i}: [SKIP] Fails at 0º (Original is misclassified)")
            max_safe_angles.append(0.0) # Resistance is 0
            continue
            
        # Iterative Search (Expansion)
        current_limit = 0.0
        broke = False
        
        # Test incremental angles: 0.5, 1.0, 1.5 ... up to MAX_ANGLE
        for angle in np.arange(STEP, MAX_ANGLE + STEP, STEP):
            # Check both positive (+angle) and negative (-angle) rotation
            angles_to_check = [angle, -angle]
            
            check_passed = True
            for a in angles_to_check:
                angle_t = torch.tensor([float(a)], dtype=torch.float32)
                # 'bilinear' is standard for images, 'nearest' avoids blurring
                rotated_img = K.rotate(image, angle_t, mode='bilinear')
                
                with torch.no_grad():
                    pred = model(rotated_img).argmax(dim=1).item()
                
                if pred != label:
                    check_passed = False
                    break # Failed at this level
            
            if check_passed:
                current_limit = angle # Update record
            else:
                broke = True
                print(f"Img {i} (Class {label}): Withstands up to +/- {current_limit}º (Breaks at {angle}º)")
                break # Exit loop for this image
        
        if not broke:
            print(f"Img {i} (Class {label}): [SUPER ROBUST] Withstands over +/- {MAX_ANGLE}º")
            current_limit = MAX_ANGLE
            
        max_safe_angles.append(current_limit)

    np.savetxt("../results/robustness_limits.txt", max_safe_angles, fmt='%.1f', header="MaxSafeAngle")
    
    # Generate "Robustness Curve" (Survival Plot)
    # This shows what % of images survive X degrees of rotation
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Sort angles from low to high
    sorted_angles = np.sort(max_safe_angles)
    # Calculate survival percentage (1.0 -> 0.0)
    y_vals = 1.0 - (np.arange(len(sorted_angles)) / float(len(sorted_angles)))
    
    # Add point (0, 1.0) so the graph starts correctly
    sorted_angles = np.concatenate(([0], sorted_angles))
    y_vals = np.concatenate(([1.0], y_vals))
    
    # Plotting
    plt.step(sorted_angles, y_vals, where='post', linewidth=3, color='blue')
    plt.fill_between(sorted_angles, y_vals, alpha=0.3, color='blue')
    
    plt.title('Geometric Robustness Curve: How much rotation can the model handle?', fontsize=14)
    plt.xlabel('Rotation Angle (Degrees)', fontsize=12)
    plt.ylabel('Success Rate (Accuracy)', fontsize=12)
    plt.xlim(0, 20) # Showing up to 20 degrees is reasonable for this context
    plt.ylim(0, 1.1)
    
    # Reference lines
    plt.axvline(x=5, color='green', linestyle='--', label='Slight Rotation (5º)')
    plt.axvline(x=10, color='orange', linestyle='--', label='Moderate Rotation (10º)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('../plots/robustness_curve.png')
    
    print("-" * 60)
    print("Plot generated: 'robustness_curve_rotation.png'")
    print(f"Data saved in 'robustness_limits.txt'")

if __name__ == "__main__":
    find_robustness_limit()