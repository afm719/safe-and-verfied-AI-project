import torch
import numpy as np
import os
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from model_data_defs import load_model

def run_verification():
    print("\n=== EXPERIMENT 2: ROBUSTNESS VERIFICATION (LIGHTING) ===")
    
    # 1. Load Data
    if not os.path.exists("data_X.npy"):
        print("[ERROR] Cannot find data_X.npy. Run the export in your notebook.")
        return

    # Load and ensure correct types (Float32 for images, Long for labels)
    X = np.load("data_X.npy")
    y = np.load("data_Y.npy")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # 2. Load Model
    if not os.path.exists("skin_model.pth"):
        print("[ERROR] Cannot find skin_model.pth. Run the training first.")
        return

    model = load_model("skin_model.pth")
    model.eval()  # <--- IMPORTANT: Disable Dropout and BatchNorm for verification

    # Replace Dropout with Identity to avoid errors in auto_LiRPA
    def replace_dropout(m):
        for name, child in m.named_children():
            if isinstance(child, (torch.nn.Dropout, torch.nn.Dropout2d)):
                setattr(m, name, torch.nn.Identity())
            else:
                replace_dropout(child)
    replace_dropout(model)
    
    # 3. Prepare Verification Engine (LiRPA)
    print("Initializing LiRPA...")
    # Dummy input to trace the graph
    dummy_input = torch.zeros_like(X_tensor[0:1])
    bounded_model = BoundedModule(model, dummy_input, bound_opts={'relu': 'same-slope'})
    
    # 4. Experiment Configuration
    DELTA = 0.04  # Perturbation (+/- 4% brightness)
    print(f"Property: Invariance to light changes (Delta = {DELTA})")
    
    safe_count = 0
    total = len(X)
    
    print(f"\nVerifying {total} images...")
    print("-" * 50)
    
    # Open file to save failure margins
    f_margins = open("../results/failure_margins.txt", "w")
    f_margins.write("Image_Index, Class, Failure_Margin\n")
    
    for i in range(total):
        image = X_tensor[i:i+1] # Batch size 1
        label = y_tensor[i].item()
        
        # Define perturbation limits (Variable global brightness)
        # The image can vary between [original - delta, original + delta]
        x_L = torch.clamp(image - DELTA, min=0.0)
        x_U = torch.clamp(image + DELTA, max=1.0)
        
        # Wrap in perturbed tensor
        ptb = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
        bounded_image = BoundedTensor(image, ptb)
        
        # Compute safe limits (Lower Bound and Upper Bound)
        lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method="CROWN")
        
        # Verification Logic:
        # The model is SAFE if the MINIMUM score of the real class (lb[label])
        # is greater than the MAXIMUM score of any other class (ub[others]).
        
        real_score_min = lb[0, label]
        
        # Mask the real class to find the maximum of the others
        mask = torch.ones_like(ub[0], dtype=torch.bool)
        mask[label] = False
        other_score_max = torch.max(ub[0][mask])
        
        if real_score_min > other_score_max:
            print(f"Img {i}: [SAFE] (Class {label})")
            safe_count += 1
        else:
            # Calculate failure margin for curiosity
            margin = other_score_max - real_score_min
            print(f"Img {i}: [UNSAFE] (Class {label}) - Failure margin: {margin.item():.4f}")
            f_margins.write(f"{i}, {label}, {margin.item():.4f}\n")

    f_margins.close()
    print("-" * 50)
    print(f"RESULT: {safe_count}/{total} images certified.")
    print(f"Safety Rate: {(safe_count/total)*100:.1f}%")

if __name__ == "__main__":
    run_verification()