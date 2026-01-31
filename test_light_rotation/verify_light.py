import sys
import torch
import numpy as np
import os
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
sys.path.insert(0, '../test_noise')
from model_data_defs import load_model



"""
=============================================================================
THEORY: FORMAL VERIFICATION VIA BOUND PROPAGATION (LiRPA)
=============================================================================
References: 
   - Xu et al., "Fast and Complete: Enabling High-Performance Artifical Neural 
     Network Verification with Efficient Crown" (NeurIPS 2020).
   - Auto_LiRPA Library: https://github.com/Verified-Intelligence/auto_LiRPA

OBJECTIVE:
   To mathematically CERTIFY that the neural network is invariant to global 
   illumination changes. Unlike standard testing (which tries random points), 
   this method guarantees that *no* perturbation within a range exists that 
   can flip the prediction.

MATHEMATICAL FORMULATION (Robustness Property):
   We define a global brightness perturbation δ (delta).
   For a given input image x and ground truth label y, we define a verification 
   region (Hyper-rectangle):
       
       x' ∈ [x - δ, x + δ]
   
   The goal is to prove that for ALL x' in this infinite set:
       
       Score(y) > Score(j)   ∀ j ≠ y (for all other classes)

METHODOLOGY: CROWN (Linear Relaxation)
   Neural networks use non-linear activations (ReLU). To verify them efficiently, 
   we use "Linear Relaxation".
   
   A. Input Bounds: We define the input intervals L (lower) and U (upper).
   B. Propagation: We propagate these intervals layer by layer. 
      For ReLU(x), we enclose the non-linear function within two linear lines 
      (upper and lower lines).
   C. Output Bounds: This results in a guaranteed Lower Bound for the true class 
      (LB_y) and an Upper Bound for other classes (UB_other).
   
   D. Certification Condition:
      IF (LB_y - UB_other) > 0 THEN:
          The image is PROVEN SAFE (Certified).
      ELSE:
          The image is UNSAFE (Verification failed, a counter-example may exist).

IMPLEMENTATION:
   - PerturbationLpNorm(norm=inf): Defines the L-infinity ball constraints.
   - BoundedModule: Wraps the PyTorch model to enable bound computations.
   - compute_bounds(method='CROWN'): Executes the backward bound propagation.
=============================================================================
"""


def run_verification():
    print("\n=== EXPERIMENT 2: ROBUSTNESS VERIFICATION (LIGHTING) ===")
    
    # Load Data
    if not os.path.exists("../code/data_X.npy"):
        print("[ERROR] Cannot find data_X.npy. Run the export in your notebook.")
        return

    # Load and ensure correct types (Float32 for images, Long for labels)
    X = np.load("data_X.npy")
    y = np.load("data_Y.npy")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Load Model
    if not os.path.exists("../code/skin_model.pth"):
        print("[ERROR] Cannot find skin_model.pth. Run the training first.")
        return

    model = load_model("skin_model.pth")
    model.eval()  

    # Replace Dropout with Identity to avoid errors in auto_LiRPA
    def replace_dropout(m):
        for name, child in m.named_children():
            if isinstance(child, (torch.nn.Dropout, torch.nn.Dropout2d)):
                setattr(m, name, torch.nn.Identity())
            else:
                replace_dropout(child)
    replace_dropout(model)
    
    # Prepare Verification Engine (LiRPA)
    print("Initializing LiRPA...")
    # Dummy input to trace the graph
    dummy_input = torch.zeros_like(X_tensor[0:1])
    bounded_model = BoundedModule(model, dummy_input, bound_opts={'relu': 'same-slope'})
    
    # Experiment Configuration
    # Test multiple delta values to decide best one
    DELTAS = [0.0005, 0.001, 0.002, 0.005]  # adjust list as needed

    total = len(X)
    results_dir = os.path.join("..", "results")
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "delta_results.csv")
    with open(summary_path, "w") as f_summary:
        f_summary.write("Delta,Safe_Count,Total,Safety_Rate\n")

        best_delta = None
        best_rate = -1.0

        for DELTA in DELTAS:
            print(f"\nProperty: Invariance to light changes (Delta = {DELTA})")
            safe_count = 0

            print(f"\nVerifying {total} images for delta {DELTA}...")
            print("-" * 50)

            # Open file to save failure margins for this delta
            fname = os.path.join(results_dir, f"failure_margins_delta_{DELTA:.6f}.txt")
            with open(fname, "w") as f_margins:
                f_margins.write("Image_Index,Class,Failure_Margin\n")

                for i in range(total):
                    image = X_tensor[i:i+1] # Batch size 1
                    label = y_tensor[i].item()

                    # Define perturbation limits (Variable global brightness)
                    x_L = torch.clamp(image - DELTA, min=0.0)
                    x_U = torch.clamp(image + DELTA, max=1.0)

                    # Wrap in perturbed tensor
                    ptb = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
                    bounded_image = BoundedTensor(image, ptb)

                    # Compute safe limits (Lower Bound and Upper Bound)
                    lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method="CROWN")

                    real_score_min = lb[0, label]

                    # Mask the real class to find the maximum of the others
                    mask = torch.ones_like(ub[0], dtype=torch.bool)
                    mask[label] = False
                    other_score_max = torch.max(ub[0][mask])

                    print(f"Img {i}: Class {label} - ", end="")
                    if real_score_min > other_score_max:
                        safe_count += 1
                        print(f"[CERTIFIED SAFE]")
                        f_margins.write(f"{i},{label},0.0\n")
                    else:
                        margin = other_score_max - real_score_min
                        print(f"[UNSAFE] Margin: {margin.item():.6f}")
                        f_margins.write(f"{i},{label},{margin.item():.6f}\n")

            # Calculate safety rate

            safety_rate = safe_count / total
            f_summary.write(f"{DELTA},{safe_count},{total},{safety_rate:.6f}\n")
            print("-" * 50)
            print(f"Delta {DELTA}: {safe_count}/{total} images certified. Safety Rate: {safety_rate*100:.2f}%")

            # Choose best: highest safety rate, tie-breaker smaller delta
            if (safety_rate > best_rate) or (abs(safety_rate - best_rate) < 1e-12 and (best_delta is None or DELTA < best_delta)):
                best_rate = safety_rate
                best_delta = DELTA

    print("\n=== SUMMARY ===")
    if best_delta is not None:
        print(f"Best Delta: {best_delta} with Safety Rate: {best_rate*100:.2f}%")
        print(f"Full summary saved to: {summary_path}")
    else:
        print("No delta evaluated.")

if __name__ == "__main__":
    run_verification()