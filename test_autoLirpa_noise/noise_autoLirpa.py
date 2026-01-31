import sys
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
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
   To mathematically CERTIFY that the neural network is invariant to the noise.
   Unlike standard testing (which tries random points), 
   this method guarantees that *no* perturbation within a range exists that 
   can flip the prediction.

MATHEMATICAL FORMULATION (Robustness Property):
   We define a global a perturbation ε (epsilon).
   For a given input image x and ground truth label y, we define a verification 
   region (Hyper-rectangle):
       
       x' ∈ [x - ε, x + ε]
   
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
    X = np.load("../code/data_X.npy")
    y = np.load("../code/data_Y.npy")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Load Model
    if not os.path.exists("../code/skin_model.pth"):
        print("[ERROR] Cannot find skin_model.pth. Run the training first.")
        return

    model = load_model("../code/skin_model.pth")
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
    
    EPSILONS = [0.0005, 0.001, 0.002, 0.005]  

    total = len(X)
    results_dir = os.path.join("..", "results_rotation_and_autoLirpa")
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "epsilon_results.csv")
    with open(summary_path, "w") as f_summary:
        f_summary.write("Epsilon,Safe_Count,Total,Safety_Rate\n")

        best_epsilon = None
        best_rate = -1.0

        for EPSILON in EPSILONS:
            print(f"\nProperty: Invariance to light changes (Epsilon = {EPSILON})")
            safe_count = 0

            print(f"\nVerifying {total} images for epsilon {EPSILON}...")
            print("-" * 50)

            # Open file to save failure margins for this epsilon
            fname = os.path.join(results_dir, f"failure_margins_epsilon_{EPSILON:.6f}.txt")
            with open(fname, "w") as f_margins:
                f_margins.write("Image_Index,Class,Failure_Margin\n")

                for i in range(total):
                    image = X_tensor[i:i+1] # Batch size 1
                    label = y_tensor[i].item()

                    # Define perturbation limits (Variable global brightness)
                    x_L = torch.clamp(image - EPSILON, min=0.0)
                    x_U = torch.clamp(image + EPSILON, max=1.0)

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
            f_summary.write(f"{EPSILON},{safe_count},{total},{safety_rate:.6f}\n")
            print("-" * 50)
            print(f"Epsilon {EPSILON}: {safe_count}/{total} images certified. Safety Rate: {safety_rate*100:.2f}%")

            # Choose best: highest safety rate, tie-breaker smaller epsilon
            if (safety_rate > best_rate) or (abs(safety_rate - best_rate) < 1e-12 and (best_epsilon is None or EPSILON < best_epsilon)):
                best_rate = safety_rate
                best_epsilon = EPSILON

    print("\n=== SUMMARY ===")
    if best_epsilon is not None:
        print(f"Best Epsilon: {best_epsilon} with Safety Rate: {best_rate*100:.2f}%")
        print(f"Full summary saved to: {summary_path}")
    else:
        print("No epsilon evaluated.")

    target_idx = 0
    target_image = X_tensor[target_idx:target_idx+1]
    target_label = y_tensor[target_idx].item()
    
    print(f"Analyzing Image {target_idx} (True Class: {target_label}) for crossing point...")
    
    plot_epsilons = np.linspace(0.0, 0.001, 50)    
    lb_true_class = []
    ub_other_class = []
    
    for d in plot_epsilons:
        x_L = torch.clamp(target_image - d, min=0.0)
        x_U = torch.clamp(target_image + d, max=1.0)
        
        ptb = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
        bounded_img = BoundedTensor(target_image, ptb)
        
        lb, ub = bounded_model.compute_bounds(x=(bounded_img,), method="CROWN")
        
        val_true = lb[0, target_label].item()
        
        mask = torch.ones_like(ub[0], dtype=torch.bool)
        mask[target_label] = False
        val_other = torch.max(ub[0][mask]).item()
        
        lb_true_class.append(val_true)
        ub_other_class.append(val_other)

    plt.figure(figsize=(10, 6))
    plt.plot(plot_epsilons, lb_true_class, 'b-', linewidth=2, label='Lower Bound (True Class)')
    plt.plot(plot_epsilons, ub_other_class, 'r--', linewidth=2, label='Upper Bound (Max Error)')
    
    lb_arr = np.array(lb_true_class)
    ub_arr = np.array(ub_other_class)
    
    plt.fill_between(plot_epsilons, lb_arr, ub_arr, where=(lb_arr > ub_arr),
                     color='green', alpha=0.15, label='Certified Safe')
    plt.fill_between(plot_epsilons, lb_arr, ub_arr, where=(lb_arr <= ub_arr),
                     color='red', alpha=0.15, label='Unsafe Region')
    
    plt.xlabel('Epsilon perturbation ($\epsilon$)')
    plt.ylabel('Logits Output')
    plt.title(f'Verification Bounds Analysis - Image {target_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(results_dir, "bounds_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")


    

if __name__ == "__main__":
    run_verification()