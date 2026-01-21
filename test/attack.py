import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime
sys.path.insert(0, '../code')
from model_data_defs import load_model

"""
=============================================================================
THEORY: FGSM (Fast Gradient Sign Method)
=============================================================================
Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)

OBJECTIVE:
   To generate an "Adversarial Example" (x_adv) that is perceptually identical 
   to the original image (x) but maximizes the neural network's classification error.
   This is a "White-Box" attack, meaning it requires full access to the model's 
   gradients and architecture.

MATHEMATICAL FORMULATION:
   In standard training (Gradient Descent), we minimize the loss function J 
   by updating the model weights (θ). In an adversarial attack, we freeze the 
   weights and MAXIMIZE the loss J by modifying the input image (x).

   The perturbation formula is:

       x_adv = x + epsilon * sign( ∇x J(θ, x, y) )

   Where:
     - x       : Original clean image.
     - x_adv   : Generated adversarial image.
     - epsilon : Perturbation magnitude (e.g., 0.02). It limits the L-infinity 
                 norm to ensure the noise remains invisible to humans.
     - J(...)  : Loss function (e.g., CrossEntropyLoss).
     - θ (theta): Model parameters/weights (frozen during attack).
     - y       : True label (Ground Truth).
     - ∇x      : Gradient of the loss with respect to the input x.
     - sign()  : Sign function (+1 or -1), extracting the direction of the gradient.

IMPLEMENTATION STEPS:
   A. Forward Pass: Feed the image to the model to compute the loss.
   B. Backward Pass: Compute gradients w.r.t. input pixels (requires_grad=True).
      This map tells us which pixels contribute most to the classification error.
   C. Perturbation: Add (epsilon * sign_of_gradient) to the original image.
   D. Clamping: Clip the resulting pixel values to the valid range [0, 1] 
      to ensure the output remains a valid image.
=============================================================================
"""


# The FGSM attack is a test method to generate adversarial examples
# Reference: https://arxiv.org/abs/1412.6572 

def fgsm_attack(image, epsilon, data_grad):
    """
    Generate adversarial example using FGSM
    
    Args:
        image: Input tensor
        epsilon: Perturbation magnitude
        data_grad: Gradient of loss with respect to input
    
    Returns:
        Perturbed image with pixel values in [0, 1]
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def attack_single_image(model, image, label, epsilon):
    """
    Attack a single image and return whether the attack was successful
    
    Args:
        model: Loaded model
        image: Image tensor (requires_grad=True)
        label: Ground truth label
        epsilon: Perturbation magnitude
    
    Returns:
        Dictionary with attack results
    """
    model.eval()
    
    # Initial prediction
    with torch.no_grad():
        output = model(image.detach())
        init_pred = output.max(1, keepdim=True)[1]
    
    # Skip if already misclassified
    if init_pred.item() != label.item():
        return {
            'already_misclassified': True,
            'initial_pred': init_pred.item(),
            'final_pred': None,
            'attack_successful': False,
            'epsilon': epsilon
        }
    
    # Calculate attack
    loss_fn = nn.CrossEntropyLoss()
    image_attack = image.clone().detach().requires_grad_(True)
    output = model(image_attack)
    loss = loss_fn(output, label)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = image_attack.grad.data
    
    # Perform FGSM attack
    perturbed_data = fgsm_attack(image_attack, epsilon, data_grad)
    
    # Re-classify
    with torch.no_grad():
        final_output = model(perturbed_data)
        final_pred = final_output.max(1, keepdim=True)[1]
    
    attack_successful = final_pred.item() != label.item()
    
    return {
        'already_misclassified': False,
        'initial_pred': init_pred.item(),
        'final_pred': final_pred.item(),
        'attack_successful': attack_successful,
        'epsilon': epsilon
    }


def run_comprehensive_attack():
    """
    Run FGSM attacks on all images with multiple epsilon values
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ADVERSARIAL ATTACK: ALL IMAGES + MULTIPLE EPSILON VALUES")
    print("="*80)
    
    # Configuration
    EPSILON_VALUES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]  # Different epsilon values
    NUM_IMAGES = 20
    
    # Load Data and Model
    if not os.path.exists("../code/data_X.npy"):
        print("[ERROR] Cannot find data_X.npy")
        return
    
    X = np.load("../code/data_X.npy")
    y = np.load("../code/data_Y.npy")
    
    print(f"\nDataset loaded: {X.shape[0]} images available")
    print(f"Testing first {NUM_IMAGES} images")
    print(f"Epsilon values to test: {EPSILON_VALUES}\n")
    
    model = load_model("../code/skin_model.pth")
    
    # Results storage
    all_results = []
    attack_success_count = {eps: 0 for eps in EPSILON_VALUES}
    total_valid_images = {eps: 0 for eps in EPSILON_VALUES}
    
    # Attack each image with each epsilon
    for img_idx in range(min(NUM_IMAGES, X.shape[0])):
        print(f"\n--- Image {img_idx} (Label: {y[img_idx]}) ---")
        
        img_tensor = torch.tensor(X[img_idx:img_idx+1], dtype=torch.float32, requires_grad=True)
        label_tensor = torch.tensor(y[img_idx:img_idx+1], dtype=torch.long)
        
        for epsilon in EPSILON_VALUES:
            result = attack_single_image(model, img_tensor, label_tensor, epsilon)
            all_results.append({
                'image_index': img_idx,
                'true_label': y[img_idx],
                'epsilon': epsilon,
                **result
            })
            
            if result['already_misclassified']:
                status = "SKIPPED (already misclassified)"
            elif result['attack_successful']:
                attack_success_count[epsilon] += 1
                total_valid_images[epsilon] += 1
                status = f"SUCCESS (predicted {result['final_pred']})"
            else:
                total_valid_images[epsilon] += 1
                status = f"FAILED (predicted {result['final_pred']})"
            
            print(f"  ε={epsilon:.2f}: {status}")
    
    # Print Summary
    print("\n" + "="*80)
    print("ATTACK SUMMARY")
    print("="*80)
    
    for epsilon in EPSILON_VALUES:
        if total_valid_images[epsilon] > 0:
            success_rate = (attack_success_count[epsilon] / total_valid_images[epsilon]) * 100
            print(f"ε={epsilon:.2f}: {attack_success_count[epsilon]}/{total_valid_images[epsilon]} successful attacks ({success_rate:.1f}%)")
    
    # Save results to CSV
    csv_filename = f"../results/attack_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs("../results", exist_ok=True)
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nDetailed results saved to: {csv_filename}")
    
    # Generate visualization
    generate_epsilon_comparison_plot(EPSILON_VALUES, attack_success_count, total_valid_images)


def generate_epsilon_comparison_plot(epsilon_values, success_counts, total_counts):
    """
    Generate comparison plot of attack success rates across epsilon values
    """
    success_rates = []
    for eps in epsilon_values:
        if total_counts[eps] > 0:
            rate = (success_counts[eps] / total_counts[eps]) * 100
            success_rates.append(rate)
        else:
            success_rates.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, success_rates, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Epsilon (ε) - Perturbation Magnitude', fontsize=12)
    plt.ylabel('Attack Success Rate (%)', fontsize=12)
    plt.title('FGSM Attack Success Rate vs. Epsilon Value\n(20 Images Tested)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    
    # Add value labels on points
    for eps, rate in zip(epsilon_values, success_rates):
        plt.annotate(f'{rate:.1f}%', xy=(eps, rate), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../plots/attack_epsilon_comparison.png', dpi=150)
    print("\nPlot saved as 'attack_epsilon_comparison.png'")


if __name__ == "__main__":
    run_comprehensive_attack()