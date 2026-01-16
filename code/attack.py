import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from model_data_defs import load_model


# The FGSM attack is a test method to generate adversarial examples
# Reference: https://arxiv.org/abs/1412.6572 

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel in the direction of the error
    perturbed_image = image + epsilon * sign_data_grad
    # Ensure values are valid (between 0 and 1)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def run_attack_demo():
    print("\n=== EXPERIMENT 4: ADVERSARIAL ATTACK (VISUALIZATION) ===")
    
    # 1. Configuration
    EPSILON = 0.02  # Amount of noise (very low, almost invisible)
    IMG_INDEX = 0   # Index of the image to attack (you can change it)
    
    # 2. Load Data and Model
    if not os.path.exists("data_X.npy"):
        print("[ERROR] Cannot find data_X.npy")
        return

    X = np.load("data_X.npy")
    y = np.load("data_Y.npy")
    
    # Select a victim image
    # Convert to tensor (1, 3, 32, 32)
    # IMPORTANT: requires_grad=True is necessary to calculate the attack
    img_tensor = torch.tensor(X[IMG_INDEX:IMG_INDEX+1], dtype=torch.float32, requires_grad=True)
    label_tensor = torch.tensor(y[IMG_INDEX:IMG_INDEX+1], dtype=torch.long)
    
    model = load_model("skin_model.pth")
    # For attacks, we need to calculate gradients, so we activate simulated training mode
    # (although we use eval() for layer consistency, we need zero_grad)
    model.eval() 
    
    # 3. Initial Prediction
    output = model(img_tensor)
    init_pred = output.max(1, keepdim=True)[1]
    
    # If the network already fails on the base image, look for another one
    if init_pred.item() != label_tensor.item():
        print(f"Image {IMG_INDEX} was already misclassified. Try another index.")
        return

    print(f"Original Image: Class {label_tensor.item()} (Correct Prediction)")
    
    # 4. Calculate the Attack
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, label_tensor)
    model.zero_grad()
    loss.backward()
    
    data_grad = img_tensor.grad.data
    
    # Call FGSM
    perturbed_data = fgsm_attack(img_tensor, EPSILON, data_grad)
    
    # 5. Re-classify the attacked image
    final_output = model(perturbed_data)
    final_pred = final_output.max(1, keepdim=True)[1]
    
    print(f"Attacked Image (Epsilon={EPSILON}): Prediction {final_pred.item()}")
    
    if final_pred.item() != label_tensor.item():
        print(">>> ATTACK SUCCESSFUL! The network has been fooled.")
    else:
        print(">>> Attack failed. The model resisted (try increasing EPSILON).")

    # 6. Generate Visual Plot "Before vs After"
    perturbed_np = perturbed_data.detach().numpy().squeeze().transpose(1, 2, 0)
    original_np = X[IMG_INDEX].transpose(1, 2, 0)
    
    # Amplify noise to be visible to the human eye in the middle plot
    noise = (perturbed_np - original_np)
    # Normalize noise for visualization (0-1)
    noise_viz = (noise - noise.min()) / (noise.max() - noise.min())
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original\nClass: {label_tensor.item()}")
    plt.imshow(original_np)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Adversarial Noise\n(Amplified)")
    plt.imshow(noise_viz)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Adversarial\nPred: {final_pred.item()}")
    plt.imshow(perturbed_np)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../plots/attack_visualization.png')
    print("Image saved as 'attack_visualization.png'")

if __name__ == "__main__":
    run_attack_demo()