import torch
import torch.nn as nn
import numpy as np
import os


class SkinCancerCNN(nn.Module):
    def __init__(self):
        super(SkinCancerCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),              
            nn.MaxPool2d(2, 2),     # 14x14
           
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 7x7
            
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            # Dense Layers
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),        
            nn.Linear(128, 7)       # 7 Classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(weights_path):
    print(f"--> Loading model from: {weights_path}")
    model = SkinCancerCNN()
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[ERROR] Doesn't find the file {weights_path}")


    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    
    # Remove Dropout for verification
    model.classifier[2] = nn.Identity()
    
    model.eval()
    return model


def load_skin_data(args=None):
    print("--> Loading verification data...")
    
    data_path = data_path = os.path.join(os.path.dirname(__file__), 'verification_data.pt')
   
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Doesn't find the file {data_path}")
            
    X_all, y_all = torch.load(data_path)
    
    real_eps = 0.002 
    
    if args is not None:
        if isinstance(args, dict):
            # If we get a dictionary, extract the 'epsilon' value
            if 'epsilon' in args:
                real_eps = args['epsilon']
            else:
                print("[WARNING] 'epsilon' key not found in args dictionary. Using default epsilon.")
        else:
            try:
                real_eps = float(args)
            except:
                pass

    print(f"--> Using Epsilon: {real_eps}")

    # Reshape epsilon for convolutional layers
    ret_eps = torch.tensor(real_eps, dtype=torch.float32).reshape(1, -1, 1, 1)

    # Set data bounds (these can be adjusted as needed)
    data_max = torch.tensor(1)
    data_min = torch.tensor(0)
    
    return X_all, y_all, data_max, data_min, ret_eps