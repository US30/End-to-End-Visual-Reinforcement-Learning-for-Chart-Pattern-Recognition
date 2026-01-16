import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from stable_baselines3 import PPO
from visual_trading_env import VisualTradingEnv

# --- GRAD-CAM HELPER CLASS ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook into the target layer to catch gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are usually a tuple, we want the first element
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        # 1. Forward Pass
        # We need to manually run the parts of the PPO Policy 
        # because SB3 wraps everything tightly.
        
        # Extract features using the encoder
        features = self.model.policy.features_extractor(x)
        
        # Get the logits (raw scores) from the policy network (MLP)
        # Note: SB3 PPO Policy has an 'mlp_extractor' and 'action_net'
        latent_pi, _ = self.model.policy.mlp_extractor(features)
        logits = self.model.policy.action_net(latent_pi)
        
        # 2. Backward Pass
        # We want to explain the score for 'class_idx' (e.g., Buy = 1)
        score = logits[0, class_idx]
        
        # Zero out previous gradients
        self.model.policy.zero_grad()
        
        # Calculate gradients of the Score w.r.t. the Feature Map
        score.backward()
        
        # 3. Generate Heatmap
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Multiply each activation channel by its importance (gradient)
        activation = self.activation[0] # remove batch dim
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        # Average the channels to get a 2D Heatmap
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        
        # Apply ReLU (we only care about positive influence)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize between 0 and 1
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        
        return heatmap

# --- MAIN EXECUTION ---
def explain_decision():
    # 1. Load Environment & Agent
    env = VisualTradingEnv()
    model = PPO.load("visual_trader_ppo")
    
    print("Searching for a generic BUY signal to explain...")
    
    # 2. Hunt for a BUY signal
    # We loop through the data until the agent wants to BUY (Action 1)
    obs, _ = env.reset()
    found = False
    
    # Run for max 1000 steps to find a good example
    for step in range(1000):
        # Get action from agent
        action, _ = model.predict(obs, deterministic=True)
        
        if action == 1: # 1 = BUY
            print(f"Found a BUY signal at Step {env.current_step}!")
            found = True
            break
        
        # Step env
        obs, _, terminated, _, _ = env.step(action)
        if terminated: break

    if not found:
        print("Could not find a BUY signal in 1000 steps. Try training longer!")
        return

    # 3. Setup Grad-CAM
    # We target the LAST Convolutional Layer of the Encoder.
    # Structure: Encoder -> [Conv, ReLU, Conv, ReLU, Conv (Target), ReLU, Flatten]
    # The last Conv2d is usually at index 4 in your nn.Sequential
    target_layer = model.policy.features_extractor.autoencoder.encoder[4]
    
    grad_cam = GradCAM(model, target_layer)
    
    # Prepare input tensor
    # SB3 expects a tensor on the correct device
    obs_tensor = torch.tensor(obs).unsqueeze(0).to(model.device).float()
    
    # 4. Generate Heatmap for 'Buy' (Action 1)
    heatmap = grad_cam(obs_tensor, class_idx=1)
    
    # 5. Visualization (The "Thesis Chart")
    
    # Resize heatmap to match original image size (84x84)
    heatmap_resized = cv2.resize(heatmap, (84, 84))
    
    # Convert heatmap to RGB (Red = High Attention)
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    # Prepare Original Image
    # Obs is (3, 84, 84) -> Transpose to (84, 84, 3) for plotting
    original_img = np.transpose(obs, (1, 2, 0))
    # Convert to standard uint8 for blending
    original_img_uint8 = original_img.astype(np.uint8) 
    # Usually the chart is black/green/red. 
    # We might need to convert it to BGR for OpenCV blending
    original_bgr = cv2.cvtColor(original_img_uint8, cv2.COLOR_RGB2BGR)
    
    # Superimpose
    superimposed_img = heatmap_color * 0.4 + original_bgr * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # PLOT
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original Chart (Step {env.current_step})")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("AI Attention Map (Grad-CAM)")
    plt.imshow(heatmap_resized, cmap='jet')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Overlay (Thesis Proof)")
    plt.imshow(superimposed_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Explanation Generated. If the red spots align with the pattern, you have proof!")

if __name__ == "__main__":
    explain_decision()