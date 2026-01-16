import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from stable_baselines3 import PPO
from visual_trading_env import VisualTradingEnv

# --- GRAD-CAM HELPER ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook into the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        # Capture the gradient flowing back
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        self.model.policy.zero_grad()
        
        # Forward pass components manually
        features = self.model.policy.features_extractor(x)
        latent_pi, _ = self.model.policy.mlp_extractor(features)
        logits = self.model.policy.action_net(latent_pi)
        
        # Backward pass on the specific class score
        score = logits[0, class_idx]
        score.backward()
        
        # Generate Heatmap
        # If self.gradients is still None here, it means the hook didn't fire
        if self.gradients is None:
            raise ValueError("Gradients are missing! Ensure the encoder is unfrozen.")

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activation[0]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap

# --- MAIN EXECUTION ---
def find_and_explain_best_signal():
    # 1. Load
    env = VisualTradingEnv()
    model = PPO.load("visual_trader_ppo")
    
    # --- CRITICAL FIX: UNFREEZE THE ENCODER ---
    # We must enable gradients so Grad-CAM can trace the signal back to the image
    print("Unfreezing encoder for visualization...")
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = True
    # ------------------------------------------

    print(f"Scanning {env.max_steps} steps for the strongest BUY signal...")
    
    max_buy_prob = -np.inf
    best_step_index = 0
    best_obs = None

    # 2. Search the WHOLE history
    obs, _ = env.reset()
    
    for step in range(env.max_steps - 1):
        obs_tensor = torch.tensor(obs).unsqueeze(0).to(model.device).float()
        
        with torch.no_grad():
            features = model.policy.features_extractor(obs_tensor)
            latent_pi, _ = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)
            probs = torch.softmax(logits, dim=1)
            buy_prob = probs[0, 1].item()
        
        if buy_prob > max_buy_prob:
            max_buy_prob = buy_prob
            best_step_index = env.current_step
            best_obs = obs 
            
            if buy_prob > 0.5:
                print(f"Found a REAL Buy Signal (>50%) at Step {best_step_index}")
                break
        
        obs, _, terminated, _, _ = env.step(0)
        if terminated: break
        
        if step % 1000 == 0:
            print(f"Scanned {step} steps... Highest Buy Prob so far: {max_buy_prob:.2%}")

    print(f"Scanning Complete. Best Step: {best_step_index} with Buy Probability: {max_buy_prob:.2%}")

    # 3. Generate Heatmap
    # Target Layer: The 3rd Conv2d (Index 4 in the sequential list)
    target_layer = model.policy.features_extractor.autoencoder.encoder[4]
    grad_cam = GradCAM(model, target_layer)
    
    obs_tensor = torch.tensor(best_obs).unsqueeze(0).to(model.device).float()
    
    # This call should now work because we set requires_grad=True
    heatmap = grad_cam(obs_tensor, class_idx=1)
    
    # 4. Plotting
    heatmap_resized = cv2.resize(heatmap, (84, 84))
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    original_img = np.transpose(best_obs, (1, 2, 0)).astype(np.uint8)
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    
    superimposed_img = heatmap_color * 0.4 + original_bgr * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.title(f"Chart at Step {best_step_index}\n(Buy Prob: {max_buy_prob:.1%})")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("What the AI Looked At")
    plt.imshow(heatmap_resized, cmap='jet')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Thesis Proof")
    plt.imshow(superimposed_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    find_and_explain_best_signal()