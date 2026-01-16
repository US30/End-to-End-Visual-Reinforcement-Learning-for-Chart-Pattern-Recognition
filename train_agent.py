import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from visual_trading_env import VisualTradingEnv
from autoencoder import VisualAutoencoder  # Your Phase 2 Model Class

# --- 1. Define Custom Vision Module ---
class PretrainedVisionExtractor(BaseFeaturesExtractor):
    """
    This class connects your Phase 2 Autoencoder to the PPO Agent.
    """
    def __init__(self, observation_space, features_dim=512):
        super(PretrainedVisionExtractor, self).__init__(observation_space, features_dim)
        
        # Initialize your Autoencoder architecture
        self.autoencoder = VisualAutoencoder(latent_dim=features_dim)
        
        # LOAD WEIGHTS (Crucial Step)
        # We load the weights you saved in Phase 2
        try:
            # We only need the encoder part's state dict
            # Note: We saved 'model.encoder.state_dict()' in Phase 2
            self.autoencoder.encoder.load_state_dict(torch.load("encoder_weights.pth"))
            print("Successfully loaded pre-trained Encoder weights!")
        except FileNotFoundError:
            print("WARNING: 'encoder_weights.pth' not found. Agent will start with random vision.")

        # Freeze the encoder? 
        # Option A: Freeze (True) -> Faster, relies 100% on pre-training.
        # Option B: Unfreeze (False) -> Agent fine-tunes the vision.
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False  # Freezing for stability

    def forward(self, observations):
        # Pass the image through the pre-trained encoder
        # Normalize 0-255 -> 0-1
        observations = observations / 255.0
        return self.autoencoder.encoder(observations)

# --- 2. Main Training Loop ---
if __name__ == "__main__":
    # Create the environment
    env = VisualTradingEnv()
    # PPO requires a vectorized environment
    env = DummyVecEnv([lambda: env])

    # CONFIGURATION
    policy_kwargs = dict(
        features_extractor_class=PretrainedVisionExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[128, 128]  # The "Brain" after the "Eyes" (simple MLP)
    )

    # Initialize PPO Agent
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
    )

    print("Starting PPO Training...")
    # Train for 50,000 steps (For thesis, do 1M+)
    model.learn(total_timesteps=50000)
    
    # Save the trained agent
    model.save("visual_trader_ppo")
    print("Training Complete. Model saved as 'visual_trader_ppo.zip'")