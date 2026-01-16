import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from chart_renderer import ChartRenderer  # Importing your Phase 1 code

# --- 1. Define the Neural Network ---
class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(VisualAutoencoder, self).__init__()
        
        # ENCODER: Compresses image (3 channels, 84x84) -> Latent Vector
        self.encoder = nn.Sequential(
            # Input: 3 x 84 x 84
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # -> 32 x 42 x 42
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 64 x 21 x 21
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # -> 64 x 19 x 19
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 19 * 19, latent_dim),
            nn.ReLU() 
        )
        
        # DECODER: Reconstructs image from Latent Vector (Only needed for training)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 19 * 19),
            nn.ReLU(),
            nn.Unflatten(1, (64, 19, 19)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, output_padding=0), 
            nn.Sigmoid() # Squishes output to [0, 1] for image display
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# --- 2. Data Prep Function ---
def prepare_training_data(renderer, num_samples=1000):
    """
    Generates a batch of chart images to train the autoencoder.
    """
    print(f"Generating {num_samples} chart images for training...")
    data = []
    
    # We skip the first 'lookback' candles to ensure we have enough data
    start_idx = renderer.lookback + 1
    
    for i in range(num_samples):
        # Generate image
        img = renderer.generate_observation(start_idx + i)
        
        # Normalize pixel values to [0, 1] and move channels to first dim (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # Change from (84,84,3) to (3,84,84)
        data.append(img)
        
        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{num_samples} images")
            
    return np.array(data)

# --- 3. Training Loop ---
def train_autoencoder():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # A. Get Data
    renderer = ChartRenderer(symbol='BTC-USD', interval='1h', lookback_window=50)
    # Generate 2000 images for training (Increase this for real thesis work)
    raw_data = prepare_training_data(renderer, num_samples=2000)
    
    # Convert to PyTorch Tensor
    tensor_data = torch.FloatTensor(raw_data).to(device)
    dataset = TensorDataset(tensor_data, tensor_data) # Input = Target (Autoencoder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # B. Initialize Model
    model = VisualAutoencoder(latent_dim=512).to(device)
    criterion = nn.MSELoss() # Compare pixels of Input vs Output
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # C. Train
    epochs = 10
    print("Starting Training...")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, _ in dataloader:
            # Forward pass
            latent, reconstructed = model(batch_features)
            
            # Calculate loss (Difference between Original and Reconstructed)
            loss = criterion(reconstructed, batch_features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")

    # D. Save the Encoder (The "Eyes" for the RL Agent)
    torch.save(model.encoder.state_dict(), "encoder_weights.pth")
    print("Training Complete. Encoder weights saved to 'encoder_weights.pth'")

    # --- Visual Verification (Thesis Requirement) ---
    # Show Original vs Reconstructed to prove it learned
    model.eval()
    with torch.no_grad():
        test_img = tensor_data[0].unsqueeze(0) # Take first image
        _, recon = model(test_img)
        
        # Move back to CPU for plotting
        orig = test_img.cpu().squeeze().permute(1, 2, 0).numpy()
        recon_img = recon.cpu().squeeze().permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Chart")
        plt.imshow(orig)
        plt.subplot(1, 2, 2)
        plt.title("AI Reconstructed (Memory)")
        plt.imshow(recon_img)
        plt.show()

if __name__ == "__main__":
    train_autoencoder()