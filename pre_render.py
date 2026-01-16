import numpy as np
import pandas as pd
from chart_renderer import ChartRenderer
from tqdm import tqdm

def cache_data():
    # 1. Setup
    print("Initializing Renderer...")
    renderer = ChartRenderer(symbol='BTC-USD', interval='1h', lookback_window=50)
    
    # Check if data actually downloaded
    if len(renderer.data) == 0:
        raise ValueError("yfinance failed to download data. Check your internet or update yfinance: pip install --upgrade yfinance")

    total_steps = len(renderer.data)
    print(f"Starting Pre-Rendering for {total_steps} candles...")

    # 2. Create containers
    # Images: (N, 3, 84, 84)
    all_images = np.zeros((total_steps, 3, 84, 84), dtype=np.uint8)
    # Prices: (N,) - We just need the Close price for rewards
    all_prices = np.zeros((total_steps,), dtype=np.float32)

    # 3. Loop and Render
    start_idx = renderer.lookback + 1
    
    for i in tqdm(range(start_idx, total_steps)):
        # A. Save Image
        img = renderer.generate_observation(i) 
        img = np.transpose(img, (2, 0, 1)) # PyTorch Format
        all_images[i] = img
        
        # B. Save Price (The Fix)
        # We capture the specific Close price at this exact index
        price = renderer.data.iloc[i]['Close']
        all_prices[i] = price

    # 4. Save BOTH to Disk
    print("Saving to disk...")
    np.save('data_cache.npy', all_images)
    np.save('price_cache.npy', all_prices)
    print(f"Done! Saved {len(all_images)} frames and prices.")

if __name__ == "__main__":
    cache_data()