import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

class ChartRenderer:
    def __init__(self, symbol='BTC-USD', interval='1h', lookback_window=50, img_size=(84, 84)):
        """
        Initializes the renderer.
        :param lookback_window: How many candles to show in one image (e.g., 50).
        :param img_size: Target resolution for the AI (width, height).
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback_window
        self.img_size = img_size
        
        # 1. Define a "Robot Vision" Style
        # High contrast: Black background, Green/Red candles, no edges
        mc = mpf.make_marketcolors(up='#00ff00', down='#ff0000', edge='inherit', wick='inherit', volume='in')
        self.style = mpf.make_mpf_style(marketcolors=mc, 
                                        gridstyle='', # No grid
                                        facecolor='black', 
                                        figcolor='black')
        
        # Load initial data
        self.data = self._fetch_data()
        
    def _fetch_data(self):
        print(f"Fetching data for {self.symbol}...")
        # Fetch enough data for training
        df = yf.download(self.symbol, period='1y', interval=self.interval, progress=False)
        
        # Flatten MultiIndex columns if they exist (yfinance v0.2.4+ returns MultiIndex by default)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df

    def generate_observation(self, current_step_index):
        """
        Takes a slice of data and returns a cleaned image as a numpy array.
        """
        # 1. Slice the data window
        if current_step_index < self.lookback:
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            
        window_df = self.data.iloc[current_step_index - self.lookback : current_step_index]

        # 2. Setup the buffer (In-memory file)
        buf = io.BytesIO()

        # 3. Plot configuration
        # strictly no axis, no labels, tighten layout to remove white borders
        fig, ax = mpf.plot(
            window_df,
            type='candle',
            style=self.style,
            volume=False, # Set True if you want volume bars
            axisoff=True, # IMPORTANT: Removes all numbers/text
            savefig=dict(fname=buf, dpi=100, bbox_inches='tight', pad_inches=0),
            returnfig=True,
            closefig=True
        )

        # 4. Convert Buffer to Image -> Numpy Array
        buf.seek(0)
        image = Image.open(buf)
        
        # Resize to target AI resolution (e.g., 84x84)
        image = image.resize(self.img_size).convert('RGB')
        
        # Convert to numpy array (Height, Width, Channels)
        observation = np.array(image)
        
        buf.close()
        return observation

# --- EXECUTION TEST ---

if __name__ == "__main__":
    # Initialize the engine
    renderer = ChartRenderer(symbol='NVDA', interval='1h', lookback_window=60)
    
    # Generate an image for a specific point in time (e.g., the 500th candle)
    print("Generating AI Vision frame...")
    ai_vision = renderer.generate_observation(500)
    
    # Verification: Check the shape and display the result
    print(f"Observation Shape: {ai_vision.shape}") # Should be (84, 84, 3)
    
    # Show what the AI sees
    plt.imshow(ai_vision)
    plt.title("What the AI Agent Sees")
    plt.axis('off')
    plt.show()
    
    print("Success: This array is ready for the CNN.")