import gymnasium as gym
import numpy as np
from gymnasium import spaces
from chart_renderer import ChartRenderer

class VisualTradingEnv(gym.Env):
    """
    Custom Environment that follows Gymnasium interface.
    The agent sees a Chart Image and decides: Hold (0), Buy (1), Sell (2).
    """
    def __init__(self):
        super(VisualTradingEnv, self).__init__()

       # --- LOAD CACHED DATA ---
        print("Loading cached data...")
        try:
            self.images = np.load('data_cache.npy')
            self.prices = np.load('price_cache.npy') # <--- NEW LOAD
            print(f"Loaded {len(self.images)} images and prices.")
        except FileNotFoundError:
            raise Exception("Run pre_render.py first!")

        # --- CONFIGURATION ---
        self.initial_balance = 10000.0
        self.commission = 0.001 # 0.1% trading fee
        self.renderer = ChartRenderer(symbol='BTC-USD', interval='1h', lookback_window=50)
        self.lookback = 50
        # --- DEFINE MAX STEPS (Crucial Fix) ---
        # It must be the smaller of the two lengths to avoid index errors
        self.max_steps = min(len(self.renderer.data), len(self.images))
        
        # --- ACTION SPACE ---
        # 0: Hold (Do nothing)
        # 1: Enter Long (Buy)
        # 2: Exit Long (Sell)
        self.action_space = spaces.Discrete(3)

        # --- OBSERVATION SPACE ---
        # The agent sees the 84x84 RGB image (pixel values 0-255)
        # We use channels-first (3, 84, 84) for PyTorch compatibility
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(3, 84, 84), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state
        self.balance = self.initial_balance
        self.position = 0 # 0 = No Position, 1 = Long
        self.entry_price = 0.0
        self.current_step = self.renderer.lookback + 1 # Start after lookback
        
        # Get first observation
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = self.images[self.current_step]
        return obs

    def step(self, action):
        # 1. Get current price data
        current_price = self.renderer.data.iloc[self.current_step]['Close']

        # Stop if price is 0 (which means we hit an empty part of the cache)
        if current_price == 0:
            terminated = True
            return self._get_obs(), 0, terminated, False, {}
        
        # 2. Execute Logic
        reward = 0
        terminated = False
        truncated = False
        
        # ACTION: HOLD (0)
        if action == 0:
            if self.position == 1:
                # If holding, reward is unrealized PnL change (optional, keeps agent attentive)
                # For simplicity, we only reward on CLOSE, but small step rewards help convergence.
                pass 
                
        # ACTION: BUY (1)
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                # Penalty for transaction cost
                reward -= (self.commission * current_price)
            else:
                # Illegal move: Buying while already long (punish slightly to teach valid moves)
                reward -= 0.1

        # ACTION: SELL (2)
        elif action == 2:
            if self.position == 1:
                self.position = 0
                # Calculate Profit/Loss
                pnl = current_price - self.entry_price
                # Reward is the actual profit
                reward += pnl
                # Penalty for transaction cost
                reward -= (self.commission * current_price)
            else:
                # Illegal move: Selling while nothing to sell
                reward -= 0.1

        # 3. Update Balance (Simplified for reward calculation)
        self.balance += reward

        # 4. Move to next time step
        self.current_step += 1
        
        # Check if dataset ended
        if self.current_step >= self.max_steps - 1:
            terminated = True
        
        # Get new observation
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        # Optional: Print stats
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Pos: {self.position}")