# End-to-End Visual Reinforcement Learning for Chart Pattern Recognition

## Project Overview
This project is an advanced algorithmic trading system that trains an AI agent to "see" and trade chart patterns using Convolutional Neural Networks (CNNs) and Reinforcement Learning (RL). Unlike traditional bots that parse raw numerical time-series data, this agent makes decisions based on visual candlestick formations (high-contrast "Robot Vision" charts), simulating a human trader's pattern recognition capabilities.

## Current System Architecture (What Has Been Built)
The project currently implements a **5-phase pipeline**:

### Phase 1: Chart Rendering (`chart_renderer.py`)
- Fetches 1 year of hourly BTC-USD data via `yfinance`.
- Renders candlestick charts using `mplfinance` with a high-contrast black/green/red style.
- Outputs clean `84x84` RGB numpy arrays containing no axes, labels, or grid lines.
- Uses a sliding window of 50 candles per observation.

### Phase 2: Visual Autoencoder (`autoencoder.py`)
- CNN autoencoder architecture: 3 convolutional layers (encoder) -> 512-dimension latent vector -> 3 transposed conv layers (decoder).
- Trained on 2,000 generated chart images for 10 epochs using MSE loss.
- Saves the encoder weights as `encoder_weights.pth` to act as the "eyes" for the RL agent.

### Phase 3: Data Caching (`pre_render.py`)
- Pre-renders all chart images and caches them to `data_cache.npy`.
- Caches close prices to `price_cache.npy`.
- This avoids expensive rendering during RL training, maximizing training speed.

### Phase 4: RL Environment + Agent Training
- **Environment** (`visual_trading_env.py`): A custom Gymnasium environment with 3 discrete actions (Hold, Buy, Sell), 0.1% transaction commission, and a $10,000 starting balance.
- **Training** (`train_agent.py`): Uses Proximal Policy Optimization (PPO) via `Stable-Baselines3` with a frozen pre-trained encoder. Uses a 2-layer MLP policy `[128, 128]` and trains for 50,000 timesteps.

### Phase 5: Explainability (Grad-CAM)
- `explain_agent.py`: Uses Grad-CAM on the last convolutional layer to find the first BUY signal and visualize the agent's attention with heatmaps.
- `explain_best_effort.py`: Scans the entire history for the highest-confidence BUY signal and unfreezes the encoder for proper gradient flow to explain exactly which price patterns triggered a trade.

---

## Planned Improvements (What Needs to be Built Next)

The following areas have been identified as the core requirements for the next phase of development. **Our primary engineering objectives are to improve accuracy, support multiple assets, and create a truly end-to-end robust pipeline.**

### 1. Accuracy and Performance Enhancements
- **Scale Up Training:** Increase RL training to 500k-1M+ timesteps, scale autoencoder to 10k-20k images, and increase training duration (50+ epochs). 
- **Autoencoder Architecture Upgrade:** Migrate from a standard CNN Autoencoder to a **Variational Autoencoder (VAE)** or **Beta-VAE** for better latent disentanglement. Add batch normalization and dropout.
- **End-to-End Fine-Tuning:** Unfreeze the encoder during PPO training and enable discriminative learning rates (lower LR for the encoder) so the visual features are optimized directly for trading rewards instead of just reconstruction.
- **Multimodal State Integration:** Integrate auxiliary numerical features (e.g., volume, RSI, MACD, Bollinger Bands) as a dense input head seamlessly merged with the visual latent vector in the policy network.
- **Reward Shaping Optimization:** Transition from raw PnL rewards to Sharpe-ratio-based rewards, adding drawdown penalties, holding costs, and incorporating unrealized PnL feedback.
- **Action Space Expansion:** Implement position sizing (fractional allocation) or transition to a continuous action space.
- **Enhanced Resolution:** Increase image observation size from 84x84 to 128x128 or 224x224 to capture finer pattern details. Include volume bars as sub-charts.

### 2. Multi-Stock Generalization
Currently hardcoded for BTC-USD hourly.
- **Dynamic Symbol Parameterization:** Modify `pre_render.py`, `autoencoder.py`, and `train_agent.py` to accept arguments for multiple ticker symbols (e.g., `AAPL`, `NVDA`, `SPY`, `ETH-USD`).
- **Multi-Asset Agent Training:** Train a unified agent on concurrent charts from various stocks to learn universal chart geometries.
- **Cross-Asset Evaluation Framework (`benchmark.py`):** Create a script strictly dedicated to evaluating the trained agent on completely unseen stock test sets, comparing against Buy-and-Hold baselines, and outputting key financial metrics (Sharpe ratio, max drawdown, win rate).
- **Timeframe Agnosticism:** Test patterns across Daily, 4H, and 15-minute intervals.

### 3. Pipeline Orchestration (End-to-End Workflow)
- **Central Execution Protocol (`main.py`):** Build a unified CLI orchestration script utilizing `argparse`. Example usage:
  - `python main.py --symbol AAPL --mode train`
  - `python main.py --symbol AAPL --mode evaluate`
- **Configuration Management:** Implement a central `.yaml` or `.json` file defining all hyper-parameters (latent dimensions, learning rates, sliding windows).
- **Comprehensive Backtesting Structure:** Decouple training and testing via a chronological train/test split (e.g. 80/20). Incorporate metric generation plugins tracking risk/reward behaviors alongside equity curves.
- **MLOps Integration:** Setup TensorBoard or Weights & Biases (W&B) logging for monitoring training curves and comparing different experiment runs.

### 4. Code Quality & Financial Realism
- Implement strict Temporal Train/Test splitting to prevent lookahead bias.
- Add robust risk management guardrails (stop-loss constraints, take-profit conditions).
- Simulate realistic transactional mechanics like slippage and bid-ask spreads.
- Develop unit tests validating custom environment step logic and state transitions.

---

## Technical Stack Guidelines for AI Studio Assistance
When contributing code to this framework, please ensure strict adherence to the project's dependency profile:
- **Language**: Python 3.9+
- **Deep Learning**: PyTorch 2.0+
- **Reinforcement Learning**: Stable Baselines3 (PPO), Gymnasium
- **Computer Vision**: OpenCV, Matplotlib, `mplfinance`
- **Data Engineering**: Pandas, Numpy, yfinance
