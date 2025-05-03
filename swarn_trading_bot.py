import yfinance as yf
import pandas as pd
import numpy as np
import random
import time
import requests
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from llama_cpp import Llama
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from config import DEEPSEEK_API_KEY, MODEL_PATH, FALLBACK_MODEL_PATH  # configure your own models and deepseek apis


# --- CONFIG ---
USE_LOCAL_LLM = False  # Set to True for local LLM , False for DeepSeek API
STOCKS = ["AAPL", "MSFT", "TSLA"]
TRADING_DAYS = 180

INITIAL_BALANCE = 10000
EVOLUTION_ROUNDS = 3
AGENTS_COUNT = 5
SELECTION_RATE = 0.6  # Keep top 60% performers for next generation
OUTPUT_DIR = "results"  # Directory to save results and visualizations

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LLM Setup ---
# Use a simpler approach to initialize the LLM
MODEL_PATH = MODEL_PATH
FALLBACK_MODEL_PATH = FALLBACK_MODEL_PATH

# Set this to False initially to avoid local LLM issues
USE_LOCAL_LLM = False  
llm = None

# Only attempt to load the LLM if explicitly requested
if USE_LOCAL_LLM:
    try:
        from llama_cpp import Llama
        print("Attempting to initialize local LLM...")
        
        # Try the main model path first
        if os.path.exists(MODEL_PATH):
            try:
                # Suppress verbose output
                import sys
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                
                llm = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=1024,         # Reduced context window
                    n_threads=2,        # Reduced threads
                    n_gpu_layers=0      # CPU only mode
                )
                
                sys.stdout = original_stdout
                print("Successfully loaded main LLM model")
            except Exception as e:
                sys.stdout = original_stdout
                print(f"Failed to load main model: {e}")
                llm = None
        
        # If main model failed, try fallback
        if llm is None and os.path.exists(FALLBACK_MODEL_PATH):
            try:
                sys.stdout = open(os.devnull, 'w')
                
                llm = Llama(
                    model_path=FALLBACK_MODEL_PATH,
                    n_ctx=1024,
                    n_threads=2,
                    n_gpu_layers=0
                )
                
                sys.stdout = original_stdout
                print("Successfully loaded fallback LLM model")
            except Exception as e:
                sys.stdout = original_stdout
                print(f"Failed to load fallback model: {e}")
                llm = None
                
        # If both models failed, disable local LLM
        if llm is None:
            print("Could not initialize any local LLM model. Disabling local LLM.")
            USE_LOCAL_LLM = False
            
    except ImportError:
        print("llama_cpp module not found. Please install it with: pip install llama-cpp-python")
        USE_LOCAL_LLM = False
    except Exception as e:
        print(f"Unexpected error initializing LLM: {e}")
        USE_LOCAL_LLM = False

DEEPSEEK_API_KEY = DEEPSEEK_API_KEY
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def predict_trend(market_summary, agent_memory=None):
    # Enhanced with agent memory if available
    memory_context = ""
    if agent_memory and len(agent_memory) > 0:
        memory_context = "\n\nRecent trading history:\n" + "\n".join(
            [f"Action: {m['action']}, Outcome: {'Profit' if m['reward'] > 0 else 'Loss'}" 
             for m in agent_memory[-5:]]  # Use last 5 memories
        )
    
    # Default to API method if local LLM is having issues
    if not USE_LOCAL_LLM or llm is None:
        try:
            return predict_trend_api(market_summary + memory_context)
        except Exception as e:
            print(f"DeepSeek API Error: {e}. Falling back to random decision.")
            return random.choice(["BUY", "SELL", "HOLD"])
    else:
        try:
            return predict_trend_local(market_summary + memory_context)
        except Exception as e:
            print(f"Local LLM Error: {e}. Falling back to DeepSeek API.")
            try:
                return predict_trend_api(market_summary + memory_context)
            except Exception as e2:
                print(f"DeepSeek API Error: {e2}. Falling back to random decision.")
                return random.choice(["BUY", "SELL", "HOLD"])

def predict_trend_local(market_summary):
    prompt = f"""
    You are a stock trading AI agent.
    Given the following stock market summary:

    {market_summary}

    Should you BUY, SELL, or HOLD today? Answer with only one word: 'BUY', 'SELL', or 'HOLD'.
    """
    
    try:
        # Redirect stdout to suppress LLM output
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        response = llm.create_completion(
            prompt=prompt.strip(),
            max_tokens=10,  # Reduced for simpler responses
            temperature=0.1,  # Lower temperature for more deterministic responses
            stop=["BUY", "SELL", "HOLD", "\n"],
            echo=False
        )
        
        # Restore stdout
        sys.stdout = original_stdout
        
        result = response["choices"][0]["text"].strip().upper()
        print(f"LLM raw response: '{result}'")
        
        # Add fallback logic if result is empty
        if not result:
            print("Empty LLM response, using random fallback")
            return random.choice(["BUY", "SELL", "HOLD"])
        
        # Check if result contains one of our expected values
        if "BUY" in result:
            return "BUY"
        elif "SELL" in result:
            return "SELL"
        else:
            return "HOLD"
    except Exception as e:
        sys.stdout = original_stdout if 'original_stdout' in locals() else sys.stdout
        print(f"Local LLM error: {str(e)}, using fallback")
        return random.choice(["BUY", "SELL", "HOLD"])

def predict_trend_api(market_summary):
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a trading assistant. Respond with only BUY, SELL, or HOLD."},
            {"role": "user", "content": f"Based on this market data, should I BUY, SELL, or HOLD? {market_summary}"}
        ],
        "max_tokens": 10
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    
    try:
        response = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=10)
        response_json = response.json()
        
        # Try to parse the response in various formats
        if "choices" in response_json and len(response_json["choices"]) > 0:
            if "message" in response_json["choices"][0]:
                content = response_json["choices"][0]["message"]["content"].strip().upper()
            elif "text" in response_json["choices"][0]:
                content = response_json["choices"][0]["text"].strip().upper()
            else:
                content = str(response_json["choices"][0]).upper()
                
            # Extract BUY, SELL, or HOLD from the response
            if "BUY" in content:
                return "BUY"
            elif "SELL" in content:
                return "SELL"
            else:
                return "HOLD"
        else:
            print(f"Unexpected API response: {response_json}")
            return random.choice(["BUY", "SELL", "HOLD"])
    except Exception as e:
        raise Exception(f"API request error: {str(e)}")

# --- Trading Agent ---
class TradingAgent:
    def __init__(self, name, strategy="RSI", params=None):
        self.name = name
        self.strategy = strategy
        self.params = params or {}
        self.balance = INITIAL_BALANCE
        self.position = 0
        self.history = []
        self.memory = deque(maxlen=20)  # Store last 20 trading decisions and outcomes
        self.fitness = 0  # Evolutionary fitness score
        self.daily_values = []  # Track portfolio value each day
    
    def clone(self, new_name):
        """Create a copy of this agent with potential mutations"""
        # Copy basic attributes
        new_agent = TradingAgent(new_name, self.strategy, self.params.copy())
        
        # Potentially mutate the strategy (20% chance)
        if random.random() < 0.2:
            new_agent.strategy = random.choice(["RSI", "LLM", "MACD", "HYBRID"])
        
        # Potentially mutate parameters
        if self.strategy == "RSI" and random.random() < 0.3:
            # Mutate RSI window
            new_agent.params['window'] = max(5, min(30, 
                                             self.params.get('window', 14) + 
                                             random.randint(-3, 3)))
            
        if self.strategy == "MACD" and random.random() < 0.3:
            # Mutate MACD parameters
            new_agent.params['fast'] = max(5, min(20, 
                                          self.params.get('fast', 12) + 
                                          random.randint(-2, 2)))
            new_agent.params['slow'] = max(10, min(40, 
                                          self.params.get('slow', 26) + 
                                          random.randint(-3, 3)))
        
        return new_agent

    # Fix potential bug in agent decision-making (around line 340-390)
# Update the decide method in TradingAgent class to better handle LLM strategy failures:

    def decide(self, market_summary, market_data):
        try:
            if self.strategy == "RSI":
                window = self.params.get('window', 14)  # Default to 14 if not specified
                rsi_value = self.calculate_rsi(market_data, window)
                
                # Check if rsi_value is a number and not NaN
                if isinstance(rsi_value, (int, float)) and not pd.isna(rsi_value):
                    overbought = self.params.get('overbought', 70)
                    oversold = self.params.get('oversold', 30)
                    action = "BUY" if rsi_value < oversold else "SELL" if rsi_value > overbought else "HOLD"
                    print(f"[{self.name}] Strategy: RSI({window}) | RSI: {rsi_value:.2f} → Action: {action}")
                    return action
                else:
                    # Fallback if RSI calculation failed
                    action = random.choice(["BUY", "SELL", "HOLD"])
                    print(f"[{self.name}] Strategy: RSI failed, using RANDOM → Action: {action}")
                    return action
                    
            elif self.strategy == "MACD":
                fast = self.params.get('fast', 12)
                slow = self.params.get('slow', 26)
                signal = self.params.get('signal', 9)
                
                macd, signal_line = self.calculate_macd(market_data, fast, slow, signal)
                
                if macd > signal_line:
                    action = "BUY"
                elif macd < signal_line:
                    action = "SELL"
                else:
                    action = "HOLD"
                    
                print(f"[{self.name}] Strategy: MACD({fast},{slow},{signal}) | MACD: {macd:.2f}, Signal: {signal_line:.2f} → Action: {action}")
                return action
                
            elif self.strategy == "LLM":
                # More reliable fallback for LLM strategy
                try:
                    # First attempt with LLM
                    action = predict_trend(market_summary, list(self.memory))
                    print(f"[{self.name}] Strategy: LLM → Action: {action}")
                    return action
                except Exception as e:
                    print(f"[{self.name}] LLM strategy failed: {e}")
                    # If fails, use RSI as fallback
                    try:
                        rsi_value = self.calculate_rsi(market_data)
                        action = "BUY" if rsi_value < 30 else "SELL" if rsi_value > 70 else "HOLD"
                        print(f"[{self.name}] Strategy: LLM failed, using RSI fallback → Action: {action}")
                        return action
                    except:
                        # Last resort: random
                        action = random.choice(["BUY", "SELL", "HOLD"])
                        print(f"[{self.name}] Strategy: LLM and RSI failed, using RANDOM → Action: {action}")
                        return action
                
            elif self.strategy == "HYBRID":
                # More reliable HYBRID strategy
                try:
                    rsi_value = self.calculate_rsi(market_data)
                    
                    # RSI suggests a strong action
                    if rsi_value < 25:  # Very oversold
                        action = "BUY"
                    elif rsi_value > 75:  # Very overbought
                        action = "SELL"
                    else:
                        # Try LLM for more nuanced decisions
                        try:
                            action = predict_trend(market_summary, list(self.memory))
                        except:
                            # If LLM fails, use MACD as fallback
                            macd, signal_line = self.calculate_macd(market_data)
                            action = "BUY" if macd > signal_line else "SELL" if macd < signal_line else "HOLD"
                    
                    print(f"[{self.name}] Strategy: HYBRID | RSI: {rsi_value:.2f} → Action: {action}")
                    return action
                except Exception as e:
                    print(f"[{self.name}] HYBRID strategy failed: {e}")
                    action = random.choice(["BUY", "SELL", "HOLD"])
                    print(f"[{self.name}] Strategy: HYBRID failed, using RANDOM → Action: {action}")
                    return action
                
            else:
                action = random.choice(["BUY", "SELL", "HOLD"])
                print(f"[{self.name}] Strategy: RANDOM → Action: {action}")
                return action
        except Exception as e:
            print(f"Error in decision making: {e}. Using RANDOM strategy.")
            action = random.choice(["BUY", "SELL", "HOLD"])
            print(f"[{self.name}] Strategy: ERROR FALLBACK → Action: {action}")
            return action
        

    def calculate_rsi(self, data, window=14):
        try:
            # Make a copy to avoid modifying original data
            delta = data['Close'].diff().copy()
            
            # Handle potential NaN values
            delta = delta.fillna(0)
            
            # Separate gains and losses
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = -losses  # Convert to positive values
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=window, min_periods=1).mean()
            avg_loss = losses.rolling(window=window, min_periods=1).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get the last valid RSI value
            last_valid_rsi = rsi.iloc[-1]
            
            # Ensure it's a single number, not a Series
            if isinstance(last_valid_rsi, pd.Series):
                last_valid_rsi = last_valid_rsi.iloc[0]
                
            # Handle potential NaN or infinity
            if pd.isna(last_valid_rsi) or np.isinf(last_valid_rsi):
                return 50.0  # Default to neutral RSI if calculation fails
                
            return float(last_valid_rsi)
        except Exception as e:
            print(f"RSI calculation error: {e}")
            return 50.0  # Default to neutral RSI

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        try:
            # Calculate MACD
            exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            
            # Get latest values
            latest_macd = macd.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            
            # Handle Series object if needed
            if isinstance(latest_macd, pd.Series):
                latest_macd = latest_macd.iloc[0]
            if isinstance(latest_signal, pd.Series):
                latest_signal = latest_signal.iloc[0]
                
            return float(latest_macd), float(latest_signal)
        except Exception as e:
            print(f"MACD calculation error: {e}")
            return 0.0, 0.0  # Default neutral values
    
    def act(self, action, price, date_str):
        try:
            previous_value = self.portfolio_value(price)
            price_value = float(price)
            
            if action == "BUY" and self.balance >= price_value:
                self.position += 1
                self.balance -= price_value
                self.history.append(f"BUY at ${price_value:.2f}")
            elif action == "SELL" and self.position > 0:
                self.position -= 1
                self.balance += price_value
                self.history.append(f"SELL at ${price_value:.2f}")
            else:
                self.history.append(f"HOLD at ${price_value:.2f}")
            
            # Calculate reward for this action
            new_value = self.portfolio_value(price)
            reward = new_value - previous_value
            
            # Update memory with this action and its outcome
            self.memory.append({"action": action, "reward": reward})
            
            # Track daily portfolio value
            self.daily_values.append({
                "date": date_str,
                "value": new_value,
                "action": action
            })
            
            return reward
            
        except Exception as e:
            print(f"Error in act method: {e}")
            self.history.append(f"ERROR: {str(e)}")
            return 0  # No reward for errors

    # In the portfolio_value method
    def portfolio_value(self, current_price):
        try:
            price = current_price
            if isinstance(price, pd.Series):
                # Fix the FutureWarning by using iloc[0] inside float()
                price = float(price.iloc[0])
            else:
                price = float(price)
            return self.balance + self.position * price
        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return self.balance  # Return just balance in case of error

    # --- Data Fetching ---


    def fetch_stock_data(ticker, days=30):
        """Fetch stock data with improved error handling and debugging"""
        try:
            print(f"Fetching data for {ticker} for the past {days} days...")
            
            # Try to fetch data with yfinance
            try:
                data = yf.download(ticker, period=f"{days}d", auto_adjust=False, progress=False)
                
                if data is None:
                    print(f"Warning: yfinance returned None for {ticker}")
                    raise ValueError("Empty dataset returned")
                    
                if data.empty:
                    print(f"Warning: yfinance returned empty DataFrame for {ticker}")
                    raise ValueError("Empty dataset returned")
                    
                # Check for required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    print(f"Warning: Missing columns in yfinance data: {missing_cols}")
                    raise ValueError(f"Missing columns: {missing_cols}")
                    
                # Check for NaN values
                if data.isnull().values.any():
                    print(f"Warning: yfinance data contains NaN values, filling them")
                    # Fill NaN values with forward fill, then backward fill
                    data = data.fillna(method='ffill').fillna(method='bfill')
                    
                # Success
                print(f"Successfully fetched {len(data)} rows of data for {ticker}")
                return data
                    
            except Exception as e:
                print(f"yfinance error: {e}, generating dummy data instead")
                # Fall through to dummy data creation
                pass
                
            # Create dummy data as fallback
            print(f"Generating synthetic data for {ticker}")
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Generate more realistic dummy data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date)
            
            # Base price depends on ticker to make it somewhat realistic
            if ticker == "AAPL":
                base_price = 170
            elif ticker == "MSFT":
                base_price = 330
            elif ticker == "TSLA":
                base_price = 200
            else:
                base_price = 100
                
            # Generate price series with random walk
            np.random.seed(hash(ticker) % 10000)  # Use ticker as seed for consistency
            
            # Generate random price movements
            changes = np.random.normal(0, 0.015, len(dates))  # 1.5% daily volatility
            
            # Calculate prices using cumulative returns
            factors = np.cumprod(1 + changes)
            close_prices = base_price * factors
            
            # Generate realistic OHLC data
            intraday_vol = 0.008  # 0.8% intraday volatility
            
            # Create DataFrame
            dummy_data = pd.DataFrame({
                'Open': close_prices * (1 + np.random.normal(0, intraday_vol/2, len(dates))),
                'High': close_prices * (1 + np.abs(np.random.normal(0, intraday_vol, len(dates)))),
                'Low': close_prices * (1 - np.abs(np.random.normal(0, intraday_vol, len(dates)))),
                'Close': close_prices,
                'Adj Close': close_prices,
                'Volume': np.random.uniform(1000000, 5000000, len(dates)) * factors,  # More volume on up days
            }, index=dates)
            
            # Ensure High > Open, Close, Low and Low < Open, Close, High
            for i in range(len(dummy_data)):
                row = dummy_data.iloc[i]
                max_val = max(row['Open'], row['Close'])
                min_val = min(row['Open'], row['Close'])
                
                if row['High'] < max_val:
                    dummy_data.at[dummy_data.index[i], 'High'] = max_val * 1.005  # Slightly above max
                    
                if row['Low'] > min_val:
                    dummy_data.at[dummy_data.index[i], 'Low'] = min_val * 0.995  # Slightly below min
            
            print(f"Successfully generated {len(dummy_data)} rows of synthetic data for {ticker}")
            return dummy_data
            
        except Exception as e:
            print(f"Critical error fetching stock data: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal dummy data as last resort
            dates = pd.date_range(start='2023-01-01', periods=days)
            fallback_data = pd.DataFrame({
                'Open': np.ones(days) * 100,
                'High': np.ones(days) * 101,
                'Low': np.ones(days) * 99,
                'Close': np.ones(days) * 100,
                'Adj Close': np.ones(days) * 100,
                'Volume': np.ones(days) * 1000000,
            }, index=dates)
            
            print("Returning emergency fallback data")
            return fallback_data

# --- Visualizations ---
def plot_candlestick(stock_data, ticker, output_path=None):
    """Generate a candlestick chart with improved error handling"""
    try:
        # Flatten MultiIndex columns if needed (like ('Open', 'AAPL'))
        if isinstance(stock_data.columns, pd.MultiIndex):
            print("🧠 Flattening MultiIndex columns...")
            stock_data.columns = stock_data.columns.droplevel(1)

        # More verbose debug info
        print(f"Generating candlestick chart for {ticker} with {len(stock_data)} data points")
        print(f"Data sample: {stock_data.head().to_dict()}")

        # Check if data is valid
        if stock_data.empty:
            print(f"Error: Empty data for {ticker}")
            return None

        # Ensure the data has the required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return None

        # Create a simpler version of the chart to avoid potential issues
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=1)

        # Add candlestick trace with explicit error handling
        try:
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=ticker
            ))
        except Exception as e:
            print(f"Error adding candlestick trace: {e}")
            # Try a fallback approach with a regular line chart
            try:
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name=f"{ticker} Close"
                ))
                print("Fallback to line chart succeeded")
            except Exception as e2:
                print(f"Fallback chart also failed: {e2}")
                return None

        # Simplify layout
        fig.update_layout(
            title=f'{ticker} Stock Price',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            height=600,
            width=800,
            showlegend=False
        )

        # Save with improved error handling
        if output_path:
            try:
                import os
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

                print(f"Attempting to save chart to: {os.path.abspath(output_path)}")

                fig.write_html(
                    output_path,
                    include_plotlyjs='cdn',
                    full_html=True,
                    auto_open=False
                )

                print(f"Successfully saved chart to {output_path}")
                return fig
            except Exception as e:
                print(f"Error saving chart: {type(e).__name__}: {e}")
                print(f"Will continue without saving chart")
                return fig

        return fig

    except Exception as e:
        print(f"Critical error in plot_candlestick: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None




def plot_strategy_performance(agents, current_price, output_path=None):
    """Generate a bar chart showing performance by strategy with improved error handling"""
    try:
        print(f"Starting plot_strategy_performance for {len(agents)} agents")
        strategies = {}
        
        for agent in agents:
            try:
                value = agent.portfolio_value(current_price)
                profit = value - INITIAL_BALANCE
                profit_pct = (profit / INITIAL_BALANCE) * 100
                
                if agent.strategy not in strategies:
                    strategies[agent.strategy] = []
                
                strategies[agent.strategy].append(profit_pct)
            except Exception as e:
                print(f"Error calculating profit for {agent.name}: {e}")
                continue
        
        if not strategies:
            print("No valid strategy data found")
            return None
        
        # Calculate average profit by strategy
        avg_profit = {}
        for strategy, profits in strategies.items():
            if profits:  # Only if we have valid profit data
                avg_profit[strategy] = sum(profits) / len(profits)
        
        if not avg_profit:
            print("No valid average profit data calculated")
            return None
        
        # Create bar chart
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(avg_profit.keys()),
                y=list(avg_profit.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(avg_profit)],
                text=[f"{p:.2f}%" for p in avg_profit.values()],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Average Performance by Strategy',
            xaxis_title='Strategy',
            yaxis_title='Profit (%)',
            template='plotly_white'
        )
        
        if output_path:
            try:
                # Create directory if it doesn't exist
                import os
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                fig.write_html(
                    output_path,
                    include_plotlyjs='cdn',
                    full_html=True
                )
                print(f"Successfully saved strategy performance chart to {output_path}")
            except Exception as e:
                print(f"Error saving strategy performance chart: {e}")
        
        return fig
    except Exception as e:
        print(f"Critical error in plot_strategy_performance: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_agent_performance(agents, stock, output_path=None):
    """Generate a line chart showing agent performance over time with improved error handling"""
    try:
        print(f"Starting plot_agent_performance for {stock} with {len(agents)} agents")
        
        # Check if agents have daily values
        valid_agents = [agent for agent in agents if agent.daily_values]
        
        if not valid_agents:
            print("No agents with valid daily values found")
            return None
            
        import plotly.graph_objects as go
        fig = go.Figure()
        
        for agent in valid_agents:
            try:
                if not agent.daily_values:
                    continue
                    
                dates = [d['date'] for d in agent.daily_values]
                values = [d['value'] for d in agent.daily_values]
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines',
                    name=f"{agent.name} ({agent.strategy})"
                ))
            except Exception as e:
                print(f"Error adding trace for {agent.name}: {e}")
                continue
        
        # Check if any traces were added
        if not fig.data:
            print("No valid performance data could be plotted")
            return None
        
        fig.update_layout(
            title=f'Agent Performance Trading {stock}',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add horizontal line for initial balance
        fig.add_shape(
            type="line",
            x0=0,
            y0=INITIAL_BALANCE,
            x1=1,
            y1=INITIAL_BALANCE,
            xref="paper",
            line=dict(color="gray", width=2, dash="dash"),
        )
        
        fig.add_annotation(
            x=0.01,
            y=INITIAL_BALANCE,
            xref="paper",
            text=f"Initial: ${INITIAL_BALANCE}",
            showarrow=False,
            yshift=10
        )
        
        if output_path:
            try:
                # Create directory if it doesn't exist
                import os
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                fig.write_html(
                    output_path,
                    include_plotlyjs='cdn',
                    full_html=True
                )
                print(f"Successfully saved agent performance chart to {output_path}")
            except Exception as e:
                print(f"Error saving agent performance chart: {e}")
        
        return fig
    except Exception as e:
        print(f"Critical error in plot_agent_performance: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_decision_distribution(agents, output_path=None):
    """Generate a pie chart showing distribution of trading decisions with improved error handling"""
    try:
        print(f"Starting plot_decision_distribution for {len(agents)} agents")
        
        # Check if agents have memory
        valid_agents = [agent for agent in agents if agent.memory]
        
        if not valid_agents:
            print("No agents with valid memory found")
            return None
            
        all_decisions = []
        
        for agent in valid_agents:
            for memory in agent.memory:
                all_decisions.append(memory['action'])
        
        if not all_decisions:
            print("No trading decisions found in agent memories")
            return None
            
        decision_counts = {
            "BUY": all_decisions.count("BUY"),
            "SELL": all_decisions.count("SELL"),
            "HOLD": all_decisions.count("HOLD")
        }
        
        # Remove any empty categories
        decision_counts = {k: v for k, v in decision_counts.items() if v > 0}
        
        if not decision_counts:
            print("No valid decision counts calculated")
            return None
            
        colors = {'BUY': '#2ca02c', 'SELL': '#d62728', 'HOLD': '#1f77b4'}
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Pie(
            labels=list(decision_counts.keys()),
            values=list(decision_counts.values()),
            marker_colors=[colors[key] for key in decision_counts.keys() if key in colors]
        )])
        
        fig.update_layout(
            title='Distribution of Trading Decisions',
            template='plotly_white'
        )
        
        if output_path:
            try:
                # Create directory if it doesn't exist
                import os
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                fig.write_html(
                    output_path,
                    include_plotlyjs='cdn',
                    full_html=True
                )
                print(f"Successfully saved decision distribution chart to {output_path}")
            except Exception as e:
                print(f"Error saving decision distribution chart: {e}")
        
        return fig
    except Exception as e:
        print(f"Critical error in plot_decision_distribution: {e}")
        import traceback
        traceback.print_exc()
        return None




# --- Swarm Logic ---



def run_swarm(stock, agents, days=30):
    print(f"\n=== Swarm Simulation: {stock} ===")
    
    try:
        print(f"Fetching stock data for {stock} for the past {days} days...")
        hist = TradingAgent.fetch_stock_data(stock, days)
        
        if hist is None or hist.empty:
            print(f"Error: No valid data retrieved for {stock}")
            return []
            
        print(f"Successfully retrieved {len(hist)} data points for {stock}")
        print(f"Data sample: {hist.head()}")
        
        # Save candlestick visualization with improved error handling
        try:
            print(f"Creating output directory if needed...")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            candlestick_path = os.path.join(OUTPUT_DIR, f"{stock}_candlestick.html")
            print(f"Generating candlestick chart at: {os.path.abspath(candlestick_path)}")
            
            fig = plot_candlestick(hist, stock, candlestick_path)
            if fig:
                print(f"Successfully created candlestick chart for {stock}")
            else:
                print(f"Failed to create candlestick chart for {stock}")
        except Exception as e:
            print(f"Error generating candlestick visualization: {e}")
            import traceback
            traceback.print_exc()
        
        # Rest of the function continues...
        # (keep your existing code from here, just ensuring we debug the chart creation)
        
        # Reset index safely
        if isinstance(hist.index, pd.DatetimeIndex):
            market_data = hist.reset_index()
        else:
            # If index is not datetime, create a simple index column
            market_data = hist.reset_index() if 'index' not in hist.columns else hist.copy()
        
        daily_performance = []  # Track daily performance
        
        for index, row in market_data.iterrows():
            try:
                # Extract price safely
                if isinstance(row['Close'], pd.Series):
                    price = float(row['Close'].iloc[0])
                else:
                    price = float(row['Close'])
                
                # Extract date safely
                if 'Date' in row:
                    if isinstance(row['Date'], pd.Series):
                        date_obj = row['Date'].iloc[0]
                    else:
                        date_obj = row['Date']
                    
                    # Convert to string safely
                    if hasattr(date_obj, 'strftime'):
                        date_str = date_obj.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date_obj)
                elif isinstance(market_data.index, pd.DatetimeIndex):
                    date_str = market_data.index[index].strftime('%Y-%m-%d')
                else:
                    date_str = f"Day {index+1}"
                
                # Create market summary with more technical indicators
                rsi_14 = agents[0].calculate_rsi(hist[:index+1]) if index > 14 else 50
                macd, signal = agents[0].calculate_macd(hist[:index+1]) if index > 26 else (0, 0)
                
                market_summary = (
                    f"Date: {date_str}, Close: {price:.2f}, "
                    f"RSI(14): {rsi_14:.2f}, MACD: {macd:.2f}, Signal: {signal:.2f}"
                )
                
                # Calculate price change for context
                if index > 0:
                    prev_price = float(market_data.iloc[index-1]['Close'])
                    price_change = ((price - prev_price) / prev_price) * 100
                    market_summary += f", Change: {price_change:.2f}%"
                
                # Get agent decisions
                actions = []
                for agent in agents:
                    try:
                        action = agent.decide(market_summary, hist[:index+1])
                        actions.append(action)
                    except Exception as e:
                        print(f"Error in agent decision: {e}")
                        actions.append(random.choice(["BUY", "SELL", "HOLD"]))
                
                # Voting with weighted influence based on past performance
                vote_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
                
                # Simple weighting: +1 for each profitable trade in memory
                agent_weights = []
                for agent in agents:
                    weight = 1.0  # Base weight
                    if len(agent.memory) > 0:
                        # Add influence for profitable trades
                        profitable_trades = sum(1 for m in agent.memory if m['reward'] > 0)
                        weight += profitable_trades / max(1, len(agent.memory))
                    agent_weights.append(weight)
                
                # Normalize weights
                total_weight = sum(agent_weights)
                if total_weight > 0:
                    agent_weights = [w / total_weight for w in agent_weights]
                else:
                    agent_weights = [1.0 / len(agents) for _ in agents]
                
                # Weighted voting
                for i, action in enumerate(actions):
                    vote_counts[action] += agent_weights[i]
                
                final_decision = max(vote_counts, key=vote_counts.get)
                print(f"\n🗳️ Final Decision: {final_decision} for {date_str} at ${price:.2f}")
                print(f"Vote distribution: BUY: {vote_counts['BUY']:.2f}, SELL: {vote_counts['SELL']:.2f}, HOLD: {vote_counts['HOLD']:.2f}")
                
                # Execution and track individual actions
                day_performance = []
                for i, agent in enumerate(agents):
                    # Each agent gets a reward based on their decision
                    individual_reward = agent.act(actions[i], price, date_str)
                    day_performance.append(individual_reward)
                    
                    # Also track what would have happened if they followed group decision
                    if actions[i] != final_decision:
                        # Calculate counterfactual performance for feedback
                        prev_value = agent.portfolio_value(price)
                        # Simulate group action without changing agent state
                        if final_decision == "BUY" and agent.balance >= price:
                            counterfactual_value = prev_value
                        elif final_decision == "SELL" and agent.position > 0:
                            counterfactual_value = prev_value
                        else:
                            counterfactual_value = prev_value
                        
                        counterfactual_reward = counterfactual_value - prev_value
                        
                        # Learn from group decision if it would have been better
                        if counterfactual_reward > individual_reward:
                            agent.memory.append({"action": final_decision, "reward": counterfactual_reward})
                    
                    time.sleep(0.01)  # Reduced sleep time
                
                # Track daily performance
                daily_performance.append({
                    "date": date_str,
                    "price": price,
                    "decision": final_decision,
                    "performance": sum(day_performance) / len(day_performance)
                })
                    
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue
        
        # Calculate final standings
        try:
            if len(hist) > 0:
                if isinstance(hist.iloc[-1]['Close'], pd.Series):
                    current_price = float(hist.iloc[-1]['Close'].iloc[0])
                else:
                    current_price = float(hist.iloc[-1]['Close'])
            else:
                current_price = 100.0  # Default fallback price
                
            # Calculate fitness scores for each agent
            for agent in agents:
                portfolio_value = agent.portfolio_value(current_price)
                profit_percentage = ((portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
                
                # Calculate fitness with multiple factors
                agent.fitness = profit_percentage
                
                # Reward consistency
                if len(agent.memory) > 0:
                    profitable_trades = sum(1 for m in agent.memory if m['reward'] > 0)
                    consistency = profitable_trades / len(agent.memory)
                    agent.fitness += consistency * 10  # Weight consistency
                
            # Sort by fitness
            scores = [(agent.name, agent.fitness, agent.portfolio_value(current_price), 
                      agent.strategy, agent.params) for agent in agents]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            print("\n🏆 FINAL STANDINGS:")
            for rank, (name, fitness, value, strategy, params) in enumerate(scores, 1):
                params_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else ""
                print(f"{rank}. {name}: Fitness = {fitness:.2f}, Value = ${value:.2f} | Strategy: {strategy} {params_str}")
            
            # Generate visualizations with improved error handling
            try:
                print("\nGenerating performance visualizations...")
                
                # Performance by strategy
                try:
                    strategy_perf_path = os.path.join(OUTPUT_DIR, f"{stock}_strategy_performance.html")
                    print(f"Creating strategy performance chart at: {os.path.abspath(strategy_perf_path)}")
                    fig = plot_strategy_performance(agents, current_price, strategy_perf_path)
                    if fig:
                        print("Successfully created strategy performance chart")
                    else:
                        print("Failed to create strategy performance chart")
                except Exception as e:
                    print(f"Error generating strategy performance chart: {e}")
                
                # Agent performance over time
                try:
                    agent_perf_path = os.path.join(OUTPUT_DIR, f"{stock}_agent_performance.html")
                    print(f"Creating agent performance chart at: {os.path.abspath(agent_perf_path)}")
                    fig = plot_agent_performance(agents, stock, agent_perf_path)
                    if fig:
                        print("Successfully created agent performance chart")
                    else:
                        print("Failed to create agent performance chart")
                except Exception as e:
                    print(f"Error generating agent performance chart: {e}")
                
                # Decision distribution
                try:
                    decision_path = os.path.join(OUTPUT_DIR, f"{stock}_decision_distribution.html")
                    print(f"Creating decision distribution chart at: {os.path.abspath(decision_path)}")
                    fig = plot_decision_distribution(agents, decision_path)
                    if fig:
                        print("Successfully created decision distribution chart")
                    else:
                        print("Failed to create decision distribution chart")
                except Exception as e:
                    print(f"Error generating decision distribution chart: {e}")
                
            except Exception as e:
                print(f"Error generating visualizations: {e}")
                import traceback
                traceback.print_exc()
            
            return scores
            
        except Exception as e:
            print(f"Error in final standings calculation: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    except Exception as e:
        print(f"Error in swarm run: {e}")
        import traceback
        traceback.print_exc()
        return []


# --- Evolution Logic ---
def evolve_swarm(stock, scores, current_generation=0):
    print(f"\n🧬 EVOLUTION (Generation {current_generation + 1})")
    
    try:
        # Extract agents and their scores
        agents_with_scores = [(name, fitness) for name, fitness, _, _, _ in scores]
        
        # Sort by fitness (higher is better)
        agents_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate how many agents to keep
        keep_count = max(1, int(len(agents_with_scores) * SELECTION_RATE))
        
        # Select top performers
        selected_agents = [agent for agent, _ in agents_with_scores[:keep_count]]
        
        # Create new generation
        new_generation = []
        
        # First, copy over the best performers directly (elitism)
        for i, name in enumerate(selected_agents):
            # Find the original agent
            for agent in all_agents:
                if agent.name == name:
                    # Keep this agent
                    new_generation.append(agent)
                    break
        
        # Fill remaining slots with mutations of top performers
        while len(new_generation) < AGENTS_COUNT:
            # Select a random agent from top performers to clone
            parent_name = random.choice(selected_agents)
            
            # Find the original agent
            for agent in all_agents:
                if agent.name == parent_name:
                    # Create a mutated version
                    child_name = f"Agent{len(new_generation)+1}_Gen{current_generation+1}"
                    child = agent.clone(child_name)
                    new_generation.append(child)
                    break
        
        print(f"New generation created with {len(new_generation)} agents")
        
        # Print strategy distribution
        strategy_counts = {}
        for agent in new_generation:
            if agent.strategy not in strategy_counts:
                strategy_counts[agent.strategy] = 0
            strategy_counts[agent.strategy] += 1
        
        print("Strategy distribution:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count}")
        
        return new_generation
    
    except Exception as e:
        print(f"Error in evolution: {e}")
        
        # Emergency fallback: create brand new agents
        print("Creating emergency fallback agents")
        return initialize_agents(AGENTS_COUNT)

def initialize_agents(count):
    """Initialize a diverse set of trading agents"""
    agents = []
    
    for i in range(count):
        name = f"Agent{i+1}"
        
        # Randomly select a strategy with probability
        strategy_probs = {
            "RSI": 0.3,
            "MACD": 0.2,
            "LLM": 0.2,
            "HYBRID": 0.3
        }
        
        strategies = list(strategy_probs.keys())
        strategy_weights = [strategy_probs[s] for s in strategies]
        strategy = random.choices(strategies, weights=strategy_weights)[0]
        
        # Set parameters based on strategy
        params = {}
        if strategy == "RSI":
            params = {
                "window": random.randint(9, 21),
                "oversold": random.randint(25, 35),
                "overbought": random.randint(65, 75)
            }
        elif strategy == "MACD":
            params = {
                "fast": random.randint(8, 15),
                "slow": random.randint(20, 30),
                "signal": random.randint(7, 12)
            }
        
        agents.append(TradingAgent(name, strategy, params))
    
    return agents

# --- Main Execution ---
def run_simulation():
    """Run the full simulation with evolution across multiple generations"""
    # Header with timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🤖 TRADING AGENT SWARM - SIMULATION STARTED AT {now} 🤖")
    print(f"Configuration: {AGENTS_COUNT} agents, {EVOLUTION_ROUNDS} evolution rounds")
    
    # Initialize results collection
    all_results = {}
    
    try:
        # Initialize first generation of agents
        global all_agents
        all_agents = initialize_agents(AGENTS_COUNT)
        
        # Print initial agent distribution
        print("\n🔍 INITIAL AGENT DISTRIBUTION:")
        strategy_counts = {}
        for agent in all_agents:
            if agent.strategy not in strategy_counts:
                strategy_counts[agent.strategy] = 0
            strategy_counts[agent.strategy] += 1
        
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count}")
        
        # Run simulation for each stock
        for stock in STOCKS:
            stock_results = []
            
            # Run multiple generations
            for gen in range(EVOLUTION_ROUNDS):
                print(f"\n📈 STOCK: {stock} - GENERATION {gen+1}/{EVOLUTION_ROUNDS}")
                
                # Run the simulation
                scores = run_swarm(stock, all_agents, TRADING_DAYS)
                
                # Store results
                if scores:
                    # Extract top performer
                    top_name, top_fitness, top_value, top_strategy, top_params = scores[0]
                    
                    gen_result = {
                        "generation": gen + 1,
                        "top_agent": top_name,
                        "top_strategy": top_strategy,
                        "top_value": top_value,
                        "top_profit_pct": ((top_value - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
                    }
                    
                    stock_results.append(gen_result)
                    print(f"Generation {gen+1} complete - Top strategy: {top_strategy} with {gen_result['top_profit_pct']:.2f}% profit")
                    
                    # Evolve for next generation (except last round)
                    if gen < EVOLUTION_ROUNDS - 1:
                        all_agents = evolve_swarm(stock, scores, gen)
                else:
                    print(f"No scores available for generation {gen+1}")
            
            # Store results for this stock
            all_results[stock] = stock_results
            
            # Generate summary report for this stock
            try:
                if stock_results:
                    print(f"\n📊 SUMMARY FOR {stock}:")
                    for gen_result in stock_results:
                        print(f"  Generation {gen_result['generation']}: "
                              f"Top strategy: {gen_result['top_strategy']} with "
                              f"{gen_result['top_profit_pct']:.2f}% profit")
                    
                    # Find best overall generation
                    best_gen = max(stock_results, key=lambda x: x['top_profit_pct'])
                    print(f"\n  Best result: Generation {best_gen['generation']} with "
                          f"{best_gen['top_profit_pct']:.2f}% profit using {best_gen['top_strategy']}")
            except Exception as e:
                print(f"Error generating summary for {stock}: {e}")
        
        # Overall summary
        print("\n🏁 SIMULATION COMPLETE")
        print(f"Total stocks: {len(STOCKS)}")
        print(f"Total generations: {EVOLUTION_ROUNDS}")
        print(f"Total agents per generation: {AGENTS_COUNT}")
        
        # Save results to file
        try:
            # Convert results to JSON-serializable format
            serializable_results = {}
            for stock, results in all_results.items():
                serializable_results[stock] = []
                for result in results:
                    serializable_results[stock].append({
                        "generation": result["generation"],
                        "top_agent": result["top_agent"],
                        "top_strategy": result["top_strategy"],
                        "top_value": float(result["top_value"]),
                        "top_profit_pct": float(result["top_profit_pct"])
                    })
            
            results_path = os.path.join(OUTPUT_DIR, "simulation_results.json")
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            print(f"Results saved to {results_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
    except Exception as e:
        print(f"Error in simulation: {e}")

if __name__ == "__main__":
    run_simulation()
