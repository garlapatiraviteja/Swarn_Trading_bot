# 🧠 Swarm Brain: An LLM-Powered, Evolving Trading Intelligence

A multi-agent, self-optimizing trading system combining swarm intelligence, strategy evolution, and large language model reasoning.

## 🔍 Features

- **Swarm Intelligence Core**  
  Agents explore and compete using diverse strategies (RSI, MACD, Hybrid, LLM).

- **LLM Integration**  
  DeepSeek API + local fallback (Phi or equivalent) for real-time market analysis.

- **Dual Operation Modes**
  - **Phase 1:** Live simulation with agent voting and paper trading  
  - **Phase 2:** Historical backtesting + swarm evolution

- **Evolutionary Learning**  
  Agents are scored, ranked, cloned, and mutated across generations to maximize profit.

- **Hybrid Decision Making**  
  Combines technical indicators with LLM insights + swarm consensus.

- **Visualization Tools**  
  Strategy performance, agent fitness trends, decision distributions.

## 🛠 Tech Stack

- Python / NumPy / Pandas / Matplotlib  
- DeepSeek LLM API (with local fallback option)  
- Custom Evolution Engine  
- Modular Strategy Layer (plug-and-play)

## 🚧 Roadmap

- [x] Swarm core + agent manager  
- [x] LLM-driven strategy layer  
- [x] Paper trading simulator  
- [x] Evolutionary learning framework  
- [ ] Real-time deployment hooks  
- [ ] Frontend dashboard (optional)

## 📎 Example Flow

1. Pull live or historical data  
2. Analyze market trends using LLMs  
3. Spawn diverse agents to simulate trades  
4. Vote + evaluate strategies  
5. Evolve swarm for next round

<p align="center">
  <img src="full_trading_system_flowchart.png" alt="Swarm Brain Flowchart" width="600"/>
</p>


## 📈 Sample_Result

After each simulation run, the following HTML reports are generated:

| File | Description |
|------|-------------|
| `AAPL_agent_performance.html` | Tracks agent fitness and profitability |
| `AAPL_candlestick.html` | Shows the candlestick chart used during simulation |
| `AAPL_decision_distribution.html` | Visualizes agent votes: BUY / SELL / HOLD |
| `AAPL_strategy_performance.html` | Compares all strategies over the run |

> These files are stored in the results folder and viewable in any browser.
>
<p align="center">
  <img src="AAPL_agent_performance.png" alt="Swarm Brain Flowchart" width="600"/>
</p>
<p align="center">
  <img src="AAPL_candlestick.png" alt="Swarm Brain Flowchart" width="600"/>
</p>
<p align="center">
  <img src="AAPL_decision_distribution.png" alt="Swarm Brain Flowchart" width="600"/>
</p>
<p align="center">
  <img src="AAPL_strategy_performance.png" alt="Swarm Brain Flowchart" width="600"/>
</p>

A truly adaptive, self-improving market analyzer—where every trade trains the brain.

---

Want to contribute? Fork, test, and build a smarter swarm 🐜
