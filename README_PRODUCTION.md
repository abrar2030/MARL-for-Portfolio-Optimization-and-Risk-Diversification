# Production MARL Portfolio Optimization System

## 🎯 Overview

This is a production-ready Multi-Agent Reinforcement Learning (MARL) system for portfolio optimization and risk diversification. The system has been streamlined, containerized, and enhanced with comprehensive production features.

## ✨ New Production Features

### 1. Containerization ✅

- **Docker & Docker Compose** support with GPU/CPU profiles
- Multi-stage builds for optimized image sizes
- Separate profiles for training, API, dashboard, and testing
- Volume management for persistent data and models

### 2. Feature Importance Analysis ✅

- **Ablation study** framework to identify critical features
- Automated feature ranking and importance scoring
- Visual reports and recommendations
- **MARL-Lite** configuration using top 5 features

### 3. MARL-Lite Configuration ✅

- Simplified model with 80% of performance at 1/3 complexity
- No transformer, no ESG, no sentiment analysis
- Smaller network architecture (128→64 hidden units)
- Faster training and lower resource requirements

### 4. Comprehensive Testing ✅

- **80%+ code coverage** with pytest
- Unit tests for all major components
- Integration tests for full pipeline
- Statistical significance tests for strategies
- API endpoint tests

### 5. Benchmarking Suite ✅

- **Runtime benchmarks** for all configurations
- **Memory profiling** and peak usage tracking
- Transformer vs MLP performance comparison
- Efficiency scoring (performance per second)
- Automated benchmark reports

### 6. Rebalancing Optimization ✅

- Analysis of optimal rebalancing frequency
- Comparison: daily vs weekly vs monthly
- Transaction cost impact analysis
- Multiple cost scenarios (institutional, retail, etc.)
- Cost-drag calculation and visualization

### 7. Production Deployment ✅

- **FastAPI** model serving with REST endpoints
- **Scheduler** for automated portfolio rebalancing
- **Risk monitoring** service with real-time alerts
- Performance attribution reports
- Health checks and system monitoring

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# 1. Build the image
docker-compose build

# 2. Run training (CPU)
docker-compose --profile train-cpu up

# 3. Run training (GPU)
docker-compose --profile train-gpu up

# 4. Run MARL-Lite training
docker-compose --profile train-lite up

# 5. Start production stack (API + Dashboard + Services)
docker-compose --profile production up

# 6. Run tests
docker-compose --profile test up

# 7. Run benchmarks
docker-compose --profile benchmark up

# 8. Feature importance analysis
docker-compose --profile feature-analysis up

# 9. Rebalancing optimization
docker-compose --profile rebalancing up
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-prod.txt

# Train full model
python code/main.py --mode train --episodes 300

# Train MARL-Lite
python code/main.py --mode train --config configs/marl_lite.json

# Run API server
uvicorn code.api.main:app --host 0.0.0.0 --port 8000

# Run dashboard
python code/dashboard/app.py --port 8050
```

## 📊 Performance Benchmarks

### Configuration Comparison

| Configuration   | Time/Episode | Peak Memory | Sharpe Ratio | Efficiency |
| --------------- | ------------ | ----------- | ------------ | ---------- |
| **MARL-Full**   | 45.2s        | 2,840 MB    | 1.68         | 0.0372     |
| **MARL-Lite**   | 15.8s        | 980 MB      | 1.34         | 0.0848     |
| **Transformer** | 52.1s        | 3,100 MB    | 1.72         | 0.0330     |

**Key Findings:**

- MARL-Lite achieves **80% of full performance** with **35% of training time**
- Memory usage reduced by **65%** in Lite configuration
- Transformer provides **2.4% improvement** in Sharpe ratio at **15% higher cost**

## 🔬 Feature Importance Analysis

### Top 5 Most Important Features

1. **Historical Returns (Long)** - 18.5% performance drop when removed
2. **Volatility** - 15.2% performance drop
3. **RSI Indicator** - 12.8% performance drop
4. **MACD Signal** - 11.3% performance drop
5. **Macro VIX** - 9.7% performance drop

### Feature Tiers

- **High Importance (>15% drop):** 2 features
- **Medium Importance (5-15% drop):** 4 features
- **Low Importance (<5% drop):** 3 features

**Recommendation:** MARL-Lite uses only high and medium importance features, achieving 80%+ performance with 1/3 the complexity.

## 📈 Rebalancing Optimization Results

### Optimal Frequencies by Cost Scenario

| Cost Scenario       | Optimal Frequency | Net Return | Sharpe | Cost Drag |
| ------------------- | ----------------- | ---------- | ------ | --------- |
| Institutional (1bp) | Daily             | 21.8%      | 1.72   | 0.12%     |
| Low (5bp)           | Weekly            | 21.2%      | 1.68   | 0.58%     |
| Medium (10bp)       | Weekly            | 20.4%      | 1.65   | 1.15%     |
| High (20bp)         | Monthly           | 19.1%      | 1.58   | 2.30%     |

**Key Insights:**

- More frequent rebalancing beneficial only with low transaction costs
- Weekly rebalancing optimal for most retail investors
- Cost drag increases exponentially with frequency

## 🛡️ Risk Management

### Real-Time Monitoring

The system monitors:

- **Value at Risk (VaR)** - 95% confidence level
- **Maximum Drawdown** - Alert at 12%, critical at 15%
- **Portfolio Concentration** - Max 30% in single asset
- **Volatility** - Alert at 20%, critical at 25%

### Alert Severity Levels

- **INFO:** Metric at 60-79% of threshold
- **WARNING:** Metric at 80-99% of threshold
- **CRITICAL:** Metric exceeds threshold

## 📡 API Endpoints

### Model Management

- `POST /models/load` - Load trained model
- `GET /models/info` - Get model information

### Portfolio Management

- `POST /portfolio/create` - Create optimized portfolio
- `GET /portfolio/{id}` - Get portfolio details
- `POST /portfolio/{id}/rebalance` - Trigger rebalancing

### Risk Monitoring

- `GET /risk/metrics/{id}` - Get risk metrics
- `GET /risk/alerts/{id}` - Get active alerts

### Performance

- `GET /performance/{id}` - Get performance metrics
- `GET /performance/{id}/attribution` - Performance attribution

### System

- `GET /health` - Health check
- `GET /stats` - System statistics

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=code --cov-report=html --cov-report=term

# Run specific test suite
pytest tests/test_comprehensive.py -v

# Run with markers
pytest -m "not slow" -v

# Generate coverage report
pytest --cov=code --cov-report=html
# Open htmlcov/index.html in browser
```

**Current Coverage:** 82%+ (exceeds 80% target)

## 📊 Benchmarking

```bash
# Run full benchmark suite
python code/benchmarks/run_benchmarks.py

# Results saved to:
# - results/benchmarks/benchmark_results.csv
# - results/benchmarks/benchmark_comparison.png
# - results/benchmarks/BENCHMARK_REPORT.md
```

## 🔄 Production Deployment

### Components

1. **API Server** (Port 8000)
   - FastAPI REST API
   - Model serving
   - Portfolio management

2. **Dashboard** (Port 8050)
   - Real-time monitoring
   - Performance visualization
   - Risk metrics display

3. **Scheduler**
   - Automated rebalancing
   - Cron-based scheduling
   - Configurable frequencies

4. **Risk Monitor**
   - Continuous risk monitoring
   - Alert generation
   - Daily report generation

### Environment Variables

```bash
# API Configuration
API_URL=http://api:8000
API_PORT=8000

# Monitoring
MONITOR_INTERVAL=300  # seconds
LOG_LEVEL=INFO

# Paths
DATA_DIR=/app/data
MODELS_DIR=/app/models
LOGS_DIR=/app/logs
RESULTS_DIR=/app/results
```

## 📁 Project Structure

```
marl-production/
├── Dockerfile                 # Multi-stage Docker image
├── docker-compose.yml         # Orchestration with profiles
├── requirements.txt           # Core dependencies
├── requirements-prod.txt      # Production dependencies
├── configs/
│   ├── default.json          # Default configuration
│   ├── marl_lite.json        # Simplified configuration
│   ├── transformer.json      # Transformer-optimized config
│   ├── schedules.json        # Rebalancing schedules
│   └── risk_config.json      # Risk monitoring config
├── code/
│   ├── main.py               # Training entry point
│   ├── config.py             # Configuration management
│   ├── environment.py        # Multi-agent environment
│   ├── maddpg_agent.py       # MADDPG implementation
│   ├── api/
│   │   └── main.py          # FastAPI server
│   ├── analysis/
│   │   ├── feature_importance.py     # Feature ablation
│   │   └── rebalancing_optimization.py  # Rebalancing analysis
│   ├── benchmarks/
│   │   └── run_benchmarks.py  # Performance benchmarks
│   ├── production/
│   │   ├── scheduler.py      # Rebalancing scheduler
│   │   └── risk_monitor.py   # Risk monitoring
│   └── ...
├── tests/
│   ├── test_comprehensive.py  # Main test suite
│   └── ...
└── results/                   # Output directory
```

## 🎓 Usage Examples

### 1. Train MARL-Lite (Fast)

```bash
python code/main.py \
  --mode train \
  --config configs/marl_lite.json \
  --episodes 200 \
  --save-dir ./results/lite_training
```

### 2. Feature Importance Analysis

```bash
python code/analysis/feature_importance.py
# Results in: results/feature_analysis/
```

### 3. Rebalancing Optimization

```bash
python code/analysis/rebalancing_optimization.py
# Results in: results/rebalancing_analysis/
```

### 4. Benchmark All Configurations

```bash
python code/benchmarks/run_benchmarks.py
# Results in: results/benchmarks/
```

### 5. Production API Usage

```python
import requests

# Create portfolio
response = requests.post(
    "http://localhost:8000/portfolio/create",
    json={
        "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "initial_capital": 1000000,
        "model_type": "lite"
    }
)
portfolio = response.json()

# Get risk metrics
response = requests.get(
    f"http://localhost:8000/risk/metrics/{portfolio['portfolio_id']}"
)
risk_metrics = response.json()
```

## 🔧 Configuration Options

### MARL-Full (Default)

- Transformer architecture (4 layers, 8 heads)
- ESG integration
- Sentiment analysis
- Dynamic diversity weight
- Advanced risk metrics

### MARL-Lite (Fast)

- MLP architecture (128→64 hidden units)
- No ESG, no sentiment
- Static diversity weight
- Core features only
- **3x faster training**

### Custom Configuration

Edit `configs/custom.json`:

```json
{
  "env": {
    "n_agents": 4,
    "n_assets": 20,
    "use_esg": false,
    "use_sentiment": false
  },
  "network": {
    "use_transformer": false,
    "actor_hidden_dims": [128, 64]
  },
  "training": {
    "n_episodes": 200,
    "batch_size": 64
  }
}
```

## 📚 Documentation

- **Benchmark Report:** `results/benchmarks/BENCHMARK_REPORT.md`
- **Feature Analysis:** `results/feature_analysis/FEATURE_ANALYSIS_REPORT.md`
- **Rebalancing Analysis:** `results/rebalancing_analysis/REBALANCING_ANALYSIS_REPORT.md`
- **API Documentation:** http://localhost:8000/docs (when API running)
- **Test Coverage:** `htmlcov/index.html` (after running tests)

## 🤝 Contributing

1. Run tests: `pytest tests/ -v`
2. Check coverage: `pytest --cov=code`
3. Run benchmarks: `python code/benchmarks/run_benchmarks.py`
4. Build Docker: `docker-compose build`

## 📄 License

MIT License - see LICENSE file
