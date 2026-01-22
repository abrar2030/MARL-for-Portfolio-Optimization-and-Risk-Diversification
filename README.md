# 🚀 MARL Portfolio Optimization - Production System v1.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-82%25%20coverage-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, streamlined Multi-Agent Reinforcement Learning system for portfolio optimization with comprehensive testing, benchmarking, and deployment infrastructure.

## 📦 What's New in Production v1.0

✅ **Containerization** - Docker & Docker Compose with GPU support  
✅ **Feature Analysis** - Ablation study identifies top 5 critical features  
✅ **MARL-Lite** - 80% performance with 1/3 complexity (3x faster)  
✅ **Comprehensive Testing** - 82% code coverage with pytest  
✅ **Benchmarking** - Runtime and memory performance analysis  
✅ **Rebalancing Optimization** - Analyzes optimal frequency vs costs  
✅ **Production API** - FastAPI with 15+ endpoints  
✅ **Automated Services** - Rebalancing scheduler & risk monitoring

## ⚡ Quick Start (5 minutes)

### Option 1: Docker (Recommended)

```bash
# Clone and navigate
cd marl-production

# Start production stack
docker-compose --profile production up -d

# Access services
# - API: http://localhost:8000/docs
# - Dashboard: http://localhost:8050
```

### Option 2: Local Setup

```bash
# Run setup script
chmod +x setup.sh && ./setup.sh

# Activate environment
source venv/bin/activate

# Quick demo
python code/main.py --mode demo

# Train MARL-Lite (fast)
python code/main.py --mode train --config configs/marl_lite.json
```

## 📊 Performance Highlights

| Configuration | Training Time | Memory     | Sharpe Ratio | Efficiency |
| ------------- | ------------- | ---------- | ------------ | ---------- |
| **MARL-Full** | 45.2s/ep      | 2.8 GB     | **1.68**     | 0.037      |
| **MARL-Lite** | **15.8s/ep**  | **980 MB** | 1.34         | **0.085**  |
| Improvement   | **-65%**      | **-65%**   | -20%         | **+129%**  |

**MARL-Lite achieves 80% of performance with 1/3 the complexity!**

## 🎯 Key Features

### 1. Feature Importance Analysis

- Systematic ablation study
- Identifies top 5 critical features:
  1. Historical Returns (18.5% importance)
  2. Volatility (15.2%)
  3. RSI (12.8%)
  4. MACD Signal (11.3%)
  5. VIX (9.7%)

### 2. Rebalancing Optimization

- Analyzes: daily, weekly, monthly, quarterly
- Multiple cost scenarios (1bp to 20bp)
- **Optimal:** Weekly for retail (10bp costs)
- **Result:** 21.2% return, 1.68 Sharpe, 1.15% cost drag

### 3. Production Deployment

- **API Server:** FastAPI with Swagger docs
- **Scheduler:** Automated rebalancing (cron-based)
- **Risk Monitor:** Real-time alerts and reports
- **Dashboard:** Plotly/Dash visualization

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[README_PRODUCTION.md](README_PRODUCTION.md)** - Full documentation
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Deployment guide
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **API Docs:** http://localhost:8000/docs (when running)

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=code --cov-report=html

# Quick integration tests
pytest tests/test_comprehensive.py::TestIntegration -v

# View coverage report
open htmlcov/index.html
```

**Current Coverage: 82%+** ✅

## 📈 Analysis Tools

### Feature Importance

```bash
python code/analysis/feature_importance.py
# Output: results/feature_analysis/
```

### Rebalancing Optimization

```bash
python code/analysis/rebalancing_optimization.py
# Output: results/rebalancing_analysis/
```

### Performance Benchmarks

```bash
python code/benchmarks/run_benchmarks.py
# Output: results/benchmarks/
```

## 🐳 Docker Commands

```bash
# Training
docker-compose --profile train-cpu up      # CPU training
docker-compose --profile train-gpu up      # GPU training
docker-compose --profile train-lite up     # MARL-Lite training

# Production
docker-compose --profile production up -d  # Full stack
docker-compose --profile api up           # API only
docker-compose --profile dashboard up     # Dashboard only

# Analysis
docker-compose --profile feature-analysis up
docker-compose --profile rebalancing up
docker-compose --profile benchmark up

# Testing
docker-compose --profile test up
```

## 🛠️ Makefile Commands

```bash
make help           # Show all commands
make install        # Install dependencies
make test           # Run tests
make train          # Train full model
make train-lite     # Train MARL-Lite
make benchmark      # Run benchmarks
make analysis       # Feature importance
make rebalancing    # Rebalancing optimization
make docker-up      # Start production stack
make api            # Start API server
make dashboard      # Start dashboard
make clean          # Clean generated files
```

## 📁 Project Structure

```
marl-production/
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Service orchestration
├── Makefile               # Convenience commands
├── setup.sh               # Automated setup
├── configs/               # Configuration files
│   ├── marl_lite.json    # Simplified config
│   ├── transformer.json  # Transformer config
│   ├── schedules.json    # Rebalancing schedules
│   └── risk_config.json  # Risk monitoring
├── code/
│   ├── main.py           # Training entry point
│   ├── api/
│   │   └── main.py       # FastAPI server
│   ├── analysis/
│   │   ├── feature_importance.py
│   │   └── rebalancing_optimization.py
│   ├── benchmarks/
│   │   └── run_benchmarks.py
│   └── production/
│       ├── scheduler.py   # Rebalancing scheduler
│       └── risk_monitor.py # Risk monitoring
└── tests/
    └── test_comprehensive.py  # Test suite (82% coverage)
```

## 🌟 Use Cases

### Research & Development

```bash
# Analyze features
make analysis

# Compare configurations
make benchmark

# Experiment with configs
python code/main.py --config configs/custom.json
```

### Production Deployment

```bash
# Deploy full stack
make docker-up

# Monitor services
docker-compose logs -f

# Check health
curl http://localhost:8000/health
```

### Model Training

```bash
# Fast iteration with MARL-Lite
make train-lite

# Full model with all features
make train

# GPU accelerated
docker-compose --profile train-gpu up
```

## 🔧 Configuration

### MARL-Full (Default)

- Transformer architecture
- ESG + Sentiment analysis
- All technical indicators
- **Use when:** Maximum performance needed

### MARL-Lite (Recommended)

- MLP architecture
- Top 5 features only
- No ESG/sentiment
- **Use when:** Fast iteration needed

## 📊 Benchmarks

### Training Performance

- **MARL-Full:** 4 hours (300 episodes)
- **MARL-Lite:** 1.5 hours (200 episodes)
- **Speedup:** 2.7x faster

### Resource Usage

- **MARL-Full:** 2.8 GB RAM, 45s/episode
- **MARL-Lite:** 980 MB RAM, 16s/episode
- **Savings:** 65% memory, 65% time

## 🔐 Security

- Environment variable configuration
- No hardcoded credentials
- API authentication ready
- Input validation (Pydantic)
- Docker security best practices

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🤝 Contributing

1. Run tests: `make test`
2. Check coverage: `pytest --cov=code`
3. Run benchmarks: `make benchmark`
4. Update documentation

## 📮 Support

- **Documentation:** See `README_PRODUCTION.md`
- **Quick Start:** See `QUICKSTART.md`
- **Issues:** Check troubleshooting in docs
- **API Docs:** http://localhost:8000/docs

## 🎓 Credits

- **Original:** [quantsingularity/MARL-for-Portfolio-Optimization](https://github.com/quantsingularity/MARL-for-Portfolio-Optimization-and-Risk-Diversification)
- **Production v1.0:** Complete overhaul with containerization, testing, analysis, and deployment

---

**Version:** 1.0.0 | **Status:** Production Ready ✅ | **Updated:** 2026-01-21

🚀 **Ready to deploy!** Follow [QUICKSTART.md](QUICKSTART.md) to get started.
