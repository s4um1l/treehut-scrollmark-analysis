# Instagram Scrollmark Analysis POC

> **Social Media Intelligence Platform for Digital Media Managers**

Transform Instagram engagement data into actionable business intelligence. Processes 17,841+ comments with sentiment analysis, trend detection, and automated alerting.

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)

## 🚀 Quick Start

### Prerequisites
- Python 3.9
- `uv` package manager ([install guide](https://github.com/astral-sh/uv))

### Setup
```bash
git clone <repository-url>
cd instagram_scrollmark_analysis_poc

# Setup environment
cd backend && uv venv --python 3.9 && source .venv/bin/activate && uv install && cd ..
```

### Run Pipeline & Dashboard
```bash
# 1. Process all data (EDA → Enrichment → Trends)
./scripts/run_pipeline.sh

# 2. Start dashboard server
./scripts/run_server.sh

# 3. Open http://localhost:8000
```

## 📊 Reports & Analysis

### Executive Summary
📈 **[Executive Summary](reports/EXECUTIVE_SUMMARY.md)** - Visual dashboard for Digital Media Managers with immediate action items

### Detailed Reports
- 📊 **[EDA Report](reports/eda_report.md)** - Community engagement patterns and performance insights
- 🔬 **[Enrichment Report](reports/enrichment_report.md)** - Sentiment, intent, and content quality analysis  
- 📈 **[Trend Analysis](reports/trend_analysis_report.md)** - Advanced intelligence with question hotspots and alerts
- 🚀 **[Extension Proposal](EXTENSION_PROPOSAL.md)** - Future roadmap for enterprise scaling

### Live Dashboard
- **Interactive Dashboard**: http://localhost:8000/ (after running server)
- **API Documentation**: http://localhost:8000/docs

## 🎯 Key Features

### 🚨 Automated Intelligence
- **5 Active Alerts**: Sentiment spikes, complaint clusters, question hotspots
- **Geographic Opportunities**: 16 Canada expansion requests detected
- **Content Strategy**: Top scents (Rose: 96, Vanilla: 74, Coffee: 45 mentions)

### ⚡ Performance
- **Processing**: 17,841 comments in 3-5 minutes (parallelized)
- **Dashboard**: <2 second response times
- **Database**: 12MB SQLite with optimized indexes

## 🏗️ Project Structure

```
instagram_scrollmark_analysis_poc/
├── reports/                          # 📊 All generated reports & database
├── scripts/                          # 🚀 run_pipeline.sh, run_server.sh
├── backend/                          # 🔧 Python analysis engine & FastAPI
│   ├── instagram_analysis/          # Data processing modules
│   ├── app/main.py                  # Dashboard server
│   └── templates/dashboard.html     # Interactive UI
├── EXTENSION_PROPOSAL.md            # 📋 Future roadmap
└── README.md                        # This file
```

## 🔧 Development Commands

```bash
# Individual components
python -c "from backend.instagram_analysis.engagement_eda import EngagementEDA; EngagementEDA().generate_comprehensive_report()"
python -c "from backend.instagram_analysis.enrichment_pipeline_parallel import ParallelEnrichmentPipeline; ParallelEnrichmentPipeline().run_pipeline()"

# API testing
curl http://localhost:8000/api/health
curl http://localhost:8000/api/dashboard/summary
```

## 🎯 Business Value

**For Digital Media Managers:**
- ⚡ **Crisis Prevention**: Real-time sentiment monitoring
- 📊 **Strategic Planning**: Data-driven content calendar  
- 🌍 **Market Intelligence**: Geographic expansion opportunities
- 🎪 **Community Management**: Question hotspots with 2-hour response targets

**ROI Delivered:**
- 997 purchase intent comments → direct sales opportunities
- 1,120 praise comments → UGC content strategy
- 89.4% community health score → excellent brand perception

## 🚀 Next Steps

1. **View Results**: Check [Executive Summary](reports/EXECUTIVE_SUMMARY.md) for immediate actions
2. **Scale Up**: See [Extension Proposal](EXTENSION_PROPOSAL.md) for enterprise roadmap
3. **Deploy**: Use `./scripts/run_server.sh` for production dashboard

---

**🎊 Complete social media intelligence platform ready for Digital Media Manager success.**
