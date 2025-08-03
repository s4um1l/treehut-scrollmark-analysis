# Instagram Scrollmark Analysis POC

A high-performance social media intelligence system for analyzing Instagram engagement data. Features parallelized data enrichment, advanced trend analysis, sentiment tracking, and automated reporting for Digital Media Managers.

## 🚀 Quick Start

### Prerequisites
- Python 3.9.6
- uv (fast Python package manager)
- Git

### Setup
1. **Clone and navigate to project:**
   ```bash
   git clone <your-repo-url>
   cd instagram_scrollmark_analysis_poc/backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   uv venv --python 3.9
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Prepare your data:**
   ```bash
   # Place your CSV file at:
   # backend/instagram_analysis/data/engagements.csv
   # 
   # Required columns: timestamp, media_id, comment_text, has_url, has_mention
   ```

## 📊 Core Operations

### 1. Exploratory Data Analysis (EDA)
```bash
cd backend
source .venv/bin/activate
python3 -c "
from instagram_analysis.engagement_eda import EngagementEDA
eda = EngagementEDA()
eda.run_comprehensive_eda()
"
```
**Output:** `reports/eda_report.md` with visualizations

### 2. Data Enrichment Pipeline (Parallelized)
```bash
# Full 17k+ dataset (production mode)
python3 -c "
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
pipeline = EnrichmentPipelineParallel(max_workers=8, demo_mode=False)
pipeline.run_complete_pipeline()
"

# Demo mode (1000 rows for testing)
python3 -c "
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
pipeline = EnrichmentPipelineParallel(max_workers=4, demo_mode=True)
pipeline.run_complete_pipeline()
"
```
**Output:** Enriched data in SQLite database + processing metrics

### 3. Generate Enrichment Report
```bash
python3 -c "
from instagram_analysis.enrichment_report_generator import EnrichmentReportGenerator
enrichment_gen = EnrichmentReportGenerator(max_workers=4)
enrichment_gen.generate_report()
"
```
**Output:** `reports/enrichment_report.md` with 6 visualizations

### 4. Advanced Trend Analysis
```bash
python3 -c "
from instagram_analysis.trend_analysis import TrendAnalysis
trend_analyzer = TrendAnalysis(max_workers=4)
trend_analyzer.run_comprehensive_analysis()
"
```
**Output:** `reports/trend_analysis_report.md` with business insights

### 5. Complete Pipeline (All Steps)
```bash
# Full production pipeline
python3 -c "
print('🚀 Running Complete Instagram Analysis Pipeline...')

# Step 1: Enrichment
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
pipeline = EnrichmentPipelineParallel(max_workers=8, demo_mode=False)
pipeline.run_complete_pipeline()

# Step 2: Enrichment Report
from instagram_analysis.enrichment_report_generator import EnrichmentReportGenerator
enrichment_gen = EnrichmentReportGenerator(max_workers=4)
enrichment_gen.generate_report()

# Step 3: Trend Analysis
from instagram_analysis.trend_analysis import TrendAnalysis
trend_analyzer = TrendAnalysis(max_workers=4)
trend_analyzer.run_comprehensive_analysis()

print('✅ Complete pipeline finished! Check reports/ directory.')
"
```

## 📈 Performance Benchmarks

| Component | Dataset Size | Processing Time | Rate |
|-----------|-------------|----------------|------|
| **Enrichment Pipeline** | 17,841 comments | 73 seconds | 245 comments/sec |
| **Enrichment Report** | 17,841 comments | 11 seconds | 1,579 comments/sec |
| **Trend Analysis** | 17,841 comments | 27 seconds | 667 comments/sec |
| **Total System** | 17,841 comments | ~2 minutes | **10-13x faster than original** |

## 🎯 Features

### Data Enrichment Layers
- **Language Detection**: Automatic comment language identification
- **Spam/Bot Detection**: Advanced pattern matching for fake accounts
- **Retailer Matching**: Geographic and brand mention extraction
- **Product/Scent Resolution**: SKU and fragrance identification
- **Intent Classification**: QUESTION, REQUEST, PURCHASE, PRAISE, COMPLAINT
- **Sentiment Analysis**: VADER-based emotional scoring
- **Engagement Proxies**: Interaction quality metrics

### Advanced Analytics
- **Weekly Share of Voice (SoV)**: Product/scent performance tracking
- **Week-over-Week Deltas**: Trend momentum analysis
- **Risers/Fallers Detection**: Emerging and declining topics
- **Question Hotspots**: High-engagement content identification
- **Retailer Heat Analysis**: Geographic market insights
- **Availability Tracking**: "Bring-back" and stock requests
- **Usage Questions**: Face-safe and application inquiries
- **Price Sensitivity**: Cost-related discussion detection

### Automated Reporting
- **Executive Dashboards**: Non-technical manager friendly
- **Visual Analytics**: Matplotlib/Seaborn charts embedded
- **Actionable Insights**: Prioritized business recommendations
- **Alert System**: Negative sentiment and spam detection

## 📁 Output Structure

```
instagram_scrollmark_analysis_poc/
├── backend/
│   ├── instagram_analysis/
│   │   ├── data/
│   │   │   └── engagements.csv          # Your input data
│   │   ├── enriched_data.db             # SQLite database
│   │   └── [analysis modules]
│   └── reports/
│       ├── eda_report.md                # Exploratory analysis
│       ├── enrichment_report.md         # Data enrichment results
│       ├── trend_analysis_report.md     # Business intelligence
│       ├── eda_images/                  # EDA visualizations
│       ├── enrichment_images/           # Enrichment charts
│       └── trend_images/                # Trend analysis plots
```

## 🔧 Configuration

### Worker Optimization
```python
# Adjust parallel processing based on your system
EnrichmentPipelineParallel(max_workers=8)    # High-end systems
EnrichmentPipelineParallel(max_workers=4)    # Standard systems  
EnrichmentPipelineParallel(max_workers=2)    # Lower-spec systems
```

### Demo Mode
```python
# Test with 1000 rows (fast)
pipeline = EnrichmentPipelineParallel(demo_mode=True)

# Production with full dataset
pipeline = EnrichmentPipelineParallel(demo_mode=False)
```

## 🎨 Customization

### Add New Intent Categories
Edit `instagram_analysis/enrichment_pipeline_parallel.py`:
```python
def _classify_intent_optimized(self, text):
    # Add your custom intent logic
    if 'custom_keyword' in text.lower():
        return 'CUSTOM_INTENT'
```

### Modify Retailer Patterns
Update retailer matching in enrichment pipeline:
```python
retailer_patterns = {
    'your_brand': r'\b(your|brand|patterns)\b'
}
```

## 📊 Business Intelligence

The system generates **3 comprehensive reports**:

1. **EDA Report**: Data overview and quality assessment
2. **Enrichment Report**: Detailed analysis of enriched dimensions  
3. **Trend Report**: Executive-ready business insights with:
   - Weekly performance trends
   - Competitor analysis
   - Customer intent patterns
   - Geographic insights
   - Actionable recommendations

Perfect for **Digital Media Managers** who need:
- Quick performance overviews
- Trend identification
- Customer sentiment tracking
- Content optimization insights
- Crisis detection alerts

## 🚀 Next Steps

1. **Run the complete pipeline** on your data
2. **Review generated reports** in `reports/` directory
3. **Customize analysis parameters** for your use case
4. **Schedule regular runs** for ongoing intelligence
5. **Integrate insights** into your social media strategy

---

**🏆 Built for Scale**: Handles 17k+ comments in under 2 minutes with professional visualizations and actionable business intelligence.
