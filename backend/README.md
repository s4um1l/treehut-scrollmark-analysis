# Instagram Scrollmark Analysis - Backend

High-performance Python 3.9 backend for social media intelligence with parallelized data processing, advanced analytics, and automated reporting.

## 🚀 Quick Start

### Prerequisites
- Python 3.9.6
- uv (fast package manager)
- 4+ CPU cores recommended for optimal performance

### Installation
```bash
# Create and activate virtual environment
uv venv --python 3.9
source .venv/bin/activate

# Install all dependencies
uv sync

# Verify installation
python -c "import pandas, numpy, matplotlib; print('✅ Dependencies ready')"
```

## 📊 Data Processing Commands

### Exploratory Data Analysis
```bash
# Basic EDA with visualizations
python3 -c "
from instagram_analysis.engagement_eda import EngagementEDA
eda = EngagementEDA()
eda.run_comprehensive_eda()
print('📊 EDA report generated: reports/eda_report.md')
"
```

### Parallelized Enrichment Pipeline
```bash
# Production mode - Full dataset (17k+ comments)
python3 -c "
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
pipeline = EnrichmentPipelineParallel(max_workers=8, demo_mode=False)
result = pipeline.run_complete_pipeline()
print(f'✅ Enriched {result[\"total_comments\"]} comments in {result[\"total_time\"]:.1f}s')
"

# Demo mode - 1000 comments for testing
python3 -c "
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
pipeline = EnrichmentPipelineParallel(max_workers=4, demo_mode=True)
result = pipeline.run_complete_pipeline()
print(f'🔬 Demo: {result[\"total_comments\"]} comments processed')
"

# Custom worker configuration
python3 -c "
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
import multiprocessing
optimal_workers = min(8, multiprocessing.cpu_count() - 1)
pipeline = EnrichmentPipelineParallel(max_workers=optimal_workers)
pipeline.run_complete_pipeline()
"
```

### Report Generation
```bash
# Generate enrichment report with visualizations
python3 -c "
from instagram_analysis.enrichment_report_generator import EnrichmentReportGenerator
enrichment_gen = EnrichmentReportGenerator(max_workers=4)
enrichment_gen.generate_report()
print('📈 Enrichment report: reports/enrichment_report.md')
"

# Generate trend analysis report
python3 -c "
from instagram_analysis.trend_analysis import TrendAnalysis
trend_analyzer = TrendAnalysis(max_workers=4)
summary = trend_analyzer.run_comprehensive_analysis()
print('🎯 Trend report: reports/trend_analysis_report.md')
print(f'📊 Key insights: {len(summary[\"actionable_insights\"])} recommendations')
"
```

### Complete Pipeline Execution
```bash
# Full production pipeline (recommended)
python3 -c "
import time
start_time = time.time()

print('🚀 Starting Complete Instagram Analysis Pipeline...')
print('=' * 60)

# Step 1: Data Enrichment
print('📊 Step 1: Data Enrichment Pipeline')
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
pipeline = EnrichmentPipelineParallel(max_workers=8, demo_mode=False)
enrich_result = pipeline.run_complete_pipeline()

# Step 2: Enrichment Report
print('📈 Step 2: Enrichment Report Generation')
from instagram_analysis.enrichment_report_generator import EnrichmentReportGenerator
enrichment_gen = EnrichmentReportGenerator(max_workers=4)
enrichment_gen.generate_report()

# Step 3: Trend Analysis
print('🎯 Step 3: Advanced Trend Analysis')
from instagram_analysis.trend_analysis import TrendAnalysis
trend_analyzer = TrendAnalysis(max_workers=4)
trend_result = trend_analyzer.run_comprehensive_analysis()

# Summary
total_time = time.time() - start_time
print('=' * 60)
print('✅ PIPELINE COMPLETE!')
print(f'📊 Processed: {enrich_result[\"total_comments\"]:,} comments')
print(f'⏱️ Total Time: {total_time:.1f} seconds')
print(f'⚡ Rate: {enrich_result[\"total_comments\"]/total_time:.0f} comments/second')
print(f'📁 Reports: reports/ directory')
print(f'🎯 Insights: {len(trend_result[\"actionable_insights\"])} business recommendations')
"
```

## 🔧 Advanced Configuration

### Performance Tuning
```bash
# System resource optimization
python3 -c "
import multiprocessing
import psutil

print(f'💻 System Info:')
print(f'   CPU Cores: {multiprocessing.cpu_count()}')
print(f'   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')

# Recommended worker counts based on system
cores = multiprocessing.cpu_count()
if cores >= 8:
    workers = 8
    print(f'🚀 High-performance config: {workers} workers')
elif cores >= 4:
    workers = 4  
    print(f'⚡ Standard config: {workers} workers')
else:
    workers = 2
    print(f'🔧 Conservative config: {workers} workers')

print(f'💡 Use: EnrichmentPipelineParallel(max_workers={workers})')
"
```

### Memory Management
```bash
# Monitor memory usage during processing
python3 -c "
import psutil
import time
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel

print('📊 Memory usage monitoring...')
process = psutil.Process()
initial_memory = process.memory_info().rss / (1024**2)
print(f'Initial memory: {initial_memory:.1f} MB')

pipeline = EnrichmentPipelineParallel(max_workers=4, demo_mode=True)
pipeline.run_complete_pipeline()

final_memory = process.memory_info().rss / (1024**2)
print(f'Final memory: {final_memory:.1f} MB')
print(f'Memory increase: {final_memory - initial_memory:.1f} MB')
"
```

### Database Operations
```bash
# Check database size and content
python3 -c "
import sqlite3
import os

db_path = 'instagram_analysis/enriched_data.db'
if os.path.exists(db_path):
    size_mb = os.path.getsize(db_path) / (1024**2)
    print(f'📁 Database size: {size_mb:.1f} MB')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM enriched_comments')
    count = cursor.fetchone()[0]
    print(f'📊 Records: {count:,} enriched comments')
    conn.close()
else:
    print('⚠️ Database not found. Run enrichment pipeline first.')
"

# Clear database (fresh start)
python3 -c "
import os
db_path = 'instagram_analysis/enriched_data.db'
if os.path.exists(db_path):
    os.remove(db_path)
    print('🗑️ Database cleared. Ready for fresh enrichment.')
else:
    print('ℹ️ Database already clean.')
"
```

## 📈 Performance Benchmarks

Run performance tests:
```bash
# Benchmark different worker configurations
python3 -c "
import time
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel

print('🏁 Performance Benchmark')
print('=' * 40)

for workers in [2, 4, 8]:
    print(f'Testing {workers} workers...')
    start_time = time.time()
    
    pipeline = EnrichmentPipelineParallel(max_workers=workers, demo_mode=True)
    result = pipeline.run_complete_pipeline()
    
    duration = time.time() - start_time
    rate = result['total_comments'] / duration
    
    print(f'  ⏱️ Time: {duration:.1f}s | Rate: {rate:.0f} comments/sec')
    print()

print('💡 Choose optimal worker count for your system')
"
```

## 📊 Analysis Modules

### 1. Enrichment Pipeline (`enrichment_pipeline_parallel.py`)
```bash
# Individual enrichment layers
python3 -c "
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
pipeline = EnrichmentPipelineParallel()

# Test individual layers
print('🔍 Available enrichment layers:')
print('  1. Data standardization & deduplication')
print('  2. Light EDA analysis')  
print('  3. Language detection')
print('  4. Spam/bot detection')
print('  5. Retailer/geographic matching')
print('  6. Product/scent resolution')
print('  7. Intent classification')
print('  8. Sentiment analysis')
print('  9. Engagement proxy calculation')
"
```

### 2. Trend Analysis (`trend_analysis.py`)
```bash
# Advanced trend computations
python3 -c "
from instagram_analysis.trend_analysis import TrendAnalysis
analyzer = TrendAnalysis()

print('📈 Advanced analytics capabilities:')
print('  • Weekly Share of Voice (SoV)')
print('  • Week-over-Week deltas')  
print('  • Risers/Fallers detection')
print('  • Question hotspots identification')
print('  • Retailer heat analysis')
print('  • Availability request tracking')
print('  • Usage question analysis')
print('  • Price sensitivity detection')
print('  • Automated alert generation')
print('  • Actionable insight extraction')
"
```

### 3. Report Generation (`enrichment_report_generator.py`)
```bash
# Visualization capabilities
python3 -c "
from instagram_analysis.enrichment_report_generator import EnrichmentReportGenerator
generator = EnrichmentReportGenerator()

print('📊 Report visualizations:')
print('  1. Language distribution analysis')
print('  2. Sentiment trends over time')
print('  3. Intent classification breakdown')
print('  4. Product/scent performance')
print('  5. Engagement quality metrics')
print('  6. Content quality assessment')
print()
print('📁 Outputs:')
print('  • Markdown report with embedded images')
print('  • Individual PNG charts')
print('  • Executive summary with recommendations')
"
```

## 🛠️ Development Commands

### Code Quality
```bash
# Format code
black . --line-length 88

# Type checking  
mypy . --ignore-missing-imports

# Lint code
flake8 . --max-line-length 88 --ignore E203,W503
```

### Testing
```bash
# Run quick functionality test
python3 -c "
try:
    from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
    from instagram_analysis.enrichment_report_generator import EnrichmentReportGenerator
    from instagram_analysis.trend_analysis import TrendAnalysis
    print('✅ All modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Test with small dataset
python3 -c "
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel
pipeline = EnrichmentPipelineParallel(max_workers=2, demo_mode=True)
result = pipeline.run_complete_pipeline()
print(f'✅ Test complete: {result[\"total_comments\"]} comments processed')
"
```

### Dependencies Management
```bash
# List installed packages
uv pip list

# Add new dependency
uv add package_name

# Add development dependency
uv add --dev package_name

# Update all dependencies
uv sync --upgrade

# Export requirements (if needed)
uv pip freeze > requirements.txt
```

## 📁 Project Structure

```
backend/
├── instagram_analysis/                 # Main analysis package
│   ├── __init__.py                    # Package initialization
│   ├── main.py                        # Application entry point
│   ├── engagement_eda.py              # Exploratory data analysis
│   ├── enrichment_pipeline_parallel.py # Multi-threaded enrichment
│   ├── enrichment_report_generator.py # Report generation
│   ├── trend_analysis.py              # Advanced analytics
│   ├── data/
│   │   └── engagements.csv            # Input data (place here)
│   └── enriched_data.db               # SQLite output database
├── reports/                           # Generated reports
│   ├── eda_report.md                  # EDA results
│   ├── enrichment_report.md           # Enrichment analysis  
│   ├── trend_analysis_report.md       # Business intelligence
│   ├── eda_images/                    # EDA visualizations
│   ├── enrichment_images/             # Enrichment charts
│   └── trend_images/                  # Trend analysis plots
├── main.py                            # Entry point script
├── pyproject.toml                     # Project configuration
├── README.md                          # This documentation
└── .venv/                            # Virtual environment
```

## 🎯 Business Intelligence Features

### Intent Classification
- **QUESTION**: Customer inquiries and product questions
- **REQUEST**: Specific product or service requests  
- **PURCHASE**: Direct purchase intent signals
- **PRAISE**: Positive feedback and testimonials
- **COMPLAINT**: Issues and negative feedback

### Sentiment Analysis
- **Compound Score**: Overall sentiment (-1 to +1)
- **Positive/Negative/Neutral**: Detailed emotional breakdown
- **Trend Tracking**: Sentiment changes over time

### Advanced Analytics
- **Share of Voice**: Product/brand mention tracking
- **Competitive Intelligence**: Retailer and competitor analysis
- **Geographic Insights**: Location-based engagement patterns
- **Crisis Detection**: Automated negative sentiment alerts

## 🚀 Production Deployment

### Optimal Configuration
```python
# High-performance production setup
EnrichmentPipelineParallel(
    max_workers=8,           # Adjust based on CPU cores
    demo_mode=False,         # Process full dataset
    chunk_size=2000         # Balance memory vs. speed
)
```

### Monitoring
```bash
# Real-time processing monitor
python3 -c "
import time
from instagram_analysis.enrichment_pipeline_parallel import EnrichmentPipelineParallel

print('🔄 Starting production monitoring...')
start_time = time.time()

pipeline = EnrichmentPipelineParallel(max_workers=8, demo_mode=False)
result = pipeline.run_complete_pipeline()

print(f'✅ Production run complete!')
print(f'📊 Comments: {result[\"total_comments\"]:,}')
print(f'⏱️ Duration: {time.time() - start_time:.1f}s')
print(f'⚡ Rate: {result[\"total_comments\"]/(time.time() - start_time):.0f}/sec')
"
```

---

**🏆 Engineered for Performance**: Process 17k+ comments in under 2 minutes with enterprise-grade analytics and automated business intelligence. 