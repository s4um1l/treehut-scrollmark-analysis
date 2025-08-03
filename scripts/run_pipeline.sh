#!/bin/bash

# Instagram Scrollmark Analysis Pipeline Runner
# This script runs the complete data processing pipeline

set -e  # Exit on any error

echo "🚀 Instagram Scrollmark Analysis Pipeline"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "backend/pyproject.toml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
cd backend
source .venv/bin/activate
cd ..

# Step 1: EDA Analysis
echo ""
echo "📊 Step 1: Running Exploratory Data Analysis..."
python -c "
from backend.instagram_analysis.engagement_eda import EngagementEDA
eda = EngagementEDA()
eda.generate_comprehensive_report()
print('✅ EDA Report generated: reports/eda_report.md')
"

# Step 2: Parallel Enrichment Pipeline  
echo ""
echo "⚡ Step 2: Running Parallel Data Enrichment Pipeline..."
python -c "
from backend.instagram_analysis.enrichment_pipeline_parallel import ParallelEnrichmentPipeline
pipeline = ParallelEnrichmentPipeline(csv_path='backend/instagram_analysis/data/engagements.csv')
results = pipeline.run_pipeline()
print(f'✅ Enrichment Pipeline Complete: {results[\"total_comments\"]} comments processed')
"

# Step 3: Generate Enrichment Report
echo ""
echo "📈 Step 3: Generating Enrichment Analysis Report..."
python -c "
from backend.instagram_analysis.enrichment_report_generator import EnrichmentReportGenerator
generator = EnrichmentReportGenerator()
generator.generate_report()
print('✅ Enrichment Report generated: reports/enrichment_report.md')
"

# Step 4: Trend Analysis
echo ""
echo "📊 Step 4: Running Advanced Trend Analysis..."
python -c "
from backend.instagram_analysis.trend_analysis import TrendAnalyzer
analyzer = TrendAnalyzer()
summary = analyzer.run_comprehensive_analysis()
print('✅ Trend Analysis Complete: reports/trend_analysis_report.md')
"

echo ""
echo "🎉 PIPELINE COMPLETE!"
echo "===================="
echo "📁 Generated Reports:"
echo "  • reports/eda_report.md - Exploratory Data Analysis"
echo "  • reports/enrichment_report.md - Data Enrichment Analysis"  
echo "  • reports/trend_analysis_report.md - Advanced Trend Intelligence"
echo "  • reports/enriched_instagram_data.sqlite - Processed Database"
echo ""
echo "📊 Report Images:"
echo "  • reports/images/ - EDA visualizations"
echo "  • reports/enrichment_images/ - Enrichment charts"
echo "  • reports/trend_images/ - Trend analysis graphs"
echo ""
echo "⚡ Performance: ~17,841 comments processed with parallel optimization"
echo "🎯 Ready for backend server: ./scripts/run_server.sh" 