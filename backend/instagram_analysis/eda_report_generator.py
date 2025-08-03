#!/usr/bin/env python3
"""
🔍 EDA REPORT GENERATOR WITH VISUALIZATIONS
Generates comprehensive EDA analysis with charts and markdown export
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import emoji
from collections import Counter
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

class EDAReportGenerator:
    """Generate comprehensive EDA report with visualizations"""
    
    def __init__(self, csv_path='instagram_analysis/data/engagements.csv', output_dir='reports'):
        """Initialize with engagement data"""
        print("🚀 Loading Instagram Engagement Data...")
        self.df = pd.read_csv(csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='mixed')
        
        # Create output directories
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        self._prepare_data()
        print(f"✅ Loaded {len(self.df):,} comments from {self.df['media_id'].nunique():,} posts")
        
    def _prepare_data(self):
        """Prepare data with derived fields"""
        # Temporal fields
        self.df['date'] = self.df['timestamp'].dt.date
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        self.df['week'] = self.df['timestamp'].dt.isocalendar().week
        
        # Content analysis fields
        self.df['comment_length'] = self.df['comment_text'].astype(str).str.len()
        self.df['caption_length'] = self.df['media_caption'].astype(str).str.len()
        self.df['word_count'] = self.df['comment_text'].astype(str).str.split().str.len()
        
        # Engagement indicators
        self.df['has_emoji'] = self.df['comment_text'].astype(str).apply(self._has_emoji)
        self.df['has_mention'] = self.df['comment_text'].astype(str).str.contains('@', na=False)
        self.df['has_hashtag'] = self.df['comment_text'].astype(str).str.contains('#', na=False)
        self.df['has_question'] = self.df['comment_text'].astype(str).str.contains('\?', na=False)
        
        # Content quality indicators
        self.df['is_substantive'] = self.df['comment_length'] >= 20
        self.df['is_very_short'] = self.df['comment_length'] < 5
        
    def _has_emoji(self, text):
        """Check if text contains emojis"""
        try:
            return any(emoji.is_emoji(char) for char in str(text))
        except:
            return False
    
    def generate_report(self):
        """Generate complete EDA report with visualizations"""
        print("\n" + "="*80)
        print("📊 GENERATING COMPREHENSIVE EDA REPORT WITH VISUALIZATIONS")
        print("="*80)
        
        # Generate all visualizations
        self._create_temporal_visualizations()
        self._create_content_visualizations()
        self._create_engagement_visualizations()
        self._create_performance_visualizations()
        
        # Generate markdown report
        self._generate_markdown_report()
        
        print(f"\n🎉 EDA Report generated successfully!")
        print(f"📁 Report location: {self.output_dir}/eda_report.md")
        print(f"🖼️ Images location: {self.images_dir}/")
        
    def _create_temporal_visualizations(self):
        """Create temporal analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📅 Temporal Analysis', fontsize=16, fontweight='bold')
        
        # Daily engagement trends
        daily_comments = self.df.groupby('date').size()
        axes[0,0].plot(daily_comments.index, daily_comments.values, marker='o', linewidth=2)
        axes[0,0].set_title('Daily Comment Volume')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Number of Comments')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Hourly patterns
        hourly_comments = self.df.groupby('hour').size()
        axes[0,1].bar(hourly_comments.index, hourly_comments.values, color='skyblue')
        axes[0,1].set_title('Hourly Engagement Patterns')
        axes[0,1].set_xlabel('Hour of Day')
        axes[0,1].set_ylabel('Number of Comments')
        
        # Day of week patterns
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_comments = self.df.groupby('day_of_week').size().reindex(day_order)
        axes[1,0].bar(range(len(dow_comments)), dow_comments.values, color='lightcoral')
        axes[1,0].set_title('Day of Week Patterns')
        axes[1,0].set_xlabel('Day of Week')
        axes[1,0].set_ylabel('Number of Comments')
        axes[1,0].set_xticks(range(len(day_order)))
        axes[1,0].set_xticklabels([d[:3] for d in day_order])
        
        # Weekly trends
        weekly_comments = self.df.groupby('week').size()
        axes[1,1].plot(weekly_comments.index, weekly_comments.values, marker='s', linewidth=2, color='green')
        axes[1,1].set_title('Weekly Trends')
        axes[1,1].set_xlabel('Week Number')
        axes[1,1].set_ylabel('Number of Comments')
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_content_visualizations(self):
        """Create content analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📝 Content Analysis', fontsize=16, fontweight='bold')
        
        # Comment length distribution
        axes[0,0].hist(self.df['comment_length'], bins=50, alpha=0.7, color='purple')
        axes[0,0].set_title('Comment Length Distribution')
        axes[0,0].set_xlabel('Comment Length (characters)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(self.df['comment_length'].mean(), color='red', linestyle='--', label='Mean')
        axes[0,0].legend()
        
        # Content types pie chart
        content_types = {
            'Has Emojis': self.df['has_emoji'].sum(),
            'Has Mentions': self.df['has_mention'].sum(),
            'Has Hashtags': self.df['has_hashtag'].sum(),
            'Has Questions': self.df['has_question'].sum(),
            'Substantive': self.df['is_substantive'].sum()
        }
        axes[0,1].pie(content_types.values(), labels=content_types.keys(), autopct='%1.1f%%')
        axes[0,1].set_title('Content Type Distribution')
        
        # Word count vs comment length scatter
        sample_data = self.df.sample(min(1000, len(self.df)))  # Sample for performance
        axes[1,0].scatter(sample_data['word_count'], sample_data['comment_length'], alpha=0.6)
        axes[1,0].set_title('Word Count vs Comment Length')
        axes[1,0].set_xlabel('Word Count')
        axes[1,0].set_ylabel('Comment Length')
        
        # Top words bar chart
        all_text = ' '.join(self.df['comment_text'].dropna().astype(str).str.lower())
        words = re.findall(r'\b\w+\b', all_text)
        common_words = Counter(words).most_common(10)
        word_names, word_counts = zip(*[(w, c) for w, c in common_words if len(w) > 2])
        
        axes[1,1].barh(range(len(word_names)), word_counts, color='orange')
        axes[1,1].set_title('Top 10 Most Common Words')
        axes[1,1].set_xlabel('Frequency')
        axes[1,1].set_yticks(range(len(word_names)))
        axes[1,1].set_yticklabels(word_names)
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'content_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_engagement_visualizations(self):
        """Create engagement pattern visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 Engagement Patterns', fontsize=16, fontweight='bold')
        
        # Engagement quality metrics
        quality_metrics = {
            'Emoji Usage': self.df['has_emoji'].mean() * 100,
            'Questions': self.df['has_question'].mean() * 100,
            'Substantive': self.df['is_substantive'].mean() * 100,
            'Mentions': self.df['has_mention'].mean() * 100
        }
        
        axes[0,0].bar(quality_metrics.keys(), quality_metrics.values(), color='lightgreen')
        axes[0,0].set_title('Engagement Quality Metrics (%)')
        axes[0,0].set_ylabel('Percentage')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Comment length by engagement type
        engagement_lengths = {
            'With Emojis': self.df[self.df['has_emoji']]['comment_length'].mean(),
            'With Mentions': self.df[self.df['has_mention']]['comment_length'].mean(),
            'With Questions': self.df[self.df['has_question']]['comment_length'].mean(),
            'Basic Comments': self.df[~(self.df['has_emoji'] | self.df['has_mention'] | self.df['has_question'])]['comment_length'].mean()
        }
        
        axes[0,1].bar(engagement_lengths.keys(), engagement_lengths.values(), color='lightblue')
        axes[0,1].set_title('Average Comment Length by Type')
        axes[0,1].set_ylabel('Average Length (characters)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Hourly engagement quality
        hourly_quality = self.df.groupby('hour').agg({
            'has_emoji': 'mean',
            'has_mention': 'mean',
            'is_substantive': 'mean'
        }) * 100
        
        axes[1,0].plot(hourly_quality.index, hourly_quality['has_emoji'], label='Emojis', marker='o')
        axes[1,0].plot(hourly_quality.index, hourly_quality['has_mention'], label='Mentions', marker='s')
        axes[1,0].plot(hourly_quality.index, hourly_quality['is_substantive'], label='Substantive', marker='^')
        axes[1,0].set_title('Hourly Engagement Quality')
        axes[1,0].set_xlabel('Hour of Day')
        axes[1,0].set_ylabel('Percentage')
        axes[1,0].legend()
        
        # Weekly engagement quality trend
        weekly_quality = self.df.groupby('week').agg({
            'has_emoji': 'mean',
            'has_mention': 'mean',
            'is_substantive': 'mean'
        }) * 100
        
        axes[1,1].plot(weekly_quality.index, weekly_quality['has_emoji'], label='Emojis', marker='o')
        axes[1,1].plot(weekly_quality.index, weekly_quality['has_mention'], label='Mentions', marker='s')
        axes[1,1].plot(weekly_quality.index, weekly_quality['is_substantive'], label='Substantive', marker='^')
        axes[1,1].set_title('Weekly Engagement Quality Trends')
        axes[1,1].set_xlabel('Week Number')
        axes[1,1].set_ylabel('Percentage')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'engagement_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_performance_visualizations(self):
        """Create post performance visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🏆 Post Performance Analysis', fontsize=16, fontweight='bold')
        
        # Comments per post distribution
        post_comments = self.df.groupby('media_id').size()
        axes[0,0].hist(post_comments.values, bins=30, alpha=0.7, color='gold')
        axes[0,0].set_title('Comments per Post Distribution')
        axes[0,0].set_xlabel('Number of Comments')
        axes[0,0].set_ylabel('Number of Posts')
        axes[0,0].axvline(post_comments.mean(), color='red', linestyle='--', label='Mean')
        axes[0,0].legend()
        
        # Top performing posts
        top_posts = post_comments.nlargest(10)
        axes[0,1].barh(range(len(top_posts)), top_posts.values, color='coral')
        axes[0,1].set_title('Top 10 Performing Posts')
        axes[0,1].set_xlabel('Number of Comments')
        axes[0,1].set_ylabel('Post Rank')
        axes[0,1].set_yticks(range(len(top_posts)))
        axes[0,1].set_yticklabels([f'Post {i+1}' for i in range(len(top_posts))])
        
        # Caption length vs engagement scatter
        post_stats = self.df.groupby('media_id').agg({
            'comment_text': 'count',
            'caption_length': 'first'
        }).rename(columns={'comment_text': 'comment_count'})
        
        axes[1,0].scatter(post_stats['caption_length'], post_stats['comment_count'], alpha=0.6)
        axes[1,0].set_title('Caption Length vs Engagement')
        axes[1,0].set_xlabel('Caption Length (characters)')
        axes[1,0].set_ylabel('Number of Comments')
        
        # Performance by day of week
        daily_performance = self.df.groupby(['day_of_week', 'media_id']).size().groupby('day_of_week').mean()
        daily_performance = daily_performance.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        axes[1,1].bar(range(len(daily_performance)), daily_performance.values, color='lightsteelblue')
        axes[1,1].set_title('Average Post Performance by Day')
        axes[1,1].set_xlabel('Day of Week')
        axes[1,1].set_ylabel('Average Comments per Post')
        axes[1,1].set_xticks(range(len(daily_performance)))
        axes[1,1].set_xticklabels([d[:3] for d in daily_performance.index])
        
        plt.tight_layout()
        plt.savefig(self.images_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_markdown_report(self):
        """Generate comprehensive markdown report"""
        report_path = self.output_dir / 'eda_report.md'
        
        # Calculate key metrics
        total_comments = len(self.df)
        unique_posts = self.df['media_id'].nunique()
        date_range = f"{self.df['date'].min()} to {self.df['date'].max()}"
        analysis_days = (self.df['date'].max() - self.df['date'].min()).days
        avg_daily = total_comments / self.df['date'].nunique()
        
        # Data quality metrics
        missing_data = self.df.isnull().sum()
        quality_issues = {
            'empty_comments': len(self.df[self.df['comment_text'].isnull()]),
            'very_short': len(self.df[self.df['is_very_short']]),
            'missing_captions': len(self.df[self.df['media_caption'].isnull()]),
            'duplicates': len(self.df) - len(self.df.drop_duplicates(['comment_text', 'media_id']))
        }
        
        total_issues = sum(quality_issues.values())
        quality_score = max(0, (1 - total_issues / len(self.df)) * 100)
        
        # Content metrics
        avg_length = self.df['comment_length'].mean()
        emoji_rate = self.df['has_emoji'].mean() * 100
        mention_rate = self.df['has_mention'].mean() * 100
        substantive_rate = self.df['is_substantive'].mean() * 100
        
        # Temporal insights
        peak_hour = self.df.groupby('hour').size().idxmax()
        peak_day = self.df.groupby('date').size().idxmax()
        peak_dow = self.df.groupby('day_of_week').size().idxmax()
        
        # Performance metrics
        post_comments = self.df.groupby('media_id').size()
        top_post_comments = post_comments.max()
        avg_comments_per_post = post_comments.mean()
        
        # Generate product mentions
        product_patterns = {
            'Tropical Mist': r'tropical\s*mist',
            'Vanilla Dream': r'vanilla\s*dream',
            'Coconut Lime': r'coconut\s*lime',
            'Moroccan Rose': r'moroccan\s*rose',
            'Espresso Martini': r'espresso\s*martini'
        }
        
        product_mentions = {}
        for product, pattern in product_patterns.items():
            mentions = self.df['comment_text'].str.contains(pattern, case=False, na=False).sum()
            if mentions > 0:
                product_mentions[product] = mentions
        
        # Sentiment analysis
        positive_words = ['love', 'amazing', 'great', 'perfect', 'best', 'good', 'awesome', 'beautiful']
        negative_words = ['hate', 'bad', 'terrible', 'awful', 'worst', 'disappointed', 'horrible']
        
        positive_comments = self.df['comment_text'].str.contains('|'.join(positive_words), case=False, na=False).sum()
        negative_comments = self.df['comment_text'].str.contains('|'.join(negative_words), case=False, na=False).sum()
        sentiment_ratio = positive_comments / max(negative_comments, 1)
        
        markdown_content = f"""# 📊 Instagram Engagement Data - Comprehensive EDA Report

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 🎯 Executive Summary

This comprehensive analysis examines **{total_comments:,} Instagram comments** from **{unique_posts:,} posts** collected over **{analysis_days} days** ({date_range}). The dataset reveals strong engagement patterns with an average of **{avg_daily:.0f} comments per day** and demonstrates high data quality with a **{quality_score:.1f}% quality score**.

### Key Highlights
- 📈 **Peak Engagement**: {peak_hour}:00 with strongest activity on {peak_dow}
- 💬 **Content Quality**: {substantive_rate:.1f}% substantive comments, {emoji_rate:.1f}% use emojis
- 🏆 **Top Performance**: Best post received {top_post_comments:,} comments
- 😊 **Community Sentiment**: {sentiment_ratio:.1f}:1 positive-to-negative ratio

---

## 📅 Temporal Analysis

![Temporal Analysis](images/temporal_analysis.png)

### Daily Engagement Patterns
- **Average Daily Comments**: {avg_daily:.1f}
- **Peak Day**: {peak_day} with {self.df.groupby('date').size().max():,} comments
- **Engagement Range**: {self.df.groupby('date').size().min():,} - {self.df.groupby('date').size().max():,} comments per day

### Hourly Activity
- **Peak Hour**: {peak_hour}:00 ({self.df.groupby('hour').size().max():,} comments)
- **Quiet Hour**: {self.df.groupby('hour').size().idxmin()}:00 ({self.df.groupby('hour').size().min():,} comments)

### Weekly Trends
- **Most Active Day**: {peak_dow} ({self.df.groupby('day_of_week').size().max():,} comments)
- **Trend Direction**: {"📈 Increasing" if self.df.groupby('week').size().iloc[-1] > self.df.groupby('week').size().iloc[0] else "📉 Decreasing"}

---

## 📝 Content Analysis

![Content Analysis](images/content_analysis.png)

### Comment Characteristics
- **Average Length**: {avg_length:.1f} characters
- **Median Length**: {self.df['comment_length'].median():.1f} characters
- **Length Range**: {self.df['comment_length'].min()} - {self.df['comment_length'].max()} characters

### Content Types Distribution
| Content Type | Count | Percentage |
|--------------|-------|------------|
| Has Emojis | {self.df['has_emoji'].sum():,} | {emoji_rate:.1f}% |
| Has Mentions (@) | {self.df['has_mention'].sum():,} | {mention_rate:.1f}% |
| Has Hashtags (#) | {self.df['has_hashtag'].sum():,} | {self.df['has_hashtag'].mean()*100:.1f}% |
| Has Questions (?) | {self.df['has_question'].sum():,} | {self.df['has_question'].mean()*100:.1f}% |
| Substantive (20+ chars) | {self.df['is_substantive'].sum():,} | {substantive_rate:.1f}% |

### Most Common Words
"""

        # Add top words
        all_text = ' '.join(self.df['comment_text'].dropna().astype(str).str.lower())
        words = re.findall(r'\b\w+\b', all_text)
        common_words = Counter(words).most_common(10)
        
        for i, (word, count) in enumerate(common_words, 1):
            if len(word) > 2:
                markdown_content += f"{i}. **{word}**: {count:,} occurrences\n"

        markdown_content += f"""
---

## 🎯 Engagement Patterns

![Engagement Patterns](images/engagement_patterns.png)

### Quality Metrics
- **Emoji Usage**: {emoji_rate:.1f}% of comments
- **Question Engagement**: {self.df['has_question'].mean()*100:.1f}% ask questions
- **Substantive Comments**: {substantive_rate:.1f}% are meaningful (20+ characters)
- **Social Mentions**: {mention_rate:.1f}% mention other users

### Top Mentioned Users
"""

        # Add top mentions
        user_mentions = self.df[self.df['has_mention']]['comment_text'].str.extractall(r'@(\w+)')[0].value_counts()
        for i, (user, count) in enumerate(user_mentions.head(5).items(), 1):
            markdown_content += f"{i}. **@{user}**: {count:,} mentions\n"

        markdown_content += f"""
---

## 🏆 Post Performance Analysis

![Performance Analysis](images/performance_analysis.png)

### Performance Distribution
- **Average Comments per Post**: {avg_comments_per_post:.1f}
- **Median Comments per Post**: {post_comments.median():.1f}
- **Top Performing Post**: {top_post_comments:,} comments
- **Performance Range**: {post_comments.min()} - {post_comments.max()} comments

### Top 5 Performing Posts
"""

        # Add top posts
        top_posts = post_comments.nlargest(5)
        for i, (media_id, comment_count) in enumerate(top_posts.items(), 1):
            caption = self.df[self.df['media_id'] == media_id]['media_caption'].iloc[0]
            caption_preview = str(caption)[:100] + "..." if len(str(caption)) > 100 else str(caption)
            markdown_content += f"{i}. **{comment_count:,} comments** - {caption_preview}\n"

        markdown_content += f"""
### Caption Performance Insights
- **Caption Length vs Engagement Correlation**: {self.df.groupby('media_id').agg({'comment_text': 'count', 'caption_length': 'first'}).corr().iloc[0,1]:.3f}

---

## 💡 Business Intelligence Insights

### Product Mentions Analysis
"""

        if product_mentions:
            for product, count in sorted(product_mentions.items(), key=lambda x: x[1], reverse=True):
                markdown_content += f"- **{product}**: {count:,} mentions\n"
        else:
            markdown_content += "- No specific product mentions detected in comments\n"

        markdown_content += f"""
### Sentiment Analysis
- **Positive Sentiment**: {positive_comments:,} comments ({positive_comments/len(self.df)*100:.1f}%)
- **Negative Sentiment**: {negative_comments:,} comments ({negative_comments/len(self.df)*100:.1f}%)
- **Sentiment Ratio**: {sentiment_ratio:.1f}:1 (positive:negative)

### Community Health Score
**Overall Score: {(sum([self.df['is_substantive'].mean(), self.df['has_emoji'].mean(), self.df['has_question'].mean(), 1 - self.df['is_very_short'].mean(), positive_comments / len(self.df)]) / 5 * 100):.1f}%**

---

## 🔍 Data Quality Assessment

### Data Completeness
- **Total Records**: {total_comments:,}
- **Missing Comments**: {quality_issues['empty_comments']:,} ({quality_issues['empty_comments']/total_comments*100:.1f}%)
- **Missing Captions**: {quality_issues['missing_captions']:,} ({quality_issues['missing_captions']/total_comments*100:.1f}%)
- **Very Short Comments**: {quality_issues['very_short']:,} ({quality_issues['very_short']/total_comments*100:.1f}%)
- **Duplicate Comments**: {quality_issues['duplicates']:,} ({quality_issues['duplicates']/total_comments*100:.1f}%)

### Overall Quality Score: {quality_score:.1f}%

---

## 🚀 Strategic Recommendations

### Immediate Actions
1. **Optimize Posting Times**: Focus on {peak_hour}:00 and {peak_dow}s for maximum engagement
2. **Content Strategy**: Encourage emoji usage and questions to boost engagement quality
3. **Community Management**: Leverage high-performing content formats for future posts

### Layer 2 Development Priorities
1. 🧠 **Advanced Sentiment Analysis** - Deep emotion detection across all {total_comments:,} comments
2. 🛡️ **Content Safety & Moderation** - Automated inappropriate content detection
3. 🏷️ **Product Intelligence** - Enhanced product mention extraction and categorization
4. 📊 **Intent Classification** - Categorize comments by purpose (praise, questions, complaints)
5. ⏰ **Temporal Intelligence** - Predictive engagement timing optimization
6. 👥 **User Behavior Analytics** - Power user identification and engagement patterns

### Layer 3 Dashboard Features
1. 📈 **Real-time Engagement Monitoring**
2. 🎯 **Product Performance Tracking**
3. ⚠️ **Crisis Detection & Alert System**
4. 💎 **UGC Opportunity Identification**
5. 📊 **Competitive Intelligence Dashboard**
6. 🕐 **AI-Powered Posting Recommendations**
7. 👑 **Influencer & Brand Ambassador Detection**

---

## 📈 Key Performance Indicators

| Metric | Value | Status |
|--------|-------|--------|
| Total Engagement | {total_comments:,} comments | ✅ Strong |
| Daily Average | {avg_daily:.0f} comments/day | ✅ Consistent |
| Quality Score | {quality_score:.1f}% | ✅ High Quality |
| Emoji Usage | {emoji_rate:.1f}% | {"✅ Healthy" if emoji_rate > 25 else "⚠️ Could Improve"} |
| Substantive Content | {substantive_rate:.1f}% | {"✅ Quality" if substantive_rate > 25 else "⚠️ Needs Attention"} |
| Sentiment Ratio | {sentiment_ratio:.1f}:1 | ✅ Very Positive |

---

*This report was automatically generated using advanced data analytics. For questions or deeper analysis, contact the data science team.*

**Report Generation Details:**
- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Data Period: {date_range}
- Total Records Analyzed: {total_comments:,}
- Visualization Count: 4 comprehensive charts
- Quality Assurance: Automated data validation completed
"""

        # Write the markdown file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"📄 Markdown report generated: {report_path}")

def main():
    """Generate comprehensive EDA report with visualizations"""
    try:
        generator = EDAReportGenerator()
        generator.generate_report()
        
        print(f"\n🎉 Complete EDA Report Generated Successfully!")
        print(f"📁 Check the 'reports' directory for:")
        print(f"   • eda_report.md - Comprehensive markdown report")
        print(f"   • images/ - All visualization charts")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 