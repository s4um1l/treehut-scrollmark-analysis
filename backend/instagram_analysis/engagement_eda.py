#!/usr/bin/env python3
"""
🔍 ENGAGEMENT DATA - COMPREHENSIVE EDA ANALYSIS
Real Instagram engagement data from client (17,841 comments, 355 posts, 32 days)

Layer 1: Deep Exploratory Data Analysis
- Data quality assessment
- Temporal patterns and trends
- Content analysis and engagement patterns
- Business insights extraction
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
warnings.filterwarnings('ignore')

class EngagementEDA:
    """Comprehensive EDA for Instagram engagement data"""
    
    def __init__(self, csv_path='instagram_analysis/data/engagements.csv'):
        """Initialize with engagement data"""
        print("🚀 Loading Real Instagram Engagement Data...")
        self.df = pd.read_csv(csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='mixed')
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
    
    def run_comprehensive_eda(self):
        """Run complete EDA analysis"""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE EDA ANALYSIS - REAL INSTAGRAM ENGAGEMENT DATA")
        print("="*80)
        
        # 1. Dataset Overview
        self._dataset_overview()
        
        # 2. Data Quality Assessment
        self._data_quality_analysis()
        
        # 3. Temporal Analysis
        self._temporal_analysis()
        
        # 4. Content Analysis
        self._content_analysis()
        
        # 5. Engagement Patterns
        self._engagement_patterns()
        
        # 6. Post Performance Analysis
        self._post_performance_analysis()
        
        # 7. Business Intelligence Insights
        self._business_insights()
        
        # 8. Recommendations for Layer 2 & 3
        self._layer_recommendations()
        
        return self._generate_summary()
    
    def _dataset_overview(self):
        """Basic dataset statistics"""
        print("\n📊 DATASET OVERVIEW")
        print("-" * 40)
        print(f"Total Comments: {len(self.df):,}")
        print(f"Unique Posts: {self.df['media_id'].nunique():,}")
        print(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Analysis Period: {(self.df['date'].max() - self.df['date'].min()).days} days")
        print(f"Average Comments/Day: {len(self.df) / self.df['date'].nunique():.1f}")
        print(f"Average Comments/Post: {len(self.df) / self.df['media_id'].nunique():.1f}")
        
    def _data_quality_analysis(self):
        """Assess data quality and completeness"""
        print("\n🔍 DATA QUALITY ASSESSMENT")
        print("-" * 40)
        
        # Missing data analysis
        missing_data = self.df.isnull().sum()
        print("Missing Data:")
        for col, missing in missing_data.items():
            if missing > 0:
                print(f"  • {col}: {missing:,} ({missing/len(self.df)*100:.1f}%)")
        
        # Data quality flags
        quality_issues = {
            'Empty comments': len(self.df[self.df['comment_text'].isnull()]),
            'Very short comments (<5 chars)': len(self.df[self.df['is_very_short']]),
            'Missing captions': len(self.df[self.df['media_caption'].isnull()]),
            'Duplicate comments': len(self.df) - len(self.df.drop_duplicates(['comment_text', 'media_id']))
        }
        
        print("\nQuality Flags:")
        for issue, count in quality_issues.items():
            if count > 0:
                print(f"  • {issue}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        # Data quality score
        total_issues = sum(quality_issues.values())
        quality_score = max(0, (1 - total_issues / len(self.df)) * 100)
        print(f"\n📈 Overall Data Quality Score: {quality_score:.1f}%")
        
    def _temporal_analysis(self):
        """Analyze temporal patterns in engagement"""
        print("\n⏰ TEMPORAL ANALYSIS")
        print("-" * 40)
        
        # Daily engagement trends
        daily_comments = self.df.groupby('date').size()
        print(f"Daily Engagement:")
        print(f"  • Peak day: {daily_comments.idxmax()} ({daily_comments.max():,} comments)")
        print(f"  • Lowest day: {daily_comments.idxmin()} ({daily_comments.min():,} comments)")
        print(f"  • Average daily: {daily_comments.mean():.1f} comments")
        print(f"  • Standard deviation: {daily_comments.std():.1f}")
        
        # Hourly patterns
        hourly_comments = self.df.groupby('hour').size()
        print(f"\nHourly Patterns:")
        print(f"  • Peak hour: {hourly_comments.idxmax()}:00 ({hourly_comments.max():,} comments)")
        print(f"  • Quiet hour: {hourly_comments.idxmin()}:00 ({hourly_comments.min():,} comments)")
        
        # Day of week patterns
        dow_comments = self.df.groupby('day_of_week').size()
        print(f"\nDay of Week Patterns:")
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            if day in dow_comments.index:
                print(f"  • {day}: {dow_comments[day]:,} comments")
        
        # Weekly trends
        weekly_comments = self.df.groupby('week').size()
        if len(weekly_comments) > 1:
            trend = "📈 Increasing" if weekly_comments.iloc[-1] > weekly_comments.iloc[0] else "📉 Decreasing"
            print(f"\nWeekly Trend: {trend}")
            print(f"  • Week 1: {weekly_comments.iloc[0]:,} comments")
            print(f"  • Latest week: {weekly_comments.iloc[-1]:,} comments")
    
    def _content_analysis(self):
        """Analyze comment content patterns"""
        print("\n📝 CONTENT ANALYSIS")
        print("-" * 40)
        
        # Comment length analysis
        print("Comment Length Distribution:")
        print(f"  • Average: {self.df['comment_length'].mean():.1f} characters")
        print(f"  • Median: {self.df['comment_length'].median():.1f} characters")
        print(f"  • Range: {self.df['comment_length'].min()} - {self.df['comment_length'].max()}")
        
        # Content type analysis
        content_types = {
            'Has emojis': self.df['has_emoji'].sum(),
            'Has mentions (@)': self.df['has_mention'].sum(),
            'Has hashtags (#)': self.df['has_hashtag'].sum(),
            'Has questions (?)': self.df['has_question'].sum(),
            'Substantive (20+ chars)': self.df['is_substantive'].sum()
        }
        
        print("\nContent Types:")
        for content_type, count in content_types.items():
            print(f"  • {content_type}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        # Most common words (simple analysis)
        all_text = ' '.join(self.df['comment_text'].dropna().astype(str).str.lower())
        words = re.findall(r'\b\w+\b', all_text)
        common_words = Counter(words).most_common(10)
        
        print("\nMost Common Words:")
        for word, count in common_words:
            if len(word) > 2:  # Filter out very short words
                print(f"  • '{word}': {count:,} times")
    
    def _engagement_patterns(self):
        """Analyze engagement patterns and behaviors"""
        print("\n🎯 ENGAGEMENT PATTERNS")
        print("-" * 40)
        
        # User behavior patterns
        user_mentions = self.df[self.df['has_mention']]['comment_text'].str.extractall(r'@(\w+)')[0].value_counts()
        if len(user_mentions) > 0:
            print("Top Mentioned Users:")
            for user, count in user_mentions.head(5).items():
                print(f"  • @{user}: {count:,} mentions")
        
        # Hashtag analysis
        hashtags = self.df[self.df['has_hashtag']]['comment_text'].str.extractall(r'#(\w+)')[0].value_counts()
        if len(hashtags) > 0:
            print("\nTop Hashtags:")
            for hashtag, count in hashtags.head(5).items():
                print(f"  • #{hashtag}: {count:,} uses")
        
        # Engagement quality indicators
        print(f"\nEngagement Quality:")
        print(f"  • Emoji usage: {self.df['has_emoji'].mean()*100:.1f}%")
        print(f"  • Question asking: {self.df['has_question'].mean()*100:.1f}%")
        print(f"  • Substantive comments: {self.df['is_substantive'].mean()*100:.1f}%")
        print(f"  • Very short comments: {self.df['is_very_short'].mean()*100:.1f}%")
    
    def _post_performance_analysis(self):
        """Analyze individual post performance"""
        print("\n🏆 POST PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        # Comments per post distribution
        post_comments = self.df.groupby('media_id').size()
        print("Comments per Post:")
        print(f"  • Average: {post_comments.mean():.1f}")
        print(f"  • Median: {post_comments.median():.1f}")
        print(f"  • Top performing: {post_comments.max():,} comments")
        print(f"  • Lowest performing: {post_comments.min():,} comments")
        
        # Top performing posts
        top_posts = post_comments.nlargest(5)
        print("\nTop 5 Performing Posts:")
        for i, (media_id, comment_count) in enumerate(top_posts.items(), 1):
            caption = self.df[self.df['media_id'] == media_id]['media_caption'].iloc[0]
            caption_preview = str(caption)[:100] + "..." if len(str(caption)) > 100 else str(caption)
            print(f"  {i}. {comment_count:,} comments - {caption_preview}")
        
        # Caption length vs engagement correlation
        post_stats = self.df.groupby('media_id').agg({
            'comment_text': 'count',
            'caption_length': 'first'
        }).rename(columns={'comment_text': 'comment_count'})
        
        correlation = post_stats['comment_count'].corr(post_stats['caption_length'])
        print(f"\nCaption Length vs Engagement Correlation: {correlation:.3f}")
        
    def _business_insights(self):
        """Extract key business insights"""
        print("\n💡 BUSINESS INTELLIGENCE INSIGHTS")
        print("-" * 40)
        
        # Tree Hut specific product mentions
        product_patterns = {
            'Tropical Mist': r'tropical\s*mist',
            'Vanilla Dream': r'vanilla\s*dream',
            'Coconut Lime': r'coconut\s*lime',
            'Shea Sugar Scrub': r'shea\s*sugar\s*scrub',
            'Moroccan Rose': r'moroccan\s*rose',
            'Brazilian Bum Bum': r'brazilian\s*bum',
            'Honey Oat': r'honey\s*oat',
            'Espresso Martini': r'espresso\s*martini'
        }
        
        product_mentions = {}
        for product, pattern in product_patterns.items():
            mentions = self.df['comment_text'].str.contains(pattern, case=False, na=False).sum()
            if mentions > 0:
                product_mentions[product] = mentions
        
        if product_mentions:
            print("Product Mentions in Comments:")
            sorted_products = sorted(product_mentions.items(), key=lambda x: x[1], reverse=True)
            for product, count in sorted_products:
                print(f"  • {product}: {count:,} mentions")
        
        # Sentiment indicators (basic)
        positive_words = ['love', 'amazing', 'great', 'perfect', 'best', 'good', 'awesome', 'beautiful']
        negative_words = ['hate', 'bad', 'terrible', 'awful', 'worst', 'disappointed', 'horrible']
        
        positive_comments = self.df['comment_text'].str.contains('|'.join(positive_words), case=False, na=False).sum()
        negative_comments = self.df['comment_text'].str.contains('|'.join(negative_words), case=False, na=False).sum()
        
        print(f"\nBasic Sentiment Indicators:")
        print(f"  • Positive keywords: {positive_comments:,} comments ({positive_comments/len(self.df)*100:.1f}%)")
        print(f"  • Negative keywords: {negative_comments:,} comments ({negative_comments/len(self.df)*100:.1f}%)")
        print(f"  • Sentiment ratio: {positive_comments/max(negative_comments, 1):.1f}:1 (positive:negative)")
        
        # Engagement health score
        health_factors = [
            self.df['is_substantive'].mean(),  # Substantive content
            self.df['has_emoji'].mean(),       # Emoji usage
            self.df['has_question'].mean(),    # Question engagement
            1 - self.df['is_very_short'].mean(), # Non-trivial comments
            positive_comments / max(len(self.df), 1)  # Positive sentiment
        ]
        
        health_score = sum(health_factors) / len(health_factors) * 100
        print(f"\n📊 Community Health Score: {health_score:.1f}%")
    
    def _layer_recommendations(self):
        """Recommendations for Layer 2 and Layer 3"""
        print("\n🚀 RECOMMENDATIONS FOR NEXT LAYERS")
        print("-" * 40)
        
        print("Layer 2 - Data Enrichment Priorities:")
        print("  1. 🧠 Sentiment Analysis - Apply enhanced sentiment to all 17,841 comments")
        print("  2. 🛡️ Safety Detection - Scan for inappropriate content and spam")
        print("  3. 🏷️ Product Intelligence - Extract and categorize product mentions")
        print("  4. 📊 Intent Classification - Categorize comments (praise, complaint, question, etc.)")
        print("  5. ⏰ Temporal Intelligence - Identify peak engagement windows")
        print("  6. 👥 User Behavior Analysis - Identify power users and engagement patterns")
        
        print("\nLayer 3 - Insights & Dashboard Features:")
        print("  1. 📈 Real-time Engagement Dashboard")
        print("  2. 🎯 Product Performance Tracking")
        print("  3. ⚠️ Crisis Detection & Alerts")
        print("  4. 💎 UGC Opportunity Identification")
        print("  5. 📊 Competitor Mention Analysis")
        print("  6. 🕐 Optimal Posting Time Recommendations")
        print("  7. 👑 Influencer & Brand Ambassador Detection")
        
    def _generate_summary(self):
        """Generate executive summary"""
        summary = {
            'dataset_size': len(self.df),
            'unique_posts': self.df['media_id'].nunique(),
            'date_range': f"{self.df['date'].min()} to {self.df['date'].max()}",
            'avg_daily_comments': len(self.df) / self.df['date'].nunique(),
            'avg_comment_length': self.df['comment_length'].mean(),
            'emoji_usage_rate': self.df['has_emoji'].mean() * 100,
            'substantive_comments_rate': self.df['is_substantive'].mean() * 100,
            'peak_hour': self.df.groupby('hour').size().idxmax(),
            'data_quality_score': max(0, (1 - (self.df.isnull().sum().sum() + len(self.df[self.df['is_very_short']])) / len(self.df)) * 100)
        }
        
        print("\n" + "="*80)
        print("📋 EXECUTIVE SUMMARY")
        print("="*80)
        print(f"✅ Analyzed {summary['dataset_size']:,} real Instagram comments")
        print(f"📊 Covering {summary['unique_posts']:,} posts over {(self.df['date'].max() - self.df['date'].min()).days} days")
        print(f"🎯 Average {summary['avg_daily_comments']:.0f} comments/day, peak at {summary['peak_hour']}:00")
        print(f"💬 {summary['avg_comment_length']:.1f} avg chars, {summary['emoji_usage_rate']:.1f}% use emojis")
        print(f"🏆 {summary['substantive_comments_rate']:.1f}% substantive comments")
        print(f"✨ Data Quality Score: {summary['data_quality_score']:.1f}%")
        print("\n🚀 READY FOR LAYER 2: Universal Pipeline Intelligence Enrichment")
        
        return summary

def main():
    """Run comprehensive EDA analysis"""
    try:
        eda = EngagementEDA()
        summary = eda.run_comprehensive_eda()
        
        print(f"\n🎉 EDA Complete! Ready to proceed with Layer 2 enrichment.")
        return summary
        
    except Exception as e:
        print(f"❌ EDA Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 