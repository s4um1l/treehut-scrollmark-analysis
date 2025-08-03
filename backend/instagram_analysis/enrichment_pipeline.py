#!/usr/bin/env python3
"""
🚀 INSTAGRAM ENGAGEMENT DATA ENRICHMENT PIPELINE
Comprehensive data processing with standardization, enrichment, and storage

Features:
- Load & standardize (UTC timestamps, duplicates, validation)
- Light EDA (fast analysis)
- Multi-layer enrichment (language, spam, products, intent, sentiment)
- SQLite storage with indexing
- Optional semantic topics
"""

import pandas as pd
import numpy as np
import sqlite3
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
from pydantic import BaseModel, Field, validator
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class Comment(BaseModel):
    """Pydantic model for comment schema validation"""
    comment_id: Optional[str] = None
    media_id: str
    timestamp: datetime
    comment_text: str
    media_caption: Optional[str] = None
    
    # Derived fields
    date: Optional[str] = None
    week_of_month: Optional[int] = None
    comment_length: Optional[int] = None
    word_count: Optional[int] = None
    
    # Enrichment fields
    language: Optional[str] = None
    is_spam: Optional[bool] = None
    is_bot: Optional[bool] = None
    retailer: Optional[str] = None
    geo_region: Optional[str] = None
    product_mentions: Optional[List[str]] = None
    scent_mentions: Optional[List[str]] = None
    intent: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    
    # Engagement proxies
    has_mention: Optional[bool] = None
    emoji_count: Optional[int] = None
    question_mark_count: Optional[int] = None
    
    @validator('comment_text')
    def validate_comment_text(cls, v):
        if pd.isna(v) or v is None:
            return ""
        return str(v).strip()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class EnrichmentPipeline:
    """Comprehensive Instagram engagement data enrichment pipeline"""
    
    def __init__(self, csv_path: str = 'instagram_analysis/data/engagements.csv', 
                 output_db: str = 'enriched_data.sqlite', enable_semantic: bool = False):
        """Initialize the enrichment pipeline"""
        self.csv_path = csv_path
        self.output_db = output_db
        self.enable_semantic = enable_semantic
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Product and scent dictionaries
        self._setup_product_dictionaries()
        self._setup_retailer_geo_dictionaries()
        
        print("🚀 Instagram Enrichment Pipeline Initialized")
        
    def _setup_product_dictionaries(self):
        """Setup product and scent recognition dictionaries"""
        self.products = {
            'body_scrub': ['scrub', 'exfoliate', 'exfoliating', 'body scrub', 'sugar scrub', 'shea sugar'],
            'body_butter': ['body butter', 'butter', 'moisturizer', 'hydrating', 'nourishing'],
            'lip_scrub': ['lip scrub', 'lip exfoliant', 'lip care'],
            'body_oil': ['body oil', 'oil', 'glow oil'],
            'lip_balm': ['lip balm', 'lip moisturizer', 'lip care', 'chapstick']
        }
        
        self.scents = [
            'tangerine', 'moroccan rose', 'vanilla', 'vanilla dream', 'coco colada', 'coconut lime',
            'pumpkin pop', 'lotus water', 'pink champagne', 'electric beach', 'tropical mist',
            'honey oat', 'espresso martini', 'brazilian bum bum', 'shea sugar scrub',
            'coconut', 'rose', 'pumpkin', 'champagne', 'beach', 'honey', 'espresso',
            'bum bum', 'mist', 'lime', 'oat'
        ]
        
    def _setup_retailer_geo_dictionaries(self):
        """Setup retailer and geographic recognition"""
        self.retailers = [
            'target', 'walmart', 'ulta', 'amazon', 'sephora', 'cvs', 'walgreens',
            'bath and body works', 'sally beauty', 'tj maxx', 'marshall',
            'costco', 'sam', 'kroger', 'meijer', 'whole foods'
        ]
        
        self.geo_regions = [
            'canada', 'uk', 'australia', 'germany', 'france', 'italy', 'spain',
            'mexico', 'brazil', 'japan', 'korea', 'india', 'netherlands',
            'sweden', 'norway', 'finland', 'poland', 'russia', 'china'
        ]
        
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete enrichment pipeline"""
        print("\n" + "="*80)
        print("🔄 STARTING COMPREHENSIVE ENRICHMENT PIPELINE")
        print("="*80)
        
        # Step 1: Load & Standardize
        df = self._load_and_standardize()
        
        # Step 2: Light EDA
        eda_results = self._light_eda(df)
        
        # Step 3: Enrichment
        enriched_df = self._enrich_data(df)
        
        # Step 4: Optional Semantic Topics
        if self.enable_semantic:
            enriched_df = self._add_semantic_topics(enriched_df)
            
        # Step 5: Storage
        self._store_enriched_data(enriched_df)
        
        # Generate summary
        summary = self._generate_pipeline_summary(enriched_df, eda_results)
        
        print(f"\n🎉 Pipeline Complete! Enriched {len(enriched_df):,} comments")
        print(f"📊 Database: {self.output_db}")
        
        return summary
        
    def _load_and_standardize(self) -> pd.DataFrame:
        """Load and standardize the raw data"""
        print("\n📥 LOADING & STANDARDIZING DATA")
        print("-" * 40)
        
        # Load data (sample first 1000 for demo)
        df = pd.read_csv(self.csv_path)
        original_count = len(df)
        df = df.head(1000)  # Demo with first 1000 rows
        print(f"✅ Loaded {original_count:,} raw comments")
        print(f"🔬 Demo mode: Processing first {len(df):,} comments")
        
        # Parse UTC timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        
        # Add derived temporal fields
        df['date'] = df['timestamp'].dt.date.astype(str)
        df['week_of_month'] = df['timestamp'].dt.day.apply(lambda x: (x - 1) // 7 + 1)
        
        # Drop exact duplicates
        df_before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed = df_before_dedup - len(df)
        print(f"🧹 Removed {duplicates_removed:,} exact duplicates")
        
        # Normalize whitespace while keeping emojis
        df['comment_text'] = df['comment_text'].astype(str).apply(self._normalize_text)
        df['media_caption'] = df['media_caption'].astype(str).apply(self._normalize_text)
        
        # Add basic derived fields
        df['comment_length'] = df['comment_text'].str.len()
        df['word_count'] = df['comment_text'].str.split().str.len()
        
        print(f"✅ Standardized data: {len(df):,} comments ready for enrichment")
        return df
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text while preserving emojis"""
        if pd.isna(text) or text == 'nan':
            return ""
        
        # Trim whitespace and normalize spaces
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        
        return text
        
    def _light_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fast EDA analysis"""
        print("\n🔍 LIGHT EDA ANALYSIS")
        print("-" * 40)
        
        # Null analysis
        nulls = df.isnull().sum()
        print("Null Values:")
        for col, null_count in nulls.items():
            if null_count > 0:
                print(f"  • {col}: {null_count:,} ({null_count/len(df)*100:.1f}%)")
        
        # Length distribution
        length_stats = df['comment_length'].describe()
        print(f"\nComment Length Distribution:")
        print(f"  • Mean: {length_stats['mean']:.1f} chars")
        print(f"  • Median: {length_stats['50%']:.1f} chars") 
        print(f"  • Range: {length_stats['min']:.0f} - {length_stats['max']:.0f} chars")
        
        # Media ID comment counts & long tail
        media_counts = df.groupby('media_id').size()
        print(f"\nPost Engagement:")
        print(f"  • Total posts: {len(media_counts):,}")
        print(f"  • Avg comments/post: {media_counts.mean():.1f}")
        print(f"  • Top post: {media_counts.max():,} comments")
        print(f"  • Long tail (1-5 comments): {(media_counts <= 5).sum():,} posts ({(media_counts <= 5).mean()*100:.1f}%)")
        
        # Basic time histogram
        hourly_dist = df['timestamp'].dt.hour.value_counts().sort_index()
        peak_hour = hourly_dist.idxmax()
        print(f"\nTemporal Patterns:")
        print(f"  • Peak hour: {peak_hour}:00 ({hourly_dist[peak_hour]:,} comments)")
        print(f"  • Date range: {df['date'].min()} to {df['date'].max()}")
        
        return {
            'total_comments': len(df),
            'null_comments': nulls.get('comment_text', 0),
            'avg_length': length_stats['mean'],
            'total_posts': len(media_counts),
            'peak_hour': peak_hour,
            'long_tail_posts': (media_counts <= 5).sum()
        }
        
    def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all enrichment layers"""
        print("\n🔬 APPLYING ENRICHMENT LAYERS")
        print("-" * 40)
        
        enriched_df = df.copy()
        
        # Language detection
        enriched_df = self._detect_language(enriched_df)
        
        # Spam/Bot detection
        enriched_df = self._detect_spam_bots(enriched_df)
        
        # Retailer & Geo matching
        enriched_df = self._match_retailer_geo(enriched_df)
        
        # Product/Scent resolution
        enriched_df = self._resolve_products_scents(enriched_df)
        
        # Intent classification
        enriched_df = self._classify_intent(enriched_df)
        
        # Sentiment analysis
        enriched_df = self._analyze_sentiment(enriched_df)
        
        # Engagement proxies
        enriched_df = self._calculate_engagement_proxies(enriched_df)
        
        print(f"✅ Enrichment complete: {len(enriched_df.columns)} total columns")
        return enriched_df
        
    def _detect_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast heuristic language detection"""
        print("  🌐 Detecting languages...")
        
        def detect_lang(text):
            try:
                if len(str(text).strip()) < 3:
                    return 'unknown'
                return detect(str(text))
            except Exception:
                return 'unknown'
        
        df['language'] = df['comment_text'].apply(detect_lang)
        
        lang_dist = df['language'].value_counts()
        print(f"     • English: {lang_dist.get('en', 0):,} ({lang_dist.get('en', 0)/len(df)*100:.1f}%)")
        print(f"     • Other languages: {len(df) - lang_dist.get('en', 0):,}")
        
        return df
        
    def _detect_spam_bots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spam and bot detection rules"""
        print("  🤖 Detecting spam/bots...")
        
        # URL spam detection
        df['has_url'] = df['comment_text'].str.contains(r'http[s]?://|www\.', case=False, na=False)
        
        # Repeated pattern spam (e.g., "doctor bradley shaffer")
        spam_patterns = [
            r'doctor\s+(bradley\s+)?shaffer',
            r'telegram\s*@',
            r'whatsapp\s*\+',
            r'dm\s+me\s+for',
            r'click\s+link',
            r'visit\s+my\s+profile'
        ]
        
        spam_pattern = '|'.join(spam_patterns)
        df['has_spam_pattern'] = df['comment_text'].str.contains(spam_pattern, case=False, na=False)
        
        # Excessive mentions (likely bots)
        df['mention_count'] = df['comment_text'].str.count('@')
        df['excessive_mentions'] = df['mention_count'] > 3
        
        # Combine spam indicators
        df['is_spam'] = (df['has_url'] | df['has_spam_pattern'] | df['excessive_mentions'])
        
        # Bot detection (repetitive short comments)
        comment_freq = df['comment_text'].value_counts()
        repetitive_comments = comment_freq[comment_freq > 50].index
        df['is_bot'] = df['comment_text'].isin(repetitive_comments) & (df['comment_length'] < 10)
        
        spam_count = df['is_spam'].sum()
        bot_count = df['is_bot'].sum()
        print(f"     • Spam detected: {spam_count:,} ({spam_count/len(df)*100:.1f}%)")
        print(f"     • Bots detected: {bot_count:,} ({bot_count/len(df)*100:.1f}%)")
        
        return df
        
    def _match_retailer_geo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Match retailer and geographic mentions"""
        print("  🏪 Matching retailers & geography...")
        
        # Retailer matching
        retailer_pattern = '|'.join(self.retailers)
        df['retailer_mentions'] = df['comment_text'].str.extractall(f'({retailer_pattern})', flags=re.IGNORECASE)[0].groupby(level=0).apply(list)
        df['retailer'] = df['retailer_mentions'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        
        # Geographic matching
        geo_pattern = '|'.join(self.geo_regions)
        df['geo_mentions'] = df['comment_text'].str.extractall(f'({geo_pattern})', flags=re.IGNORECASE)[0].groupby(level=0).apply(list)
        df['geo_region'] = df['geo_mentions'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        
        retailer_count = df['retailer'].notna().sum()
        geo_count = df['geo_region'].notna().sum()
        print(f"     • Retailer mentions: {retailer_count:,}")
        print(f"     • Geographic mentions: {geo_count:,}")
        
        return df
        
    def _resolve_products_scents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Product and scent resolution with fuzzy matching"""
        print("  🧴 Resolving products & scents...")
        
        # Product matching
        all_products = []
        for category, products in self.products.items():
            all_products.extend(products)
        
        product_pattern = '|'.join(all_products)
        df['product_mentions'] = df['comment_text'].str.extractall(f'({product_pattern})', flags=re.IGNORECASE)[0].groupby(level=0).apply(list)
        
        # Scent matching
        scent_pattern = '|'.join(self.scents)
        df['scent_mentions'] = df['comment_text'].str.extractall(f'({scent_pattern})', flags=re.IGNORECASE)[0].groupby(level=0).apply(list)
        
        product_count = df['product_mentions'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        scent_count = df['scent_mentions'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        
        print(f"     • Product mentions: {product_count:,}")
        print(f"     • Scent mentions: {scent_count:,}")
        
        return df
        
    def _classify_intent(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intent classification using weak supervision"""
        print("  🎯 Classifying intent...")
        
        def classify_intent(text):
            text_lower = str(text).lower()
            
            # Question patterns
            question_patterns = [
                r'\?', r'what\'?s the difference', r'where are these sold', 
                r'can you use on face', r'how do', r'what is', r'where can i'
            ]
            if any(re.search(pattern, text_lower) for pattern in question_patterns):
                return 'QUESTION'
            
            # Request patterns
            request_patterns = [
                r'bring back', r'please carry', r'restock', r'please make',
                r'need more', r'please add', r'can you make'
            ]
            if any(re.search(pattern, text_lower) for pattern in request_patterns):
                return 'REQUEST'
            
            # Purchase intent patterns
            purchase_patterns = [
                r'i need to try', r'i\'?m getting this', r'bought mine',
                r'ordering', r'purchasing', r'buying', r'need this'
            ]
            if any(re.search(pattern, text_lower) for pattern in purchase_patterns):
                return 'PURCHASE_INTENT'
            
            # Praise patterns
            praise_patterns = [
                r'love it', r'obsessed', r'amazing', r'perfect', r'best',
                r'so good', r'incredible', r'favorite'
            ]
            if any(re.search(pattern, text_lower) for pattern in praise_patterns):
                return 'PRAISE'
            
            # Complaint patterns
            complaint_patterns = [
                r'trash scent', r'bad', r'pisses me off', r'hate',
                r'terrible', r'awful', r'disappointing', r'overpriced'
            ]
            if any(re.search(pattern, text_lower) for pattern in complaint_patterns):
                return 'COMPLAINT'
            
            return 'GENERAL'
        
        df['intent'] = df['comment_text'].apply(classify_intent)
        
        intent_dist = df['intent'].value_counts()
        print(f"     Intent distribution:")
        for intent, count in intent_dist.items():
            print(f"       • {intent}: {count:,} ({count/len(df)*100:.1f}%)")
        
        return df
        
    def _analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """VADER sentiment analysis"""
        print("  😊 Analyzing sentiment...")
        
        def get_sentiment(text):
            if pd.isna(text) or str(text).strip() == "":
                return 0.0, 'neutral'
            
            scores = self.sentiment_analyzer.polarity_scores(str(text))
            compound = scores['compound']
            
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
                
            return compound, label
        
        sentiment_results = df['comment_text'].apply(get_sentiment)
        df['sentiment_score'] = sentiment_results.apply(lambda x: x[0])
        df['sentiment_label'] = sentiment_results.apply(lambda x: x[1])
        
        sentiment_dist = df['sentiment_label'].value_counts()
        avg_sentiment = df['sentiment_score'].mean()
        
        print(f"     • Average sentiment: {avg_sentiment:.3f}")
        print(f"     • Positive: {sentiment_dist.get('positive', 0):,} ({sentiment_dist.get('positive', 0)/len(df)*100:.1f}%)")
        print(f"     • Negative: {sentiment_dist.get('negative', 0):,} ({sentiment_dist.get('negative', 0)/len(df)*100:.1f}%)")
        print(f"     • Neutral: {sentiment_dist.get('neutral', 0):,} ({sentiment_dist.get('neutral', 0)/len(df)*100:.1f}%)")
        
        return df
        
    def _calculate_engagement_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate engagement proxy metrics"""
        print("  📊 Calculating engagement proxies...")
        
        # Tag rate (has @)
        df['has_mention'] = df['comment_text'].str.contains('@', na=False)
        
        # Emoji count
        def count_emojis(text):
            import emoji
            try:
                return len([char for char in str(text) if emoji.is_emoji(char)])
            except:
                return 0
        
        df['emoji_count'] = df['comment_text'].apply(count_emojis)
        
        # Question mark count
        df['question_mark_count'] = df['comment_text'].str.count('\?')
        
        tag_rate = df['has_mention'].mean() * 100
        avg_emojis = df['emoji_count'].mean()
        avg_questions = df['question_mark_count'].mean()
        
        print(f"     • Tag rate: {tag_rate:.1f}%")
        print(f"     • Avg emojis per comment: {avg_emojis:.2f}")
        print(f"     • Avg question marks: {avg_questions:.2f}")
        
        return df
        
    def _add_semantic_topics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional semantic topic modeling (Tier-2)"""
        print("  🧠 Adding semantic topics (experimental)...")
        
        # This would require sentence-transformers, UMAP, HDBSCAN
        # For now, create placeholder topic assignments
        print("     • Semantic topics disabled (install sentence-transformers for full functionality)")
        df['semantic_topic'] = 'topic_extraction_disabled'
        
        return df
        
    def _store_enriched_data(self, df: pd.DataFrame):
        """Store enriched data in SQLite with proper indexing"""
        print("\n💾 STORING ENRICHED DATA")
        print("-" * 40)
        
        # Convert lists to JSON strings for SQLite storage
        list_columns = ['product_mentions', 'scent_mentions', 'retailer_mentions', 'geo_mentions']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else None)
        
        # Create SQLite connection
        conn = sqlite3.connect(self.output_db)
        
        # Store enriched data
        df.to_sql('comments_enriched', conn, if_exists='replace', index=False)
        
        # Create indices for performance
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_date ON comments_enriched(date)",
            "CREATE INDEX IF NOT EXISTS idx_week ON comments_enriched(week_of_month)",
            "CREATE INDEX IF NOT EXISTS idx_media_id ON comments_enriched(media_id)",
            "CREATE INDEX IF NOT EXISTS idx_intent ON comments_enriched(intent)",
            "CREATE INDEX IF NOT EXISTS idx_sentiment ON comments_enriched(sentiment_label)",
            "CREATE INDEX IF NOT EXISTS idx_language ON comments_enriched(language)",
            "CREATE INDEX IF NOT EXISTS idx_spam ON comments_enriched(is_spam)",
            "CREATE INDEX IF NOT EXISTS idx_retailer ON comments_enriched(retailer)"
        ]
        
        for idx in indices:
            conn.execute(idx)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Stored {len(df):,} enriched comments in SQLite")
        print(f"📊 Database: {self.output_db}")
        print(f"🔍 Created {len(indices)} performance indices")
        
    def _generate_pipeline_summary(self, df: pd.DataFrame, eda_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary"""
        
        summary = {
            'pipeline_info': {
                'total_comments_processed': len(df),
                'total_columns': len(df.columns),
                'processing_timestamp': datetime.now().isoformat(),
                'database_file': self.output_db
            },
            'data_quality': {
                'language_distribution': df['language'].value_counts().to_dict(),
                'spam_rate': df['is_spam'].mean() * 100,
                'bot_rate': df['is_bot'].mean() * 100,
                'avg_comment_length': df['comment_length'].mean()
            },
            'enrichment_results': {
                'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
                'intent_distribution': df['intent'].value_counts().to_dict(),
                'retailer_mentions': df['retailer'].value_counts().head(10).to_dict(),
                'geo_mentions': df['geo_region'].value_counts().head(10).to_dict(),
                'product_mention_rate': df['product_mentions'].notna().mean() * 100,
                'scent_mention_rate': df['scent_mentions'].notna().mean() * 100
            },
            'engagement_metrics': {
                'tag_rate': df['has_mention'].mean() * 100,
                'avg_emoji_count': df['emoji_count'].mean(),
                'question_rate': (df['question_mark_count'] > 0).mean() * 100
            },
            'eda_insights': eda_results
        }
        
        return summary

def main():
    """Run the comprehensive enrichment pipeline"""
    try:
        # Initialize pipeline
        pipeline = EnrichmentPipeline(
            csv_path='instagram_analysis/data/engagements.csv',
            output_db='reports/enriched_instagram_data.sqlite',
            enable_semantic=False  # Set to True when ML dependencies are available
        )
        
        # Run pipeline
        summary = pipeline.run_pipeline()
        
        # Save summary report
        summary_path = 'reports/enrichment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📋 Pipeline Summary:")
        print(f"   • Total Comments: {summary['pipeline_info']['total_comments_processed']:,}")
        print(f"   • Enrichment Columns: {summary['pipeline_info']['total_columns']}")
        print(f"   • Sentiment Score: {summary['enrichment_results']['sentiment_distribution']}")
        print(f"   • Intent Distribution: {summary['enrichment_results']['intent_distribution']}")
        print(f"   • Summary Report: {summary_path}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 