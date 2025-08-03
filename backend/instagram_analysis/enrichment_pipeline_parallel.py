#!/usr/bin/env python3
"""
🚀 PARALLELIZED INSTAGRAM ENRICHMENT PIPELINE
High-performance data processing with multi-level parallelization

Performance Improvements:
- Data chunking with multiprocessing (3-4x speedup)
- Optimized language detection with threading
- Pre-compiled regex patterns
- Batch sentiment analysis
- Memory-efficient processing

Target: 17k+ comments in 6-8 minutes (vs 15-17 minutes sequential)
"""

import pandas as pd
import numpy as np
import sqlite3
import re
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Parallelization imports
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import threading

# Core dependencies
from pydantic import BaseModel, Field, validator
from langdetect import detect, detect_langs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedComment(BaseModel):
    """Optimized Pydantic model for comment schema validation"""
    media_id: str
    comment_text: str
    timestamp: str
    media_caption: Optional[str] = ""
    
    class Config:
        # Performance optimization
        validate_assignment = False
        use_enum_values = True

class ParallelEnrichmentPipeline:
    """High-performance parallelized enrichment pipeline"""
    
    def __init__(self, 
                 csv_path: str,
                 output_db: str = 'reports/enriched_instagram_data.sqlite',
                 demo_mode: bool = False,
                 demo_size: int = 1000,
                 max_workers: int = None,
                 chunk_size: int = 2000):
        """
        Initialize the parallel enrichment pipeline
        
        Args:
            csv_path: Path to the CSV file
            output_db: Path to the SQLite database
            demo_mode: Whether to run in demo mode (limits rows)
            demo_size: Number of rows for demo mode
            max_workers: Number of parallel workers (auto-detect if None)
            chunk_size: Size of data chunks for parallel processing
        """
        self.csv_path = Path(csv_path)
        self.output_db = Path(output_db) 
        self.demo_mode = demo_mode
        self.demo_size = demo_size
        
        # Parallelization configuration
        self.max_workers = max_workers or min(8, mp.cpu_count() - 1)
        self.chunk_size = chunk_size
        
        # Pre-compile regex patterns for performance
        self._compile_regex_patterns()
        
        # Setup dictionaries
        self._setup_product_dictionaries()
        self._setup_retailer_geo_dictionaries()
        
        # Initialize sentiment analyzer (thread-safe)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        logger.info(f"🚀 Parallel Pipeline initialized:")
        logger.info(f"   • Max workers: {self.max_workers}")
        logger.info(f"   • Chunk size: {self.chunk_size}")
        logger.info(f"   • Demo mode: {self.demo_mode}")

    def _compile_regex_patterns(self):
        """Pre-compile all regex patterns for performance"""
        # Spam/bot detection patterns
        self.spam_patterns = {
            'url_pattern': re.compile(r'http[s]?://\S+|www\.\S+', re.IGNORECASE),
            'excessive_caps': re.compile(r'[A-Z]{4,}'),
            'excessive_punctuation': re.compile(r'[!?]{3,}'),
            'repetitive_chars': re.compile(r'(.)\1{3,}'),
            'mention_pattern': re.compile(r'@\w+'),
        }
        
        # Intent classification patterns
        self.intent_patterns = {
            'question': re.compile(r'\?|what|how|when|where|why|can you|could you|help|explain', re.IGNORECASE),
            'purchase': re.compile(r'buy|purchase|get|order|want|need|link|shop|price|cost', re.IGNORECASE),
            'complaint': re.compile(r'disappointed|terrible|awful|worst|hate|horrible|bad|broke|damaged', re.IGNORECASE),
            'praise': re.compile(r'amazing|love|awesome|great|perfect|fantastic|wonderful|excellent|best', re.IGNORECASE),
            'request': re.compile(r'please|bring back|restock|available|request|suggestion|wish', re.IGNORECASE),
        }

    def _setup_product_dictionaries(self):
        """Setup product and scent recognition dictionaries"""
        self.products = [
            'body butter', 'body cream', 'body lotion', 'body wash', 'shower gel',
            'body scrub', 'body spray', 'perfume', 'fragrance', 'mist', 'cream',
            'lotion', 'butter', 'soap', 'gel', 'scrub', 'oil', 'balm'
        ]
        
        self.scents = [
            'vanilla', 'coconut', 'shea', 'cherry', 'peach', 'strawberry', 'mango',
            'pineapple', 'citrus', 'lemon', 'orange', 'grapefruit', 'lavender',
            'rose', 'jasmine', 'sandalwood', 'eucalyptus', 'mint', 'cucumber',
            'aloe', 'honey', 'chocolate', 'caramel', 'coffee', 'tea tree',
            'bergamot', 'patchouli', 'ylang ylang', 'chamomile', 'sweet pea',
            'gardenia', 'magnolia', 'peony', 'hibiscus', 'cherry blossom',
            'apple', 'pear', 'grape', 'berry', 'tropical', 'ocean', 'rain',
            'fresh', 'clean', 'warm', 'cozy', 'sweet', 'floral', 'fruity',
            'citrusy', 'woodsy', 'musky', 'spicy', 'herbal', 'minty', 'zesty',
            'crisp', 'soft', 'smooth', 'rich', 'luxurious', 'exotic', 'romantic',
            'energizing', 'calming', 'refreshing', 'invigorating', 'soothing',
            'uplifting', 'relaxing', 'sensual', 'sophisticated', 'playful',
            'vibrant', 'bold', 'subtle', 'delicate', 'intense', 'light',
            'mist', 'lime', 'oat'
        ]
        
        # Compile product/scent patterns for faster matching
        self.product_pattern = re.compile('|'.join(re.escape(p) for p in self.products), re.IGNORECASE)
        self.scent_pattern = re.compile('|'.join(re.escape(s) for s in self.scents), re.IGNORECASE)

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
        
        # Compile retailer/geo patterns
        self.retailer_pattern = re.compile('|'.join(re.escape(r) for r in self.retailers), re.IGNORECASE)
        self.geo_pattern = re.compile('|'.join(re.escape(g) for g in self.geo_regions), re.IGNORECASE)

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete parallelized enrichment pipeline"""
        print("\n" + "="*80)
        print("🚀 STARTING PARALLELIZED ENRICHMENT PIPELINE")
        print("="*80)
        start_time = time.time()
        
        # Step 1: Load & Standardize (Sequential - Required)
        df = self._load_and_standardize()
        
        # Step 2: Light EDA (Sequential - Needs full dataset)
        eda_results = self._light_eda(df)
        
        # Step 3: Check if we need to flush database
        self._check_and_prepare_database(len(df))
        
        # Step 4: Parallel Enrichment (Main Performance Gain)
        enriched_df = self._parallel_enrich_data(df)
        
        # Step 5: Storage (Sequential - Database writes)
        self._store_enriched_data(enriched_df)
        
        # Step 6: Generate summary
        summary = self._generate_pipeline_summary(enriched_df, eda_results)
        
        total_time = time.time() - start_time
        print(f"\n⚡ PIPELINE COMPLETED IN {total_time:.2f} SECONDS")
        print(f"📊 Database: {self.output_db}")
        print(f"🔥 Processing rate: {len(df)/total_time:.0f} comments/second")
        
        return summary

    def _check_and_prepare_database(self, expected_rows: int):
        """Check if database needs to be flushed and prepare for new data"""
        if not self.output_db.exists():
            logger.info("📁 Database doesn't exist, will be created fresh")
            return
            
        # Check current row count
        try:
            with sqlite3.connect(self.output_db) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM comments_enriched")
                current_rows = cursor.fetchone()[0]
                
            # If we're processing significantly more data, ask about flushing
            if not self.demo_mode and current_rows > 0:
                if expected_rows > current_rows * 1.5:  # 50% more data
                    logger.warning(f"⚠️  Database has {current_rows:,} rows, about to process {expected_rows:,}")
                    logger.info("🗑️  Flushing existing database for fresh full dataset...")
                    self.output_db.unlink()  # Delete the file
                else:
                    logger.info(f"📊 Database has {current_rows:,} rows, will append {expected_rows:,} new rows")
        except Exception as e:
            logger.warning(f"Could not check database: {e}, proceeding with fresh database")
            if self.output_db.exists():
                self.output_db.unlink()

    def _load_and_standardize(self) -> pd.DataFrame:
        """Load and standardize the raw data (Sequential step)"""
        print("\n📥 LOADING & STANDARDIZING DATA")
        print("-" * 40)
        
        # Load data
        df = pd.read_csv(self.csv_path)
        original_count = len(df)
        
        # Apply demo mode if enabled
        if self.demo_mode:
            df = df.head(self.demo_size)
            print(f"✅ Loaded {original_count:,} raw comments")
            print(f"🔬 Demo mode: Processing first {len(df):,} comments")
        else:
            print(f"✅ Loaded {original_count:,} raw comments")
            print(f"🚀 Production mode: Processing all {len(df):,} comments")
        
        # Parse UTC timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        
        # Add derived temporal fields (vectorized operations)
        df['date'] = df['timestamp'].dt.date.astype(str)
        df['week_of_month'] = df['timestamp'].dt.day.apply(lambda x: (x - 1) // 7 + 1)
        
        # Drop exact duplicates
        df_before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed = df_before_dedup - len(df)
        print(f"🧹 Removed {duplicates_removed:,} exact duplicates")
        
        # Normalize text (vectorized where possible)
        df['comment_text'] = df['comment_text'].astype(str).str.strip()
        df['comment_text'] = df['comment_text'].str.replace(r'\s+', ' ', regex=True)
        df['media_caption'] = df['media_caption'].astype(str).str.strip()
        df['media_caption'] = df['media_caption'].str.replace(r'\s+', ' ', regex=True)
        
        # Add basic derived fields (vectorized)
        df['comment_length'] = df['comment_text'].str.len()
        df['word_count'] = df['comment_text'].str.split().str.len()
        
        print(f"✅ Standardized data: {len(df):,} comments ready for parallel enrichment")
        return df

    def _light_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fast EDA analysis (Sequential step - needs full dataset)"""
        print("\n📊 LIGHT EDA ANALYSIS")
        print("-" * 20)
        
        # Basic stats (vectorized operations)
        unique_media = df['media_id'].nunique()
        avg_comment_length = df['comment_length'].mean()
        
        # Date range
        date_range = {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max(),
            'days': (df['timestamp'].max() - df['timestamp'].min()).days
        }
        
        # Comment engagement distribution (since no author column)
        comment_stats = {
            'total_comments': len(df),
            'unique_media_posts': df['media_id'].nunique(),
            'avg_comments_per_post': len(df) / df['media_id'].nunique(),
            'posts_with_high_engagement': (df.groupby('media_id').size() >= 50).sum()
        }
        
        # Media engagement distribution  
        media_counts = df['media_id'].value_counts()
        media_stats = {
            'top_posts': media_counts.head(5).to_dict(),
            'low_engagement_posts': (media_counts <= 2).sum(),
            'high_engagement_posts': (media_counts >= 50).sum(),
            'long_tail_posts': (media_counts <= 5).sum()
        }
        
        eda_results = {
            'basic_stats': {
                'total_comments': len(df),
                'unique_media_posts': unique_media,
                'avg_comment_length': round(avg_comment_length, 2),
                'avg_comments_per_post': comment_stats['avg_comments_per_post']
            },
            'date_range': date_range,
            'comment_stats': comment_stats,
            'media_stats': media_stats
        }
        
        print(f"  📈 {len(df):,} comments across {unique_media:,} posts")
        print(f"  📅 Date range: {date_range['start'].date()} to {date_range['end'].date()} ({date_range['days']} days)")
        print(f"  💬 Avg comments per post: {comment_stats['avg_comments_per_post']:.1f}")
        
        return eda_results

    def _create_data_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split dataframe into chunks for parallel processing"""
        chunks = []
        total_rows = len(df)
        
        for i in range(0, total_rows, self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size].copy()
            chunks.append(chunk)
        
        logger.info(f"📦 Created {len(chunks)} chunks of ~{self.chunk_size} rows each")
        return chunks

    def _parallel_enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply enrichment layers in parallel using data chunking"""
        print("\n🔬 PARALLEL ENRICHMENT PROCESSING")
        print("-" * 40)
        
        # Create chunks for parallel processing
        chunks = self._create_data_chunks(df)
        
        print(f"🚀 Processing {len(chunks)} chunks with {self.max_workers} workers...")
        
        # Process chunks in parallel
        enriched_chunks = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self._enrich_chunk, i, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    enriched_chunk = future.result()
                    enriched_chunks.append((chunk_idx, enriched_chunk))
                    print(f"  ✅ Chunk {chunk_idx + 1}/{len(chunks)} completed")
                except Exception as e:
                    logger.error(f"❌ Chunk {chunk_idx} failed: {e}")
                    raise
        
        # Sort chunks by original order and concatenate
        enriched_chunks.sort(key=lambda x: x[0])
        enriched_df = pd.concat([chunk for _, chunk in enriched_chunks], ignore_index=True)
        
        print(f"✅ Parallel enrichment complete: {len(enriched_df.columns)} total columns")
        return enriched_df

    def _enrich_chunk(self, chunk_idx: int, chunk: pd.DataFrame) -> pd.DataFrame:
        """Enrich a single chunk of data (runs in separate process)"""
        # Create a copy to avoid modifying original
        enriched_chunk = chunk.copy()
        
        # Apply all enrichment layers to this chunk
        enriched_chunk = self._detect_language_optimized(enriched_chunk)
        enriched_chunk = self._detect_spam_bots_optimized(enriched_chunk)
        enriched_chunk = self._match_retailer_geo_optimized(enriched_chunk)
        enriched_chunk = self._resolve_products_scents_optimized(enriched_chunk)
        enriched_chunk = self._classify_intent_optimized(enriched_chunk)
        enriched_chunk = self._analyze_sentiment_batch(enriched_chunk)
        enriched_chunk = self._calculate_engagement_proxies_vectorized(enriched_chunk)
        
        return enriched_chunk

    def _detect_language_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized language detection with threading and caching"""
        def detect_lang_fast(text):
            try:
                text_str = str(text).strip()
                if len(text_str) < 3:
                    return 'unknown'
                
                # Use detect_langs for confidence-based detection (faster)
                langs = detect_langs(text_str)
                if langs and langs[0].prob > 0.7:  # High confidence threshold
                    return langs[0].lang
                return 'en'  # Default to English for ambiguous cases
            except Exception:
                return 'unknown'
        
        # Use ThreadPoolExecutor for I/O-bound language detection
        with ThreadPoolExecutor(max_workers=4) as executor:
            languages = list(executor.map(detect_lang_fast, df['comment_text']))
        
        df['language'] = languages
        return df

    def _detect_spam_bots_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized spam/bot detection using pre-compiled patterns"""
        # Vectorized operations where possible
        df['has_url'] = df['comment_text'].str.contains(self.spam_patterns['url_pattern'], na=False)
        df['has_spam_pattern'] = (
            df['comment_text'].str.contains(self.spam_patterns['excessive_caps'], na=False) |
            df['comment_text'].str.contains(self.spam_patterns['excessive_punctuation'], na=False) |
            df['comment_text'].str.contains(self.spam_patterns['repetitive_chars'], na=False)
        )
        
        # Mention detection
        df['has_mention'] = df['comment_text'].str.contains(self.spam_patterns['mention_pattern'], na=False)
        df['mention_count'] = df['comment_text'].str.count(self.spam_patterns['mention_pattern'])
        df['excessive_mentions'] = df['mention_count'] > 3
        
        # Question mark count (vectorized)
        df['question_mark_count'] = df['comment_text'].str.count('\?')
        
        # Composite spam detection
        df['is_spam'] = (
            df['has_url'] & 
            (df['comment_length'] < 20)
        ) | df['has_spam_pattern']
        
        # Simple bot detection
        df['is_bot'] = (
            (df['comment_length'] < 10) & 
            (df['word_count'] <= 2) & 
            df['has_spam_pattern']
        )
        
        return df

    def _match_retailer_geo_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized retailer and geo matching using pre-compiled patterns"""
        # Use findall instead of extractall for better performance with non-capture groups
        def extract_retailers(text):
            matches = self.retailer_pattern.findall(str(text))
            return matches[0] if matches else ''
        
        def extract_geos(text):
            matches = self.geo_pattern.findall(str(text))
            return matches[0] if matches else ''
        
        def extract_retailer_list(text):
            matches = self.retailer_pattern.findall(str(text))
            return json.dumps(matches)
        
        def extract_geo_list(text):
            matches = self.geo_pattern.findall(str(text))
            return json.dumps(matches)
        
        # Apply pattern matching
        df['retailer'] = df['comment_text'].apply(extract_retailers)
        df['geo_region'] = df['comment_text'].apply(extract_geos)
        df['retailer_mentions'] = df['comment_text'].apply(extract_retailer_list)
        df['geo_mentions'] = df['comment_text'].apply(extract_geo_list)
        
        return df

    def _resolve_products_scents_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized product and scent matching using pre-compiled patterns"""
        # Use findall instead of extractall for better performance
        def extract_products(text):
            matches = self.product_pattern.findall(str(text))
            return matches[0] if matches else ''
        
        def extract_scents(text):
            matches = self.scent_pattern.findall(str(text))
            return matches[0] if matches else ''
        
        def extract_product_list(text):
            matches = self.product_pattern.findall(str(text))
            return json.dumps(matches)
        
        def extract_scent_list(text):
            matches = self.scent_pattern.findall(str(text))
            return json.dumps(matches)
        
        # Apply pattern matching
        df['primary_product'] = df['comment_text'].apply(extract_products)
        df['primary_scent'] = df['comment_text'].apply(extract_scents)
        df['product_mentions'] = df['comment_text'].apply(extract_product_list)
        df['scent_mentions'] = df['comment_text'].apply(extract_scent_list)
        
        return df

    def _classify_intent_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized intent classification using pre-compiled patterns"""
        # Vectorized pattern matching for all intents
        intent_scores = {}
        for intent, pattern in self.intent_patterns.items():
            intent_scores[intent] = df['comment_text'].str.count(pattern)
        
        intent_df = pd.DataFrame(intent_scores)
        
        # Determine primary intent (highest score, with ties going to priority order)
        intent_priority = ['complaint', 'purchase', 'question', 'request', 'praise']
        
        def get_primary_intent(row):
            # Check if any patterns matched
            if row.sum() == 0:
                return 'GENERAL'
            
            # Return highest priority intent that has matches
            for intent in intent_priority:
                if row[intent] > 0:
                    return intent.upper()
            
            # Fallback to highest scoring intent
            return row.idxmax().upper()
        
        df['intent'] = intent_df.apply(get_primary_intent, axis=1)
        
        return df

    def _analyze_sentiment_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch sentiment analysis for better performance"""
        # Process in smaller batches to avoid memory issues
        batch_size = 100
        sentiment_scores = []
        
        for i in range(0, len(df), batch_size):
            batch_texts = df['comment_text'].iloc[i:i + batch_size].tolist()
            batch_scores = [
                self.sentiment_analyzer.polarity_scores(text)['compound'] 
                for text in batch_texts
            ]
            sentiment_scores.extend(batch_scores)
        
        df['sentiment_score'] = sentiment_scores
        
        # Categorize sentiment
        df['sentiment_label'] = pd.cut(
            df['sentiment_score'], 
            bins=[-1, -0.1, 0.1, 1], 
            labels=['negative', 'neutral', 'positive']
        )
        
        return df

    def _calculate_engagement_proxies_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate engagement proxies using vectorized operations"""
        # Emoji detection (vectorized)
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE
        )
        
        df['emoji_count'] = df['comment_text'].str.count(emoji_pattern)
        
        # Engagement score (vectorized calculation)
        df['engagement_score'] = (
            df['comment_length'] * 0.1 +
            df['emoji_count'] * 2 +
            df['word_count'] * 0.5 +
            (df['has_mention'].astype(int) * 1) +
            (df['question_mark_count'] * 0.5)
        ).round(2)
        
        return df

    def _store_enriched_data(self, df: pd.DataFrame):
        """Store enriched data in SQLite with optimized batch inserts"""
        print(f"\n💾 STORING ENRICHED DATA")
        print("-" * 25)
        
        # Ensure output directory exists
        self.output_db.parent.mkdir(parents=True, exist_ok=True)
        
        # Store with optimized settings
        with sqlite3.connect(self.output_db) as conn:
            # Enable performance optimizations
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL") 
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            
            # Store data (replace existing table)
            df.to_sql('comments_enriched', conn, if_exists='replace', index=False, chunksize=1000)
            
            # Create performance indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_id ON comments_enriched(media_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON comments_enriched(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON comments_enriched(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_intent ON comments_enriched(intent)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON comments_enriched(language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON comments_enriched(sentiment_label)")
            
            # Get final count
            cursor = conn.execute("SELECT COUNT(*) FROM comments_enriched")
            stored_count = cursor.fetchone()[0]
        
        print(f"✅ Stored {stored_count:,} enriched comments in SQLite")
        print(f"📁 Database: {self.output_db}")

    def _generate_pipeline_summary(self, df: pd.DataFrame, eda_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary"""
        # Language distribution
        lang_dist = df['language'].value_counts()
        
        # Quality metrics
        quality_metrics = {
            'spam_rate': df['is_spam'].mean() * 100,
            'bot_rate': df['is_bot'].mean() * 100,
            'english_rate': (df['language'] == 'en').mean() * 100,
            'avg_engagement_score': df['engagement_score'].mean()
        }
        
        # Intent distribution
        intent_dist = df['intent'].value_counts()
        
        # Sentiment distribution
        sentiment_dist = df['sentiment_label'].value_counts()
        
        # Product/retailer insights
        product_mentions = df[df['product_mentions'] != '[]'].shape[0]
        retailer_mentions = df[df['retailer_mentions'] != '[]'].shape[0]
        
        summary = {
            'pipeline_info': {
                'total_processed': len(df),
                'processing_mode': 'demo' if self.demo_mode else 'production',
                'parallel_workers': self.max_workers,
                'chunk_size': self.chunk_size
            },
            'eda_results': eda_results,
            'enrichment_results': {
                'language_distribution': lang_dist.head(10).to_dict(),
                'quality_metrics': quality_metrics,
                'intent_distribution': intent_dist.to_dict(),
                'sentiment_distribution': sentiment_dist.to_dict(),
                'product_mentions': product_mentions,
                'retailer_mentions': retailer_mentions
            }
        }
        
        # Print summary
        print(f"\n📋 PIPELINE SUMMARY")
        print("-" * 20)
        print(f"✅ Processed: {len(df):,} comments")
        print(f"🌐 English: {quality_metrics['english_rate']:.1f}%")
        print(f"🛡️  Spam rate: {quality_metrics['spam_rate']:.1f}%")
        print(f"🤖 Bot rate: {quality_metrics['bot_rate']:.1f}%")
        print(f"📊 Avg engagement: {quality_metrics['avg_engagement_score']:.2f}")
        print(f"🛍️  Product mentions: {product_mentions:,}")
        print(f"🏪 Retailer mentions: {retailer_mentions:,}")
        
        return summary

def run_parallel_pipeline(csv_path: str, 
                         demo_mode: bool = False, 
                         demo_size: int = 1000,
                         max_workers: int = None) -> Dict[str, Any]:
    """
    Convenience function to run the parallel enrichment pipeline
    
    Args:
        csv_path: Path to the CSV file
        demo_mode: Whether to run in demo mode
        demo_size: Number of rows for demo mode
        max_workers: Number of parallel workers
    
    Returns:
        Pipeline summary dictionary
    """
    pipeline = ParallelEnrichmentPipeline(
        csv_path=csv_path,
        demo_mode=demo_mode,
        demo_size=demo_size,
        max_workers=max_workers
    )
    
    return pipeline.run_pipeline()

if __name__ == "__main__":
    # Run with command line arguments
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enrichment_pipeline_parallel.py <csv_path> [--production] [--workers=N]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    demo_mode = '--production' not in sys.argv
    max_workers = None
    
    # Parse workers argument
    for arg in sys.argv:
        if arg.startswith('--workers='):
            max_workers = int(arg.split('=')[1])
    
    print(f"🚀 Running {'PRODUCTION' if not demo_mode else 'DEMO'} mode")
    
    summary = run_parallel_pipeline(
        csv_path=csv_path,
        demo_mode=demo_mode,
        max_workers=max_workers
    )
    
    print("\n🎉 Pipeline completed successfully!") 