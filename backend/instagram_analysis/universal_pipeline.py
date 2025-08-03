#!/usr/bin/env python3
"""
Universal Pipeline Design for DMM
Data-agnostic processing system that can handle any social media data format
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging

# Import our specialized analyzers
from enhanced_sentiment import analyze_comment_enhanced
from safety_detection import analyze_comment_safety_comprehensive
from product_intelligence import ProductPerformanceAnalyzer
from model_validation import ModelPerformanceTracker
from aspect import aspect_extract_rule
from config import INTENT_LABELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UniversalComment:
    """Universal comment format that any data source can be mapped to"""
    # Core required fields
    text: str
    comment_id: str
    created_at: str
    
    # Optional social media fields
    username: Optional[str] = None
    post_id: Optional[str] = None
    parent_id: Optional[str] = None
    like_count: Optional[int] = 0
    reply_count: Optional[int] = 0
    
    # Platform-specific metadata
    platform: Optional[str] = "unknown"
    source_metadata: Optional[Dict] = None
    
    # Processing results (populated by pipeline)
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    intent: Optional[str] = None
    priority: Optional[float] = None

class DataAdapter:
    """Converts various data formats to UniversalComment format"""
    
    @staticmethod
    def from_csv(df: pd.DataFrame, mapping: Dict[str, str] = None) -> List[UniversalComment]:
        """Convert CSV data to universal format"""
        # Default mapping for our current CSV format
        default_mapping = {
            'text': 'text',
            'comment_id': 'comment_id', 
            'created_at': 'created_at',
            'username': 'username',
            'post_id': 'post_id',
            'parent_id': 'parent_id',
            'like_count': 'like_count',
            'reply_count': 'reply_count'
        }
        
        if mapping:
            default_mapping.update(mapping)
        
        comments = []
        for _, row in df.iterrows():
            try:
                comment = UniversalComment(
                    text=str(row.get(default_mapping['text'], '')),
                    comment_id=str(row.get(default_mapping['comment_id'], f"csv_{len(comments)}")),
                    created_at=str(row.get(default_mapping['created_at'], datetime.now().isoformat())),
                    username=str(row.get(default_mapping.get('username'), 'anonymous')),
                    post_id=str(row.get(default_mapping.get('post_id'), '')),
                    parent_id=str(row.get(default_mapping.get('parent_id'), '')),
                    like_count=int(row.get(default_mapping.get('like_count'), 0) or 0),
                    reply_count=int(row.get(default_mapping.get('reply_count'), 0) or 0),
                    platform="csv_import",
                    source_metadata={k: v for k, v in row.to_dict().items() if k not in default_mapping.values()}
                )
                comments.append(comment)
            except Exception as e:
                logger.warning(f"Failed to convert row to UniversalComment: {e}")
                continue
                
        logger.info(f"✅ Converted {len(comments)} comments from CSV")
        return comments
    
    @staticmethod
    def from_instagram_api(data: List[Dict]) -> List[UniversalComment]:
        """Convert Instagram API data to universal format"""
        comments = []
        for item in data:
            try:
                comment = UniversalComment(
                    text=item.get('text', ''),
                    comment_id=item.get('id', f"ig_{len(comments)}"),
                    created_at=item.get('timestamp', datetime.now().isoformat()),
                    username=item.get('username', 'anonymous'),
                    post_id=item.get('media_id', ''),
                    like_count=item.get('like_count', 0),
                    reply_count=len(item.get('replies', [])),
                    platform="instagram",
                    source_metadata={
                        'user_id': item.get('user_id'),
                        'media_type': item.get('media_type'),
                        'hashtags': item.get('hashtags', []),
                        'mentions': item.get('mentions', [])
                    }
                )
                comments.append(comment)
            except Exception as e:
                logger.warning(f"Failed to convert Instagram data: {e}")
                continue
                
        logger.info(f"✅ Converted {len(comments)} comments from Instagram API")
        return comments
    
    @staticmethod
    def from_twitter_api(data: List[Dict]) -> List[UniversalComment]:
        """Convert Twitter API data to universal format"""
        comments = []
        for tweet in data:
            try:
                comment = UniversalComment(
                    text=tweet.get('text', ''),
                    comment_id=tweet.get('id', f"tw_{len(comments)}"),
                    created_at=tweet.get('created_at', datetime.now().isoformat()),
                    username=tweet.get('author_username', 'anonymous'),
                    like_count=tweet.get('public_metrics', {}).get('like_count', 0),
                    reply_count=tweet.get('public_metrics', {}).get('reply_count', 0),
                    platform="twitter",
                    source_metadata={
                        'author_id': tweet.get('author_id'),
                        'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                        'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0),
                        'entities': tweet.get('entities', {})
                    }
                )
                comments.append(comment)
            except Exception as e:
                logger.warning(f"Failed to convert Twitter data: {e}")
                continue
                
        logger.info(f"✅ Converted {len(comments)} comments from Twitter API")
        return comments

class MultiModalBERTopic:
    """Enhanced BERTopic with multi-modal analysis layers"""
    
    def __init__(self):
        self.layers = {
            'emoji': self._analyze_emoji_patterns,
            'user': self._analyze_user_patterns,
            'product': self._analyze_product_patterns,
            'temporal': self._analyze_temporal_patterns,
            'safety': self._analyze_safety_patterns,
            'business': self._analyze_business_patterns
        }
        
    def analyze_topics(self, comments: List[UniversalComment], layers: List[str] = None) -> Dict[str, Any]:
        """Run multi-modal topic analysis"""
        if layers is None:
            layers = list(self.layers.keys())
            
        results = {
            'total_comments': len(comments),
            'analysis_layers': layers,
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract text for basic topic modeling
        texts = [c.text for c in comments]
        
        # Run each analysis layer
        for layer in layers:
            if layer in self.layers:
                try:
                    layer_result = self.layers[layer](comments, texts)
                    results[f'{layer}_analysis'] = layer_result
                    logger.info(f"✅ Completed {layer} analysis")
                except Exception as e:
                    logger.error(f"❌ Failed {layer} analysis: {e}")
                    results[f'{layer}_analysis'] = {'error': str(e)}
        
        return results
    
    def _analyze_emoji_patterns(self, comments: List[UniversalComment], texts: List[str]) -> Dict:
        """Analyze emoji usage patterns"""
        import emoji
        emoji_patterns = {}
        emoji_sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for comment in comments:
            text_emojis = [c for c in comment.text if emoji.is_emoji(c)]
            for em in text_emojis:
                emoji_patterns[em] = emoji_patterns.get(em, 0) + 1
                
            # Correlate with sentiment if available
            if comment.sentiment_label:
                emoji_sentiment[comment.sentiment_label] += len(text_emojis)
        
        return {
            'top_emojis': dict(sorted(emoji_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            'emoji_sentiment_correlation': emoji_sentiment,
            'total_emoji_usage': sum(emoji_patterns.values())
        }
    
    def _analyze_user_patterns(self, comments: List[UniversalComment], texts: List[str]) -> Dict:
        """Analyze user behavior patterns"""
        user_stats = {}
        
        for comment in comments:
            username = comment.username or 'anonymous'
            if username not in user_stats:
                user_stats[username] = {
                    'comment_count': 0,
                    'total_likes': 0,
                    'sentiments': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'avg_text_length': 0,
                    'text_lengths': []
                }
            
            stats = user_stats[username]
            stats['comment_count'] += 1
            stats['total_likes'] += comment.like_count or 0
            stats['text_lengths'].append(len(comment.text))
            
            if comment.sentiment_label:
                stats['sentiments'][comment.sentiment_label] += 1
        
        # Calculate averages
        for username, stats in user_stats.items():
            if stats['text_lengths']:
                stats['avg_text_length'] = sum(stats['text_lengths']) / len(stats['text_lengths'])
        
        # Top users by activity
        top_users = sorted(user_stats.items(), key=lambda x: x[1]['comment_count'], reverse=True)[:10]
        
        return {
            'total_unique_users': len(user_stats),
            'top_active_users': [(user, stats['comment_count']) for user, stats in top_users],
            'user_engagement_avg': sum(stats['total_likes'] for stats in user_stats.values()) / len(user_stats) if user_stats else 0
        }
    
    def _analyze_product_patterns(self, comments: List[UniversalComment], texts: List[str]) -> Dict:
        """Analyze product mention patterns"""
        # Use our existing product intelligence
        analyzer = ProductPerformanceAnalyzer()
        
        # Convert to DataFrame for existing analyzer
        df_data = []
        for comment in comments:
            df_data.append({
                'text': comment.text,
                'sentiment_label': comment.sentiment_label or 'neutral',
                'sentiment_score': comment.sentiment_score or 0.0,
                'created_at': comment.created_at
            })
        
        df = pd.DataFrame(df_data)
        
        try:
            product_analysis = analyzer.analyze_product_health(df, days=30)
            return {
                'products_mentioned': len(product_analysis),
                'top_products': list(product_analysis.keys())[:10] if product_analysis else [],
                'analysis_method': 'product_intelligence_integration'
            }
        except Exception as e:
            return {'error': str(e), 'fallback': 'basic_keyword_extraction'}
    
    def _analyze_temporal_patterns(self, comments: List[UniversalComment], texts: List[str]) -> Dict:
        """Analyze temporal patterns"""
        temporal_data = {'hourly': {}, 'daily': {}, 'weekly': {}}
        
        for comment in comments:
            try:
                dt = pd.to_datetime(comment.created_at)
                
                # Hourly patterns
                hour = dt.hour
                temporal_data['hourly'][hour] = temporal_data['hourly'].get(hour, 0) + 1
                
                # Daily patterns
                day = dt.strftime('%A')
                temporal_data['daily'][day] = temporal_data['daily'].get(day, 0) + 1
                
                # Weekly patterns
                week = dt.isocalendar()[1]
                temporal_data['weekly'][week] = temporal_data['weekly'].get(week, 0) + 1
                
            except Exception:
                continue
        
        return {
            'peak_hour': max(temporal_data['hourly'].items(), key=lambda x: x[1]) if temporal_data['hourly'] else None,
            'peak_day': max(temporal_data['daily'].items(), key=lambda x: x[1]) if temporal_data['daily'] else None,
            'temporal_distribution': temporal_data
        }
    
    def _analyze_safety_patterns(self, comments: List[UniversalComment], texts: List[str]) -> Dict:
        """Analyze safety and risk patterns"""
        safety_stats = {
            'total_safety_issues': 0,
            'severity_breakdown': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'common_safety_terms': {},
            'escalation_needed': 0
        }
        
        for comment in comments:
            # This would be populated if safety analysis was run
            if hasattr(comment, 'safety_detected') and comment.safety_detected:
                safety_stats['total_safety_issues'] += 1
                
                if hasattr(comment, 'safety_severity'):
                    severity = getattr(comment, 'safety_severity', 'low')
                    safety_stats['severity_breakdown'][severity] += 1
                    
                if hasattr(comment, 'escalation_level') and getattr(comment, 'escalation_level') != 'none':
                    safety_stats['escalation_needed'] += 1
        
        return safety_stats
    
    def _analyze_business_patterns(self, comments: List[UniversalComment], texts: List[str]) -> Dict:
        """Analyze business intelligence patterns"""
        business_insights = {
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'intent_distribution': {intent: 0 for intent in INTENT_LABELS},
            'high_priority_comments': 0,
            'engagement_metrics': {
                'total_likes': 0,
                'total_replies': 0,
                'avg_engagement': 0
            }
        }
        
        total_engagement = 0
        
        for comment in comments:
            # Sentiment distribution
            if comment.sentiment_label:
                business_insights['sentiment_distribution'][comment.sentiment_label] += 1
            
            # Intent distribution
            if comment.intent and comment.intent in business_insights['intent_distribution']:
                business_insights['intent_distribution'][comment.intent] += 1
            
            # Priority analysis
            if comment.priority and comment.priority > 5.0:  # High priority threshold
                business_insights['high_priority_comments'] += 1
            
            # Engagement metrics
            likes = comment.like_count or 0
            replies = comment.reply_count or 0
            business_insights['engagement_metrics']['total_likes'] += likes
            business_insights['engagement_metrics']['total_replies'] += replies
            total_engagement += likes + replies
        
        # Calculate averages
        if comments:
            business_insights['engagement_metrics']['avg_engagement'] = total_engagement / len(comments)
        
        return business_insights

class UniversalPipeline:
    """Main pipeline orchestrator for data-agnostic processing"""
    
    def __init__(self):
        self.data_adapter = DataAdapter()
        self.topic_analyzer = MultiModalBERTopic()
        self.performance_tracker = ModelPerformanceTracker()
        
    def process_data_source(self, 
                          source_type: str, 
                          data: Union[str, pd.DataFrame, List[Dict]], 
                          mapping: Dict[str, str] = None,
                          processing_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Universal entry point for processing any data source
        
        Args:
            source_type: 'csv', 'instagram_api', 'twitter_api', etc.
            data: The raw data in appropriate format
            mapping: Field mapping for custom data formats
            processing_options: Processing configuration
        """
        
        processing_options = processing_options or {}
        use_enhanced_pipeline = processing_options.get('use_enhanced_pipeline', True)
        analysis_layers = processing_options.get('analysis_layers', ['emoji', 'user', 'temporal', 'business'])
        
        logger.info(f"🚀 Starting Universal Pipeline for {source_type}")
        
        # Stage 1: Data Normalization
        logger.info("📊 Stage 1: Data Normalization")
        if source_type == 'csv':
            if isinstance(data, (str, Path)):
                df = pd.read_csv(str(data))
            else:
                df = data
            comments = self.data_adapter.from_csv(df, mapping)
        elif source_type == 'instagram_api':
            comments = self.data_adapter.from_instagram_api(data)
        elif source_type == 'twitter_api':
            comments = self.data_adapter.from_twitter_api(data)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        if not comments:
            return {'error': 'No comments could be processed from data source'}
        
        # Stage 2: Basic NLP Processing
        logger.info("🧠 Stage 2: Basic NLP Processing")
        processed_comments = []
        
        for comment in comments:
            try:
                if use_enhanced_pipeline:
                    # Enhanced sentiment analysis
                    enhanced_result = analyze_comment_enhanced(comment.text)
                    comment.sentiment_label = enhanced_result['sentiment_label']
                    comment.sentiment_score = enhanced_result['sentiment_score']
                    comment.intent = enhanced_result['intent']
                    
                    # Safety analysis
                    safety_result = analyze_comment_safety_comprehensive(
                        comment_id=comment.comment_id,
                        text=comment.text,
                        sentiment_label=comment.sentiment_label,
                        sentiment_score=comment.sentiment_score
                    )
                    
                    # Add safety attributes to comment
                    comment.safety_detected = bool(safety_result)
                    if safety_result:
                        # Handle both object and dict returns from safety analysis
                        if hasattr(safety_result, 'severity'):
                            comment.safety_severity = safety_result.severity
                            comment.escalation_level = safety_result.escalation_level
                        elif isinstance(safety_result, dict):
                            comment.safety_severity = safety_result.get('severity', 'none')
                            comment.escalation_level = safety_result.get('escalation_level', 'none')
                else:
                    # Basic processing fallback
                    from transformers import pipeline
                    from config import SENTIMENT_MODEL, INTENT_MODEL
                    
                    sent_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
                    intent_pipe = pipeline("zero-shot-classification", model=INTENT_MODEL, multi_label=False)
                    
                    sent_result = sent_pipe(comment.text)[0]
                    intent_result = intent_pipe(comment.text, candidate_labels=INTENT_LABELS)
                    
                    comment.sentiment_label = sent_result['label'].lower()
                    comment.sentiment_score = sent_result['score']
                    comment.intent = intent_result['labels'][0]
                
                # Calculate priority score
                comment.priority = self._calculate_priority(comment)
                
                processed_comments.append(comment)
                
            except Exception as e:
                logger.warning(f"Failed to process comment {comment.comment_id}: {e}")
                continue
        
        logger.info(f"✅ Processed {len(processed_comments)}/{len(comments)} comments")
        
        # Stage 3: Multi-Modal Analysis
        logger.info("🔍 Stage 3: Multi-Modal Analysis")
        
        # Use the comprehensive Multi-Modal BERTopic instead of basic analysis
        try:
            from multi_modal_bertopic import analyze_comments_multimodal
            
            # Convert processed comments back to DataFrame for Multi-Modal BERTopic
            df_data = []
            for comment in processed_comments:
                df_data.append({
                    'text': comment.text,
                    'comment_id': comment.comment_id,
                    'username': comment.username,
                    'created_at': comment.created_at,
                    'like_count': comment.like_count,
                    'reply_count': comment.reply_count,
                    'sentiment_label': comment.sentiment_label,
                    'sentiment_score': comment.sentiment_score,
                    'intent': comment.intent
                })
            
            df = pd.DataFrame(df_data)
            
            # Run comprehensive Multi-Modal BERTopic analysis
            multimodal_results = analyze_comments_multimodal(
                df,
                n_topics=min(10, len(processed_comments) // 5),  # Dynamic topic count
                min_topic_size=max(3, len(processed_comments) // 20)  # Dynamic min size
            )
            
            # Use the comprehensive results
            topic_analysis = {
                'analysis_type': 'Multi-Modal BERTopic (6-Layer)',
                'core_topics': multimodal_results['core_topics'],
                'enhanced_topics': multimodal_results['enhanced_topics'],
                'modal_insights': multimodal_results['modal_insights'],
                'business_insights': multimodal_results['business_insights']
            }
            
            logger.info(f"✅ Multi-Modal BERTopic: {multimodal_results['model_info']['unique_topics']} topics discovered")
            
        except Exception as e:
            logger.warning(f"Multi-Modal BERTopic failed, using basic analysis: {e}")
            # Fallback to basic topic analysis
            topic_analysis = self.topic_analyzer.analyze_topics(processed_comments, analysis_layers)
        
        # Stage 4: Quality Assurance
        logger.info("🛡️ Stage 4: Quality Assurance")
        qa_results = self._run_quality_assurance(processed_comments)
        
        # Stage 5: Results Compilation
        logger.info("📋 Stage 5: Results Compilation")
        results = {
            'pipeline_info': {
                'source_type': source_type,
                'processing_mode': 'enhanced' if use_enhanced_pipeline else 'basic',
                'total_input_comments': len(comments),
                'successfully_processed': len(processed_comments),
                'processing_timestamp': datetime.now().isoformat(),
                'analysis_layers': analysis_layers
            },
            'processed_comments': [self._comment_to_dict(c) for c in processed_comments],
            'topic_analysis': topic_analysis,
            'quality_assurance': qa_results,
            'summary_stats': self._generate_summary_stats(processed_comments)
        }
        
        logger.info("🎉 Universal Pipeline processing complete!")
        return results
    
    def _calculate_priority(self, comment: UniversalComment) -> float:
        """Calculate priority score for a comment"""
        try:
            created = pd.to_datetime(comment.created_at)
            now = datetime.now()
            recency_days = max(1, (now - created).days)
            rec = 1.0 / recency_days
            
            neg = 1 if comment.sentiment_label == 'negative' else 0
            engagement = np.log1p((comment.like_count or 0) + (comment.reply_count or 0))
            
            # Safety multiplier
            safety_multiplier = 3.0 if getattr(comment, 'safety_detected', False) else 1.0
            
            return round((neg * 3 * rec * safety_multiplier) + engagement, 3)
        except Exception:
            return 1.0  # Default priority
    
    def _run_quality_assurance(self, comments: List[UniversalComment]) -> Dict[str, Any]:
        """Run quality assurance checks on processed comments"""
        qa_results = {
            'total_comments_checked': len(comments),
            'consistency_issues': 0,
            'missing_sentiment': 0,
            'missing_intent': 0,
            'data_quality_score': 0.0,
            'recommendations': []
        }
        
        for comment in comments:
            # Check for missing core analysis
            if not comment.sentiment_label:
                qa_results['missing_sentiment'] += 1
            if not comment.intent:
                qa_results['missing_intent'] += 1
            
            # Check sentiment-intent consistency
            if comment.sentiment_label and comment.intent:
                consistency_rules = {
                    'positive': ['praise', 'ugc', 'other'],
                    'negative': ['complaint', 'spam'],
                    'neutral': ['question', 'suggestion', 'other', 'ugc']
                }
                
                if comment.intent not in consistency_rules.get(comment.sentiment_label, []):
                    qa_results['consistency_issues'] += 1
        
        # Calculate quality score
        total_issues = qa_results['consistency_issues'] + qa_results['missing_sentiment'] + qa_results['missing_intent']
        qa_results['data_quality_score'] = max(0, 100 - (total_issues / len(comments) * 100)) if comments else 0
        
        # Generate recommendations
        if qa_results['consistency_issues'] > len(comments) * 0.1:
            qa_results['recommendations'].append("High consistency issues detected - consider model retraining")
        if qa_results['missing_sentiment'] > 0:
            qa_results['recommendations'].append("Some comments missing sentiment analysis - check processing pipeline")
        if qa_results['data_quality_score'] < 80:
            qa_results['recommendations'].append("Overall data quality below 80% - review processing configuration")
        
        return qa_results
    
    def _comment_to_dict(self, comment: UniversalComment) -> Dict[str, Any]:
        """Convert UniversalComment to dictionary for JSON serialization"""
        result = {
            'comment_id': comment.comment_id,
            'text': comment.text,
            'created_at': comment.created_at,
            'username': comment.username,
            'platform': comment.platform,
            'sentiment_label': comment.sentiment_label,
            'sentiment_score': float(comment.sentiment_score) if comment.sentiment_score else None,
            'intent': comment.intent,
            'priority': float(comment.priority) if comment.priority else None,
            'like_count': comment.like_count,
            'reply_count': comment.reply_count
        }
        
        # Add safety attributes if present
        if hasattr(comment, 'safety_detected'):
            result['safety_detected'] = comment.safety_detected
        if hasattr(comment, 'safety_severity'):
            result['safety_severity'] = comment.safety_severity
        if hasattr(comment, 'escalation_level'):
            result['escalation_level'] = comment.escalation_level
        
        return result
    
    def _generate_summary_stats(self, comments: List[UniversalComment]) -> Dict[str, Any]:
        """Generate summary statistics for processed comments"""
        if not comments:
            return {}
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        intent_counts = {intent: 0 for intent in INTENT_LABELS}
        safety_count = 0
        total_engagement = 0
        
        for comment in comments:
            if comment.sentiment_label:
                sentiment_counts[comment.sentiment_label] += 1
            if comment.intent and comment.intent in intent_counts:
                intent_counts[comment.intent] += 1
            if getattr(comment, 'safety_detected', False):
                safety_count += 1
            
            total_engagement += (comment.like_count or 0) + (comment.reply_count or 0)
        
        return {
            'sentiment_distribution': sentiment_counts,
            'intent_distribution': intent_counts,
            'safety_issues_detected': safety_count,
            'total_engagement': total_engagement,
            'avg_engagement': total_engagement / len(comments),
            'high_priority_comments': len([c for c in comments if c.priority and c.priority > 5.0])
        }

# Convenience functions for common use cases
def process_csv_file(csv_path: str, **kwargs) -> Dict[str, Any]:
    """Quick function to process a CSV file"""
    pipeline = UniversalPipeline()
    return pipeline.process_data_source('csv', csv_path, **kwargs)

def process_instagram_data(data: List[Dict], **kwargs) -> Dict[str, Any]:
    """Quick function to process Instagram API data"""
    pipeline = UniversalPipeline()
    return pipeline.process_data_source('instagram_api', data, **kwargs)

def process_twitter_data(data: List[Dict], **kwargs) -> Dict[str, Any]:
    """Quick function to process Twitter API data"""
    pipeline = UniversalPipeline()
    return pipeline.process_data_source('twitter_api', data, **kwargs)

if __name__ == "__main__":
    # Test the universal pipeline
    print("🧪 Testing Universal Pipeline...")
    
    # Test with our existing CSV data
    from config import COMMENTS_CSV
    
    try:
        results = process_csv_file(
            COMMENTS_CSV, 
            processing_options={
                'use_enhanced_pipeline': True,
                'analysis_layers': ['emoji', 'temporal', 'business']
            }
        )
        
        print(f"✅ Pipeline test successful!")
        print(f"📊 Processed: {results['pipeline_info']['successfully_processed']} comments")
        print(f"🎯 Quality Score: {results['quality_assurance']['data_quality_score']:.1f}%")
        print(f"🔍 Analysis Layers: {results['pipeline_info']['analysis_layers']}")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}") 