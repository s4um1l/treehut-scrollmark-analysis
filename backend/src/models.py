#!/usr/bin/env python3
"""
Pydantic models for Instagram Trend Analysis API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum

class IntentType(str, Enum):
    GENERAL = "GENERAL"
    PRAISE = "PRAISE"
    QUESTION = "QUESTION"
    PURCHASE_INTENT = "PURCHASE_INTENT"
    COMPLAINT = "COMPLAINT"
    REQUEST = "REQUEST"

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class AlertSeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class AlertType(str, Enum):
    NEGATIVE_SPIKE = "NEGATIVE_SPIKE"
    COMPLAINT_CLUSTER = "COMPLAINT_CLUSTER"
    SPAM_BURST = "SPAM_BURST"

# Weekly Trends Models
class WeeklySOVItem(BaseModel):
    week: str
    item_name: str
    mentions: int
    total_comments: int
    sov_percentage: float = Field(..., description="Share of Voice percentage")
    
class WeeklySOVDelta(BaseModel):
    item_name: str
    current_week_sov: float
    previous_week_sov: float
    absolute_change: float
    percentage_change: float
    trend: str = Field(..., description="RISER, FALLER, or STABLE")

class WeeklyTrends(BaseModel):
    week_range: str = Field(..., description="Date range covered")
    products: List[WeeklySOVItem] = []
    scents: List[WeeklySOVItem] = []
    week_over_week_deltas: List[WeeklySOVDelta] = []
    
# Intent Summary Models
class WeeklyIntentCount(BaseModel):
    week: str
    intent: IntentType
    count: int
    percentage: float

class IntentSummary(BaseModel):
    date_range: str
    total_comments: int
    weekly_breakdown: List[WeeklyIntentCount]
    intent_totals: Dict[str, int]
    
# Retailer Models
class RetailerMention(BaseModel):
    week: str
    retailer: str
    mentions: int
    share_percentage: float

class RetailerSummary(BaseModel):
    date_range: str
    total_mentions: int
    weekly_breakdown: List[RetailerMention]
    top_retailers: Dict[str, int]
    
# Media Profile Models
class QuestionSample(BaseModel):
    comment_text: str
    sentiment_score: float
    timestamp: datetime

class MediaProfile(BaseModel):
    media_id: str
    media_caption: str
    total_comments: int
    intent_breakdown: Dict[str, int]
    avg_sentiment: float
    question_rate: float
    sample_questions: List[QuestionSample]
    is_hotspot: bool = Field(..., description="Whether this media is a question hotspot")
    
# Alert Models
class Alert(BaseModel):
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: AlertType
    severity: AlertSeverity
    item: str = Field(..., description="The item (product, scent, media_id) that triggered the alert")
    value: float = Field(..., description="The metric value that triggered the alert")
    threshold: float = Field(..., description="The threshold that was exceeded")
    created_at: datetime
    description: str = Field(..., description="Human-readable alert description")
    action_required: str = Field(..., description="Suggested action for DMM")
    
class AlertsSummary(BaseModel):
    total_alerts: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    latest_alerts: List[Alert]
    
# Dashboard Summary Models
class TopPerformer(BaseModel):
    name: str
    value: float
    change: float
    trend: str

class DashboardCard(BaseModel):
    title: str
    value: Union[int, float, str]
    change: Optional[float] = None
    trend: Optional[str] = None
    description: str
    action_required: bool = False

class DashboardSummary(BaseModel):
    overview_cards: List[DashboardCard]
    top_risers: List[TopPerformer]
    top_fallers: List[TopPerformer]
    urgent_actions: int
    question_hotspots: int
    
# Interactive Chart Data Models
class ChartDataPoint(BaseModel):
    x: Union[str, float, int]
    y: Union[str, float, int]
    label: Optional[str] = None
    color: Optional[str] = None
    
class ChartSeries(BaseModel):
    name: str
    data: List[ChartDataPoint]
    type: str = Field(..., description="line, bar, pie, scatter, etc.")
    
class InteractiveChart(BaseModel):
    chart_id: str
    title: str
    subtitle: Optional[str] = None
    chart_type: str = Field(..., description="Chart.js chart type")
    series: List[ChartSeries]
    x_axis_label: str
    y_axis_label: str
    dmm_insight: str = Field(..., description="Key insight for digital media manager")
    recommended_action: str = Field(..., description="Specific action recommendation")
    
# FAQ Generation Models
class FAQItem(BaseModel):
    question: str
    answer: str
    category: str = Field(..., description="face_safe, usage, ingredients, etc.")
    priority: str = Field(..., description="HIGH, MEDIUM, LOW")
    source_comments: List[str] = Field(..., description="Original customer questions")
    
class FAQDraft(BaseModel):
    week_period: str
    total_questions_analyzed: int
    generated_faqs: List[FAQItem]
    content_suggestions: List[str] = Field(..., description="Additional content recommendations")
    
# Error Response Model
class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime

# Success Response Model
class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now) 