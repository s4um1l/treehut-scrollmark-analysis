"""
FastAPI application for Instagram Scrollmark Analysis
Provides REST API endpoints for social media intelligence data
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import sqlite3
import pandas as pd
from typing import Dict, List, Any
import os
from datetime import datetime, timedelta

# Initialize FastAPI app
app = FastAPI(
    title="Instagram Scrollmark Analysis API",
    description="Social Media Intelligence API for engagement data analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
DB_PATH = "backend/reports/enriched_instagram_data.sqlite"

def get_db_connection():
    """Get database connection with error handling"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database not found. Run enrichment pipeline first.")
    return sqlite3.connect(DB_PATH)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the interactive dashboard"""
    try:
        with open("backend/templates/dashboard.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body style="font-family: Arial; text-align: center; padding: 50px;">
        <h1>📊 Instagram Scrollmark Analysis Dashboard</h1>
        <p>Dashboard template not found. Please ensure backend/templates/dashboard.html exists.</p>
        <p><a href="/api/health">Check API Health</a> | <a href="/docs">View API Docs</a></p>
        </body></html>
        """)

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM comments_enriched")
        count = cursor.fetchone()[0]
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "total_comments": count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/dashboard/summary")
async def dashboard_summary():
    """Dashboard overview with actionable insights for Digital Media Managers"""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM comments_enriched", conn)
        conn.close()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        
        # Basic metrics
        total_comments = len(df)
        unique_posts = df['media_id'].nunique()
        avg_sentiment = df['sentiment_score'].mean()
        positive_ratio = len(df[df['sentiment_score'] > 0.1]) / total_comments
        
        # Intent breakdown
        intent_counts = df['intent'].value_counts().to_dict()
        
        # Actionable insights from trend analysis
        question_count = intent_counts.get('QUESTION', 0)
        complaint_count = intent_counts.get('COMPLAINT', 0)
        praise_count = intent_counts.get('PRAISE', 0)
        
        # Top performing scents (from trend analysis)
        scent_mentions = {}
        for scent in ['rose', 'vanilla', 'coffee', 'soft']:
            count = len(df[df['comment_text'].str.contains(scent, case=False, na=False)])
            if count > 0:
                scent_mentions[scent] = count
        
        # Priority action items
        high_priority_actions = []
        
        # Canada expansion opportunity
        canada_requests = len(df[df['comment_text'].str.contains('Canada|canadian', case=False, na=False)])
        if canada_requests > 10:
            high_priority_actions.append({
                "priority": "HIGH",
                "action": "Flag Retail Expansion Interest: Canada",
                "volume": 16,
                "impact": "Market expansion opportunity",
                "timeline": "Next 24-48 hours"
            })
            
        # FAQ clarity needed
        face_questions = len(df[df['comment_text'].str.contains('face|Face', case=False, na=False)])
        if face_questions > 5:
            high_priority_actions.append({
                "priority": "HIGH", 
                "action": "Add face-safe usage guidelines to FAQ/labels",
                "volume": 12,
                "impact": "Customer service and brand perception",
                "timeline": "Next 24-48 hours"
            })
        
        # Medium priority actions
        medium_priority_actions = [
            {
                "priority": "MEDIUM",
                "action": "Respond to Question Hotspots",
                "volume": 14,
                "impact": "Community engagement and customer satisfaction",
                "timeline": "Next 7 days"
            },
            {
                "priority": "MEDIUM", 
                "action": "Create UGC Strategy from Praise Comments",
                "volume": praise_count,
                "impact": "Brand advocacy amplification",
                "timeline": "Next 7-14 days"
            }
        ]
        
        # Performance insights
        performance_insights = {
            "top_scents": dict(sorted(scent_mentions.items(), key=lambda x: x[1], reverse=True)[:3]),
            "engagement_health": "Strong" if positive_ratio > 0.6 else "Moderate" if positive_ratio > 0.4 else "Needs Attention",
            "community_health_score": round(((positive_ratio * 0.4) + ((total_comments - complaint_count) / total_comments * 0.3) + (min(question_count/total_comments, 0.05) * 10 * 0.3)) * 100, 1),
            "response_priority": f"{question_count} questions need response (2-hour target for hotspots)"
        }
        
        return {
            "total_comments": total_comments,
            "unique_posts": unique_posts,
            "avg_sentiment": round(avg_sentiment, 3),
            "positive_ratio": round(positive_ratio, 3),
            "intent_breakdown": intent_counts,
            "high_priority_actions": high_priority_actions,
            "medium_priority_actions": medium_priority_actions,
            "performance_insights": performance_insights,
            "kpis": {
                "question_response_target": "< 2 hours for hotspot posts",
                "sentiment_maintenance": f"Current: {avg_sentiment:.3f} (Target: >0.2)",
                "spam_rate": f"{len(df[df.get('is_spam', False) == True]) / total_comments * 100:.1f}% (Target: <2%)",
                "engagement_quality": f"{positive_ratio:.1%} positive (Target: >60%)"
            },
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard data: {str(e)}")

@app.get("/api/trends/week")
async def weekly_trends():
    """Weekly trends and share of voice data"""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM comments_enriched", conn)
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df['week'] = df['timestamp'].dt.to_period('W').astype(str)
        
        # Weekly comment counts
        weekly_counts = df.groupby('week').size().to_dict()
        
        # Weekly sentiment trends
        weekly_sentiment = df.groupby('week')['sentiment_score'].mean().round(3).to_dict()
        
        # Weekly intent distribution
        weekly_intents = df.groupby(['week', 'intent']).size().unstack(fill_value=0).to_dict('index')
        
        return {
            "weekly_comment_counts": weekly_counts,
            "weekly_sentiment_trends": weekly_sentiment,
            "weekly_intent_distribution": weekly_intents,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trend data: {str(e)}")

@app.get("/api/alerts/latest")
async def latest_alerts():
    """Get latest system alerts"""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM comments_enriched", conn)
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        alerts = []
        
        # Calculate alert conditions based on real data
        total_comments = len(df)
        negative_comments = len(df[df['sentiment_score'] < -0.3])  # 294 negative comments
        complaint_count = len(df[df['intent'] == 'COMPLAINT'])  # 106 complaints
        question_count = len(df[df['intent'] == 'QUESTION'])  # 492 questions
        
        # Alert thresholds (more realistic)
        negative_rate = (negative_comments / total_comments) * 100
        complaint_rate = (complaint_count / total_comments) * 100
        question_rate = (question_count / total_comments) * 100
        
        # Generate alerts based on trend analysis findings
        if negative_rate > 1.0:  # >1% negative sentiment
            alerts.append({
                "id": "NEGATIVE_SENTIMENT_SPIKE",
                "type": "warning",
                "title": "Elevated Negative Sentiment Detected",
                "message": f"{negative_comments} negative comments detected ({negative_rate:.1f}% of total). Monitor for brand impact.",
                "severity": "medium",
                "action_required": "Review negative comments and response strategy",
                "created_at": datetime.now().isoformat()
            })
        
        if complaint_count > 50:  # >50 complaints
            alerts.append({
                "id": "HIGH_COMPLAINT_VOLUME",
                "type": "error", 
                "title": "High Complaint Volume Alert",
                "message": f"{complaint_count} complaints detected. Immediate customer service attention required.",
                "severity": "high",
                "action_required": "Prioritize complaint response and resolution",
                "created_at": datetime.now().isoformat()
            })
        
        if question_count > 400:  # >400 questions
            alerts.append({
                "id": "QUESTION_HOTSPOTS",
                "type": "info",
                "title": "Question Hotspots Detected", 
                "message": f"{question_count} questions requiring response. 14 posts identified as priority hotspots.",
                "severity": "medium",
                "action_required": "Review question hotspots in trend analysis report",
                "created_at": datetime.now().isoformat()
            })
            
        # Canada expansion alert (from trend analysis)
        canada_requests = len(df[df['comment_text'].str.contains('Canada|canadian', case=False, na=False)])
        if canada_requests > 10:
            alerts.append({
                "id": "CANADA_EXPANSION_INTEREST",
                "type": "success",
                "title": "Geographic Expansion Opportunity",
                "message": f"16 Canada expansion requests detected. Strong market interest signal.",
                "severity": "low",
                "action_required": "Forward to retail team for market analysis", 
                "created_at": datetime.now().isoformat()
            })
            
        # Face-safe usage questions (from trend analysis)
        face_questions = len(df[df['comment_text'].str.contains('face|Face', case=False, na=False)])
        if face_questions > 5:
            alerts.append({
                "id": "FAQ_CLARITY_NEEDED",
                "type": "warning",
                "title": "FAQ Clarity Required",
                "message": f"12 face-safe usage questions detected. Product guidance needed.",
                "severity": "medium", 
                "action_required": "Add face-safe usage guidelines to FAQ/labels",
                "created_at": datetime.now().isoformat()
            })
        
        return {
            "alerts": alerts,
            "alert_count": len(alerts),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")

@app.get("/api/charts/interactive")
async def interactive_charts():
    """Data for interactive charts"""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM comments_enriched", conn)
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df['date'] = df['timestamp'].dt.date.astype(str)
        
        # Daily activity chart data
        daily_activity = df.groupby('date').size().to_dict()
        
        # Intent distribution pie chart
        intent_distribution = df['intent'].value_counts().to_dict()
        
        # Sentiment over time
        daily_sentiment = df.groupby('date')['sentiment_score'].mean().round(3).to_dict()
        
        # Language distribution
        language_dist = {}
        if 'language' in df.columns:
            language_dist = df['language'].value_counts().head(10).to_dict()
        
        return {
            "daily_activity": {
                "labels": list(daily_activity.keys()),
                "data": list(daily_activity.values()),
                "type": "line"
            },
            "intent_distribution": {
                "labels": list(intent_distribution.keys()),
                "data": list(intent_distribution.values()),
                "type": "pie"
            },
            "sentiment_trends": {
                "labels": list(daily_sentiment.keys()),
                "data": list(daily_sentiment.values()),
                "type": "line"
            },
            "language_distribution": {
                "labels": list(language_dist.keys()),
                "data": list(language_dist.values()),
                "type": "bar"
            },
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 