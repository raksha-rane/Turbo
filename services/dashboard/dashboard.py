import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import redis
import psycopg2
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

# Configure page
st.set_page_config(
    page_title="Aircraft Engine Monitoring",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main layout */
    .main > div {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Metrics styling */
    .stMetric {
        background-color: #ffffff;
        border: 2px solid #e1e4e8;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric label {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #24292e !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Alert cards - HIGH VISIBILITY */
    .alert-critical {
        background-color: #ffffff;
        border: 2px solid #d73a49;
        border-left: 6px solid #d73a49;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(215, 58, 73, 0.2);
    }
    
    .alert-critical .alert-title {
        color: #d73a49;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .alert-critical .alert-content {
        color: #24292e;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .alert-critical .alert-meta {
        color: #586069;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .alert-warning {
        background-color: #ffffff;
        border: 2px solid #f9c513;
        border-left: 6px solid #f9c513;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(249, 197, 19, 0.2);
    }
    
    .alert-warning .alert-title {
        color: #b08800;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .alert-warning .alert-content {
        color: #24292e;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .alert-warning .alert-meta {
        color: #586069;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .alert-info {
        background-color: #ffffff;
        border: 2px solid #0366d6;
        border-left: 6px solid #0366d6;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(3, 102, 214, 0.2);
    }
    
    .alert-info .alert-title {
        color: #0366d6;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .alert-info .alert-content {
        color: #24292e;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .alert-info .alert-meta {
        color: #586069;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .status-healthy {
        color: #28a745;
        font-weight: 600;
    }
    
    .status-warning {
        color: #f9c513;
        font-weight: 600;
    }
    
    .status-critical {
        color: #d73a49;
        font-weight: 600;
    }
    
    /* Headers */
    h1 {
        color: #24292e;
        text-align: center;
        padding-bottom: 1.5rem;
        font-weight: 600;
    }
    
    h2, h3 {
        color: #24292e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Spacing improvements */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f6f8fa;
        padding: 1.5rem;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        margin: 1rem 0;
    }
    
    /* Section dividers */
    .section-divider {
        border-top: 2px solid #e1e4e8;
        margin: 2rem 0 1.5rem 0;
    }
    
    .section-container {
        background-color: #f6f8fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e1e4e8;
    }
    
    /* Info tooltips */
    .info-tooltip {
        color: #586069;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    /* System health indicators */
    .health-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .health-online {
        background-color: #28a745;
    }
    
    .health-offline {
        background-color: #d73a49;
    }
    
    .health-degraded {
        background-color: #f9c513;
    }
</style>
""", unsafe_allow_html=True)

class DashboardService:
    def __init__(self):
        # Database configuration
        self.postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'aircraft_monitoring'),
            'user': os.getenv('POSTGRES_USER', 'admin'),
            'password': os.getenv('POSTGRES_PASSWORD', 'secure_password')
        }
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        
        # Initialize connections
        self.redis_client = None
        self._connect_redis()
    
    def _connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
        except Exception as e:
            st.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.postgres_config)
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None
    
    def get_system_health(self) -> Dict:
        """Get system health metrics from Redis"""
        if not self.redis_client:
            return {}
        
        try:
            health_data = self.redis_client.get("system_health")
            if health_data:
                return json.loads(health_data)
        except Exception as e:
            st.error(f"Failed to get system health: {e}")
        
        return {}
    
    def get_engine_list(self) -> List[str]:
        """Get list of active engines"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT engine_id 
                    FROM engines 
                    ORDER BY engine_id
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Failed to get engine list: {e}")
            return []
        finally:
            conn.close()
    
    def get_latest_telemetry(self, engine_id: str) -> Optional[Dict]:
        """Get latest telemetry data for an engine"""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(f"latest_telemetry:{engine_id}")
            if data:
                return json.loads(data)
        except Exception as e:
            st.error(f"Failed to get latest telemetry: {e}")
        
        return None
    
    def get_latest_prediction(self, engine_id: str) -> Optional[Dict]:
        """Get latest prediction for an engine"""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(f"latest_prediction:{engine_id}")
            if data:
                return json.loads(data)
        except Exception as e:
            st.error(f"Failed to get latest prediction: {e}")
        
        return None
    
    def get_telemetry_history(self, engine_id: str, hours: int = 1) -> pd.DataFrame:
        """Get telemetry history for an engine"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT timestamp, cycle, sensor_1, sensor_2, sensor_3, sensor_4, sensor_5,
                       sensor_7, sensor_11, sensor_12, sensor_14, sensor_15
                FROM telemetry 
                WHERE engine_id = %s 
                AND timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            df = pd.read_sql(query, conn, params=(engine_id, hours))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            st.error(f"Failed to get telemetry history: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_prediction_history(self, engine_id: str, hours: int = 24) -> pd.DataFrame:
        """Get prediction history for an engine"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT timestamp, predicted_rul, confidence, anomaly_score, is_anomaly
                FROM predictions 
                WHERE engine_id = %s 
                AND timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            df = pd.read_sql(query, conn, params=(engine_id, hours))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            st.error(f"Failed to get prediction history: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts from Redis"""
        if not self.redis_client:
            return []
        
        try:
            alerts = self.redis_client.lrange("global_alerts", 0, limit - 1)
            return [json.loads(alert) for alert in alerts]
        except Exception as e:
            st.error(f"Failed to get recent alerts: {e}")
            return []
    
    def check_system_health(self) -> Dict[str, str]:
        """Check health of all system components"""
        health_status = {}
        
        # Check Redis
        try:
            if self.redis_client:
                self.redis_client.ping()
                health_status['redis'] = 'online'
            else:
                health_status['redis'] = 'offline'
        except:
            health_status['redis'] = 'offline'
        
        # Check PostgreSQL
        try:
            conn = self.get_db_connection()
            if conn:
                conn.close()
                health_status['postgresql'] = 'online'
            else:
                health_status['postgresql'] = 'offline'
        except:
            health_status['postgresql'] = 'offline'
        
        # Check Kafka (via Redis health data)
        try:
            if self.redis_client:
                health_data = self.redis_client.get("system_health")
                if health_data:
                    data = json.loads(health_data)
                    # If we're getting recent telemetry, Kafka is working
                    if data.get('telemetry_count_1h', 0) > 0:
                        health_status['kafka'] = 'online'
                    else:
                        health_status['kafka'] = 'degraded'
                else:
                    health_status['kafka'] = 'degraded'
            else:
                health_status['kafka'] = 'offline'
        except:
            health_status['kafka'] = 'offline'
        
        return health_status

# Initialize dashboard service
@st.cache_resource
def get_dashboard_service():
    return DashboardService()

dashboard = get_dashboard_service()

# Main dashboard
def main():
    st.title("Aircraft Engine Monitoring System")
    
    # System Health Status Section
    st.markdown("### System Status")
    health_status = dashboard.check_system_health()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        status = health_status.get('kafka', 'offline')
        icon_class = f'health-{status}'
        st.markdown(f'<span class="health-indicator {icon_class}"></span> Kafka: {status.upper()}', unsafe_allow_html=True)
    with col2:
        status = health_status.get('postgresql', 'offline')
        icon_class = f'health-{status}'
        st.markdown(f'<span class="health-indicator {icon_class}"></span> PostgreSQL: {status.upper()}', unsafe_allow_html=True)
    with col3:
        status = health_status.get('redis', 'offline')
        icon_class = f'health-{status}'
        st.markdown(f'<span class="health-indicator {icon_class}"></span> Redis: {status.upper()}', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # System metrics header
    st.markdown("### System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    health_data = dashboard.get_system_health()
    
    with col1:
        telemetry_count = health_data.get('telemetry_count_1h', 0)
        st.metric("Telemetry Records (1h)", f"{telemetry_count:,}")
        st.markdown('<p class="info-tooltip">Total sensor readings received</p>', unsafe_allow_html=True)
    
    with col2:
        predictions_count = health_data.get('predictions_count_1h', 0)
        st.metric("Predictions (1h)", f"{predictions_count:,}")
        st.markdown('<p class="info-tooltip">ML predictions generated</p>', unsafe_allow_html=True)
    
    with col3:
        active_engines = health_data.get('active_engines_count', 0)
        st.metric("Active Engines", active_engines)
        st.markdown('<p class="info-tooltip">Engines currently monitored</p>', unsafe_allow_html=True)
    
    with col4:
        active_alerts = health_data.get('active_alerts_count', 0)
        if active_alerts > 0:
            st.metric("Active Alerts", active_alerts, delta=active_alerts, delta_color="inverse")
        else:
            st.metric("Active Alerts", active_alerts)
        st.markdown('<p class="info-tooltip">Anomalies and warnings</p>', unsafe_allow_html=True)
    
    # Get engine list
    engines = dashboard.get_engine_list()
    
    if not engines:
        st.warning("No engines found in the system. Please check if the data simulator is running and Kafka is streaming data.")
        return
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Sidebar for engine selection
    with st.sidebar:
        st.header("Engine Selection")
        selected_engine = st.selectbox("Select Engine", engines, index=0)
        
        st.header("Time Range")
        time_range = st.selectbox("Telemetry History", ["1 Hour", "6 Hours", "24 Hours"], index=0)
        hours_map = {"1 Hour": 1, "6 Hours": 6, "24 Hours": 24}
        selected_hours = hours_map[time_range]
        
        prediction_range = st.selectbox("Prediction History", ["1 Hour", "6 Hours", "24 Hours"], index=2)
        prediction_hours = hours_map[prediction_range]
        
        # Auto-refresh option
        st.header("Refresh Settings")
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 3, 30, 5)
        else:
            refresh_interval = 5
        
        if st.button("Refresh Data Now"):
            st.rerun()
    
    # Main content area
    if selected_engine:
        # Engine overview
        st.markdown(f"### Engine {selected_engine} - Real-time Monitoring")
        
        # Get latest data
        latest_telemetry = dashboard.get_latest_telemetry(selected_engine)
        latest_prediction = dashboard.get_latest_prediction(selected_engine)
        
        # Engine status cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if latest_prediction:
                rul = latest_prediction.get('predicted_rul', 'N/A')
                confidence = latest_prediction.get('confidence', 0)
                
                if isinstance(rul, (int, float)):
                    st.metric("Remaining Useful Life", f"{int(rul)} cycles")
                    st.markdown(f'<p class="info-tooltip">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)
                else:
                    st.metric("Remaining Useful Life", "N/A")
                    st.markdown('<p class="info-tooltip">No prediction available</p>', unsafe_allow_html=True)
            else:
                st.metric("Remaining Useful Life", "No data")
                st.markdown('<p class="info-tooltip">Waiting for ML predictions</p>', unsafe_allow_html=True)
        
        with col2:
            if latest_prediction:
                anomaly_score = latest_prediction.get('anomaly_score', 0)
                is_anomaly = latest_prediction.get('is_anomaly', False)
                
                status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
                st.metric("Anomaly Status", status)
                st.markdown(f'<p class="info-tooltip">Anomaly Score: {anomaly_score:.3f}</p>', unsafe_allow_html=True)
            else:
                st.metric("Anomaly Status", "No data")
                st.markdown('<p class="info-tooltip">Waiting for anomaly detection</p>', unsafe_allow_html=True)
        
        with col3:
            if latest_telemetry:
                cycle = latest_telemetry.get('cycle', 0)
                timestamp = latest_telemetry.get('timestamp', '')
                
                st.metric("Current Cycle", f"{cycle:,}")
                
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_ago = datetime.now() - dt.replace(tzinfo=None)
                    if time_ago.total_seconds() < 60:
                        time_str = f"{int(time_ago.total_seconds())}s ago"
                    elif time_ago.total_seconds() < 3600:
                        time_str = f"{int(time_ago.total_seconds() // 60)}m ago"
                    else:
                        time_str = f"{int(time_ago.total_seconds() // 3600)}h ago"
                    st.markdown(f'<p class="info-tooltip">Last update: {time_str}</p>', unsafe_allow_html=True)
                except:
                    st.markdown('<p class="info-tooltip">Last update: Unknown</p>', unsafe_allow_html=True)
            else:
                st.metric("Current Cycle", "No data")
                st.markdown('<p class="info-tooltip">No telemetry received</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Telemetry charts
        st.markdown("### Real-time Sensor Data")
        
        with st.spinner("Loading sensor data..."):
            telemetry_df = dashboard.get_telemetry_history(selected_engine, selected_hours)
        
        if not telemetry_df.empty:
            # Create sensor charts with proper labels and spacing
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Total Temperature (Sensor 2)',
                    'Total Pressure (Sensor 3)',
                    'Physical Fan Speed (Sensor 11)',
                    'Corrected Fan Speed (Sensor 14)'
                ),
                vertical_spacing=0.20,  # Increased vertical spacing
                horizontal_spacing=0.15  # Increased horizontal spacing
            )
            
            # Temperature sensors (°R - Rankine)
            fig.add_trace(
                go.Scatter(
                    x=telemetry_df['timestamp'],
                    y=telemetry_df['sensor_2'],
                    name='Total Temp',
                    line=dict(color='#ff7f0e', width=2),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Pressure sensors (psia)
            fig.add_trace(
                go.Scatter(
                    x=telemetry_df['timestamp'],
                    y=telemetry_df['sensor_3'],
                    name='Total Pressure',
                    line=dict(color='#2ca02c', width=2),
                    mode='lines'
                ),
                row=1, col=2
            )
            
            # Physical fan speed (rpm)
            fig.add_trace(
                go.Scatter(
                    x=telemetry_df['timestamp'],
                    y=telemetry_df['sensor_11'],
                    name='Physical Fan Speed',
                    line=dict(color='#d62728', width=2),
                    mode='lines'
                ),
                row=2, col=1
            )
            
            # Corrected fan speed (rpm)
            fig.add_trace(
                go.Scatter(
                    x=telemetry_df['timestamp'],
                    y=telemetry_df['sensor_14'],
                    name='Corrected Fan Speed',
                    line=dict(color='#9467bd', width=2),
                    mode='lines'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=700,  # Increased height for better spacing
                showlegend=False,
                title_text="Key Sensor Readings Over Time",
                title_font=dict(size=16, color='#24292e'),
                font=dict(size=11),
                margin=dict(l=80, r=80, t=100, b=80)  # Added proper margins
            )
            
            # Update axes with units and better formatting
            fig.update_xaxes(
                title_text="Time",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                title_standoff=15,
                row=1, col=1
            )
            fig.update_xaxes(
                title_text="Time",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                title_standoff=15,
                row=1, col=2
            )
            fig.update_xaxes(
                title_text="Time",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                title_standoff=15,
                row=2, col=1
            )
            fig.update_xaxes(
                title_text="Time",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                title_standoff=15,
                row=2, col=2
            )
            
            fig.update_yaxes(
                title_text="Temperature (°R)",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                title_standoff=10,
                row=1, col=1
            )
            fig.update_yaxes(
                title_text="Pressure (psia)",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                title_standoff=10,
                row=1, col=2
            )
            fig.update_yaxes(
                title_text="Speed (rpm)",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                title_standoff=10,
                row=2, col=1
            )
            fig.update_yaxes(
                title_text="Speed (rpm)",
                title_font=dict(size=12),
                tickfont=dict(size=10),
                title_standoff=10,
                row=2, col=2
            )
            
            # Update subplot titles appearance
            for annotation in fig['layout']['annotations']:
                annotation['font'] = dict(size=13, color='#24292e')
                annotation['y'] = annotation['y'] + 0.01  # Move titles up slightly
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No telemetry data available for the selected time range ({time_range}). The data simulator may have started recently.")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # RUL Prediction Trend
        st.markdown("### RUL Prediction Trend")
        
        with st.spinner("Loading prediction data..."):
            prediction_df = dashboard.get_prediction_history(selected_engine, prediction_hours)
        
        if not prediction_df.empty:
            fig = go.Figure()
            
            # Calculate proper confidence intervals
            # Using a simple approach: confidence as percentage of prediction value
            margin = prediction_df['predicted_rul'] * (1 - prediction_df['confidence'])
            upper_bound = prediction_df['predicted_rul'] + margin
            lower_bound = prediction_df['predicted_rul'] - margin
            lower_bound = lower_bound.clip(lower=0)  # RUL can't be negative
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=prediction_df['timestamp'],
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False,
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                x=prediction_df['timestamp'],
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Range',
                fillcolor='rgba(31,119,180,0.2)',
                showlegend=True
            ))
            
            # Add main prediction line
            fig.add_trace(go.Scatter(
                x=prediction_df['timestamp'],
                y=prediction_df['predicted_rul'],
                mode='lines+markers',
                name='Predicted RUL',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6, color='#1f77b4')
            ))
            
            # Add critical threshold line
            fig.add_hline(
                y=20,
                line_dash="dash",
                line_color="red",
                annotation_text="Critical Threshold (20 cycles)",
                annotation_position="right"
            )
            
            # Add warning threshold line
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="orange",
                annotation_text="Warning Threshold (50 cycles)",
                annotation_position="right"
            )
            
            fig.update_layout(
                title="RUL Prediction with Confidence Intervals",
                xaxis_title="Time",
                yaxis_title="Remaining Useful Life (cycles)",
                height=450,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation help
            st.markdown('<p class="info-tooltip">The shaded area represents prediction uncertainty based on model confidence. Lower confidence creates wider bands.</p>', unsafe_allow_html=True)
        else:
            st.info(f"No prediction data available for the selected time range ({prediction_range}). The ML service may be processing initial data.")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Recent Alerts Section
    st.markdown("### Recent System Alerts")
    
    alerts = dashboard.get_recent_alerts(20)
    
    if alerts:
        # Alert summary
        critical_count = sum(1 for a in alerts if a.get('severity') == 'critical')
        warning_count = sum(1 for a in alerts if a.get('severity') == 'medium')
        info_count = len(alerts) - critical_count - warning_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Critical Alerts", critical_count)
        with col2:
            st.metric("Warning Alerts", warning_count)
        with col3:
            st.metric("Info Alerts", info_count)
        
        st.markdown("---")
        
        # Display alerts
        for alert in alerts:
            severity = alert.get('severity', 'low')
            message = alert.get('message', '')
            engine_id = alert.get('engine_id', '')
            timestamp = alert.get('timestamp', '')
            alert_type = alert.get('alert_type', '')
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                time_str = timestamp
            
            # Choose alert style based on severity
            if severity == 'critical':
                alert_class = 'alert-critical'
            elif severity == 'medium':
                alert_class = 'alert-warning'
            else:
                alert_class = 'alert-info'
            
            st.markdown(f"""
            <div class="{alert_class}">
                <div class="alert-title">{engine_id} - {alert_type.upper()}</div>
                <div class="alert-content">{message}</div>
                <div class="alert-meta">{time_str}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No recent alerts - All systems operating normally")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()