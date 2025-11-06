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
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .alert-high {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    
    .alert-medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    
    .alert-low {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-critical {
        color: #dc3545;
        font-weight: bold;
    }
    
    .engine-card {
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
    }
    
    h1 {
        color: #2c3e50;
        text-align: center;
        padding-bottom: 2rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
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

# Initialize dashboard service
@st.cache_resource
def get_dashboard_service():
    return DashboardService()

dashboard = get_dashboard_service()

# Main dashboard
def main():
    st.title("Aircraft Engine Monitoring System")
    
    # System status header
    col1, col2, col3, col4 = st.columns(4)
    
    health_data = dashboard.get_system_health()
    
    with col1:
        telemetry_count = health_data.get('telemetry_count_1h', 0)
        st.metric("Telemetry Records (1h)", telemetry_count)
    
    with col2:
        predictions_count = health_data.get('predictions_count_1h', 0)
        st.metric("Predictions (1h)", predictions_count)
    
    with col3:
        active_engines = health_data.get('active_engines_count', 0)
        st.metric("Active Engines", active_engines)
    
    with col4:
        active_alerts = health_data.get('active_alerts_count', 0)
        if active_alerts > 0:
            st.metric("Active Alerts", active_alerts, delta=active_alerts, delta_color="inverse")
        else:
            st.metric("Active Alerts", active_alerts)
    
    # Get engine list
    engines = dashboard.get_engine_list()
    
    if not engines:
        st.warning("No engines found in the system")
        return
    
    # Sidebar for engine selection
    with st.sidebar:
        st.header("Engine Selection")
        selected_engine = st.selectbox("Select Engine", engines, index=0)
        
        st.header("Time Range")
        time_range = st.selectbox("Telemetry History", ["1 Hour", "6 Hours", "24 Hours"], index=0)
        hours_map = {"1 Hour": 1, "6 Hours": 6, "24 Hours": 24}
        selected_hours = hours_map[time_range]
        
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto Refresh (5s)", value=True)
        
        if st.button("Refresh Data"):
            st.experimental_rerun()
    
    # Main content area
    if selected_engine:
        # Engine overview
        st.header(f"Engine {selected_engine} - Real-time Monitoring")
        
        # Get latest data
        latest_telemetry = dashboard.get_latest_telemetry(selected_engine)
        latest_prediction = dashboard.get_latest_prediction(selected_engine)
        
        # Engine status cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if latest_prediction:
                rul = latest_prediction.get('predicted_rul', 'N/A')
                confidence = latest_prediction.get('confidence', 0)
                
                if isinstance(rul, int):
                    if rul < 20:
                        status_class = "status-critical"
                        delta_color = "inverse"
                    elif rul < 50:
                        status_class = "status-warning"
                        delta_color = "off"
                    else:
                        status_class = "status-healthy"
                        delta_color = "normal"
                    
                    st.metric(
                        "Remaining Useful Life",
                        f"{rul} cycles",
                        delta=f"{confidence:.1%} confidence"
                    )
                else:
                    st.metric("Remaining Useful Life", "N/A")
            else:
                st.metric("Remaining Useful Life", "No data")
        
        with col2:
            if latest_prediction:
                anomaly_score = latest_prediction.get('anomaly_score', 0)
                is_anomaly = latest_prediction.get('is_anomaly', False)
                
                status = "ANOMALY" if is_anomaly else "NORMAL"
                delta_val = f"{anomaly_score:.3f}" if anomaly_score else None
                
                st.metric(
                    "Anomaly Status",
                    status,
                    delta=delta_val,
                    delta_color="inverse" if is_anomaly else "normal"
                )
            else:
                st.metric("Anomaly Status", "No data")
        
        with col3:
            if latest_telemetry:
                cycle = latest_telemetry.get('cycle', 0)
                timestamp = latest_telemetry.get('timestamp', '')
                
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_ago = datetime.now() - dt.replace(tzinfo=None)
                    if time_ago.total_seconds() < 60:
                        time_str = f"{int(time_ago.total_seconds())}s ago"
                    else:
                        time_str = f"{int(time_ago.total_seconds() // 60)}m ago"
                except:
                    time_str = "Unknown"
                
                st.metric("Current Cycle", cycle, delta=time_str)
            else:
                st.metric("Current Cycle", "No data")
        
        # Telemetry charts
        st.subheader("Real-time Sensor Data")
        
        telemetry_df = dashboard.get_telemetry_history(selected_engine, selected_hours)
        
        if not telemetry_df.empty:
            # Create sensor charts
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Temperature Sensors', 'Pressure Sensors', 'Efficiency Sensors', 'Performance Sensors'),
                vertical_spacing=0.12
            )
            
            # Temperature sensors
            fig.add_trace(
                go.Scatter(x=telemetry_df['timestamp'], y=telemetry_df['sensor_2'], 
                          name='Temp Sensor 2', line=dict(color='#ff7f0e')),
                row=1, col=1
            )
            
            # Pressure sensors
            fig.add_trace(
                go.Scatter(x=telemetry_df['timestamp'], y=telemetry_df['sensor_3'], 
                          name='Pressure Sensor 3', line=dict(color='#2ca02c')),
                row=1, col=2
            )
            
            # Efficiency sensors
            fig.add_trace(
                go.Scatter(x=telemetry_df['timestamp'], y=telemetry_df['sensor_11'], 
                          name='Efficiency Sensor 11', line=dict(color='#d62728')),
                row=2, col=1
            )
            
            # Performance sensors
            fig.add_trace(
                go.Scatter(x=telemetry_df['timestamp'], y=telemetry_df['sensor_14'], 
                          name='Performance Sensor 14', line=dict(color='#9467bd')),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Key Sensor Readings Over Time"
            )
            
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Value")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No telemetry data available for the selected time range")
        
        # RUL Prediction Trend
        st.subheader("RUL Prediction Trend")
        
        prediction_df = dashboard.get_prediction_history(selected_engine, 24)
        
        if not prediction_df.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=prediction_df['timestamp'],
                y=prediction_df['predicted_rul'],
                mode='lines+markers',
                name='Predicted RUL',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6)
            ))
            
            # Add confidence bands
            upper_bound = prediction_df['predicted_rul'] * (1 + (1 - prediction_df['confidence']))
            lower_bound = prediction_df['predicted_rul'] * prediction_df['confidence']
            
            fig.add_trace(go.Scatter(
                x=prediction_df['timestamp'],
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=prediction_df['timestamp'],
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0)',
                name='Confidence Range',
                fillcolor='rgba(31,119,180,0.2)'
            ))
            
            fig.update_layout(
                title="RUL Prediction with Confidence Bands",
                xaxis_title="Time",
                yaxis_title="Remaining Useful Life (cycles)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction data available")
    
    # Recent Alerts Section
    st.header("Recent System Alerts")
    
    alerts = dashboard.get_recent_alerts(20)
    
    if alerts:
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
                alert_class = 'alert-high'
            elif severity == 'medium':
                alert_class = 'alert-medium'
            else:
                alert_class = 'alert-low'
            
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>{engine_id}</strong> - {alert_type.upper()}<br>
                {message}<br>
                <small>{time_str}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent alerts")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(5)
        st.experimental_rerun()

if __name__ == "__main__":
    main()