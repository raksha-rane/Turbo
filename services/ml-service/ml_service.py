import os
import json
import time
import logging
import pandas as pd
import numpy as np
import joblib
import redis
from datetime import datetime, timedelta
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import signal
import sys
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        # Kafka configuration
        self.kafka_broker = os.getenv('KAFKA_BROKER', 'kafka:9092')
        self.telemetry_topic = os.getenv('KAFKA_TELEMETRY_TOPIC', 'engine-telemetry')
        self.predictions_topic = os.getenv('KAFKA_PREDICTIONS_TOPIC', 'engine-predictions')
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        
        # ML configuration
        self.model_update_interval = int(os.getenv('MODEL_UPDATE_INTERVAL', '3600'))
        self.prediction_threshold = float(os.getenv('PREDICTION_THRESHOLD', '0.8'))
        self.anomaly_threshold = float(os.getenv('ANOMALY_THRESHOLD', '0.95'))
        
        # Sensor columns
        self.sensor_columns = [
            'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
            'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
            'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
            'sensor_19', 'sensor_20', 'sensor_21'
        ]
        
        # Initialize components
        self.consumer = None
        self.producer = None
        self.redis_client = None
        self.rul_model = None
        self.anomaly_model = None
        self.scaler = None
        self.running = False
        
        self.last_model_update = None
        self.engine_data_buffer = {}  # Buffer for engine data
        
    def _connect_services(self) -> bool:
        """Connect to Kafka and Redis"""
        try:
            # Connect to Kafka consumer
            self.consumer = KafkaConsumer(
                self.telemetry_topic,
                bootstrap_servers=[self.kafka_broker],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                group_id='ml-service-group',
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=5000
            )
            
            # Connect to Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_broker],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=5,
                retry_backoff_ms=1000
            )
            
            # Connect to Redis
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test Redis connection
            self.redis_client.ping()
            
            logger.info("Successfully connected to Kafka and Redis")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to services: {e}")
            return False
    
    def _load_training_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load C-MAPSS training data and RUL labels"""
        try:
            # Load training data
            train_file = '/app/data/train_FD001.txt'
            rul_file = '/app/data/RUL_FD001.txt'
            
            if not os.path.exists(train_file) or not os.path.exists(rul_file):
                logger.warning("Training data files not found")
                return None, None
            
            # Load training data
            train_data = pd.read_csv(
                train_file,
                sep=' ',
                header=None,
                names=['unit_number', 'time_cycles'] + self.sensor_columns,
                usecols=range(26)
            )
            
            # Load RUL labels
            rul_data = pd.read_csv(rul_file, header=None, names=['rul'])
            
            logger.info(f"Loaded training data: {len(train_data)} records, {len(rul_data)} RUL labels")
            return train_data, rul_data
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None, None
    
    def _prepare_training_features(self, train_data: pd.DataFrame, rul_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        try:
            # For each engine, calculate features based on last N cycles
            window_size = 50  # Use last 50 cycles for feature engineering
            features_list = []
            labels_list = []
            
            for unit in train_data['unit_number'].unique():
                unit_data = train_data[train_data['unit_number'] == unit].sort_values('time_cycles')
                
                if len(unit_data) < window_size:
                    continue
                
                # Get the last window_size cycles
                last_cycles = unit_data.tail(window_size)
                
                # Calculate statistical features
                features = []
                for col in self.sensor_columns:
                    if col in last_cycles.columns:
                        values = last_cycles[col].values
                        features.extend([
                            np.mean(values),
                            np.std(values),
                            np.min(values),
                            np.max(values),
                            np.median(values),
                            values[-1] - values[0],  # Trend
                            np.mean(np.diff(values)) if len(values) > 1 else 0  # Average rate of change
                        ])
                    else:
                        features.extend([0.0] * 7)  # Fill with zeros if column missing
                
                features_list.append(features)
                
                # Get corresponding RUL label
                unit_idx = unit - 1  # Units are 1-indexed
                if unit_idx < len(rul_data):
                    labels_list.append(rul_data.iloc[unit_idx]['rul'])
            
            X = np.array(features_list)
            y = np.array(labels_list)
            
            logger.info(f"Prepared training features: {X.shape}, labels: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training features: {e}")
            return np.array([]), np.array([])
    
    def _train_models(self) -> bool:
        """Train RUL prediction and anomaly detection models"""
        try:
            logger.info("Training ML models...")
            
            # Load training data
            train_data, rul_data = self._load_training_data()
            if train_data is None or rul_data is None:
                logger.error("Cannot train models without training data")
                return False
            
            # Prepare features
            X, y = self._prepare_training_features(train_data, rul_data)
            if len(X) == 0 or len(y) == 0:
                logger.error("No features prepared for training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train RUL prediction model
            self.rul_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            )
            
            self.rul_model.fit(X_train_scaled, y_train)
            
            # Evaluate RUL model
            y_pred = self.rul_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"RUL Model - MSE: {mse:.2f}, MAE: {mae:.2f}")
            
            # Train anomaly detection model with lower contamination for demo
            self.anomaly_model = IsolationForest(
                contamination=0.01,  # Expect only 1% anomalies (less sensitive)
                random_state=42,
                n_jobs=-1
            )
            
            self.anomaly_model.fit(X_train_scaled)
            
            # Save models
            self._save_models()
            
            self.last_model_update = datetime.now()
            logger.info("Models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs('/app/models', exist_ok=True)
            
            if self.rul_model:
                joblib.dump(self.rul_model, '/app/models/rul_model.pkl')
            
            if self.anomaly_model:
                joblib.dump(self.anomaly_model, '/app/models/anomaly_model.pkl')
            
            if self.scaler:
                joblib.dump(self.scaler, '/app/models/scaler.pkl')
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            rul_model_path = '/app/models/rul_model.pkl'
            anomaly_model_path = '/app/models/anomaly_model.pkl'
            scaler_path = '/app/models/scaler.pkl'
            
            if all(os.path.exists(path) for path in [rul_model_path, anomaly_model_path, scaler_path]):
                self.rul_model = joblib.load(rul_model_path)
                self.anomaly_model = joblib.load(anomaly_model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Models loaded successfully")
                return True
            else:
                logger.info("No pre-trained models found, will train new models")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _extract_features(self, engine_data: List[Dict]) -> Optional[np.ndarray]:
        """Extract features from engine telemetry data"""
        try:
            if len(engine_data) < 10:  # Need minimum data for feature extraction
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(engine_data)
            
            # Calculate statistical features
            features = []
            for col in self.sensor_columns:
                if col in df.columns:
                    values = df[col].values
                    features.extend([
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values),
                        np.median(values),
                        values[-1] - values[0] if len(values) > 1 else 0,  # Trend
                        np.mean(np.diff(values)) if len(values) > 1 else 0  # Average rate of change
                    ])
                else:
                    features.extend([0.0] * 7)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _make_prediction(self, telemetry: Dict) -> Optional[Dict]:
        """Make RUL prediction and anomaly detection for telemetry data"""
        try:
            engine_id = telemetry['engine_id']
            
            # Add telemetry to engine buffer
            if engine_id not in self.engine_data_buffer:
                self.engine_data_buffer[engine_id] = []
            
            self.engine_data_buffer[engine_id].append(telemetry)
            
            # Keep only last 100 data points
            if len(self.engine_data_buffer[engine_id]) > 100:
                self.engine_data_buffer[engine_id] = self.engine_data_buffer[engine_id][-100:]
            
            # Extract features
            features = self._extract_features(self.engine_data_buffer[engine_id])
            if features is None:
                return None
            
            # Scale features
            if self.scaler is None:
                logger.warning("Scaler not available")
                return None
            
            features_scaled = self.scaler.transform(features)
            
            # Make RUL prediction
            predicted_rul = None
            confidence = 0.0
            if self.rul_model:
                predicted_rul = max(0, int(self.rul_model.predict(features_scaled)[0]))
                
                # Calculate confidence based on model's decision function
                # For RandomForest, we can use the variance of tree predictions as uncertainty
                try:
                    tree_predictions = [tree.predict(features_scaled)[0] for tree in self.rul_model.estimators_]
                    prediction_std = np.std(tree_predictions)
                    confidence = max(0.1, min(1.0, 1.0 - (prediction_std / np.mean(tree_predictions))))
                except:
                    confidence = 0.8  # Default confidence
            
            # Anomaly detection
            is_anomaly = False
            anomaly_score = 0.0
            if self.anomaly_model:
                anomaly_pred = self.anomaly_model.predict(features_scaled)[0]
                is_anomaly = bool(anomaly_pred == -1)  # Convert numpy bool to Python bool
                
                # Get anomaly score
                try:
                    anomaly_score = self.anomaly_model.decision_function(features_scaled)[0]
                    anomaly_score = float(max(0.0, min(1.0, (anomaly_score + 0.5) * 2)))  # Normalize to 0-1
                except:
                    anomaly_score = 0.5
            
            # Create prediction result
            prediction = {
                'engine_id': engine_id,
                'timestamp': telemetry['timestamp'],
                'predicted_rul': int(predicted_rul) if predicted_rul is not None else None,
                'confidence': round(float(confidence), 3),
                'anomaly_score': round(float(anomaly_score), 3),
                'is_anomaly': is_anomaly,
                'cycle': int(telemetry.get('cycle', 0))
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def _send_prediction(self, prediction: Dict) -> bool:
        """Send prediction to Kafka and cache in Redis"""
        try:
            # Send to Kafka
            future = self.producer.send(
                self.predictions_topic,
                key=prediction['engine_id'],
                value=prediction
            )
            future.get(timeout=10)
            
            # Cache in Redis
            redis_key = f"prediction:{prediction['engine_id']}"
            self.redis_client.setex(
                redis_key,
                300,  # 5 minutes TTL
                json.dumps(prediction)
            )
            
            # Store in recent predictions list
            recent_key = f"recent_predictions:{prediction['engine_id']}"
            self.redis_client.lpush(recent_key, json.dumps(prediction))
            self.redis_client.ltrim(recent_key, 0, 99)  # Keep last 100 predictions
            self.redis_client.expire(recent_key, 3600)  # 1 hour TTL
            
            logger.debug(f"Sent prediction for {prediction['engine_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending prediction: {e}")
            return False
    
    def _process_telemetry(self):
        """Process incoming telemetry data"""
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                telemetry = message.value
                logger.debug(f"Processing telemetry for {telemetry.get('engine_id', 'unknown')}")
                
                # Make prediction
                prediction = self._make_prediction(telemetry)
                if prediction:
                    self._send_prediction(prediction)
                
                # Check if models need retraining
                if (self.last_model_update is None or 
                    datetime.now() - self.last_model_update > timedelta(seconds=self.model_update_interval)):
                    logger.info("Triggering model retraining...")
                    # In a production system, this would trigger a separate training job
                    # For now, we just update the timestamp
                    self.last_model_update = datetime.now()
                    
        except Exception as e:
            logger.error(f"Error processing telemetry: {e}")
    
    def start_service(self):
        """Start the ML service"""
        logger.info("Starting ML Service...")
        
        # Connect to services
        if not self._connect_services():
            logger.error("Failed to connect to services. Exiting.")
            return
        
        # Load or train models
        if not self._load_models():
            if not self._train_models():
                logger.error("Failed to load or train models. Exiting.")
                return
        
        self.running = True
        
        try:
            logger.info("ML Service started successfully")
            self._process_telemetry()
            
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        except Exception as e:
            logger.error(f"Service error: {e}")
        finally:
            self.stop_service()
    
    def stop_service(self):
        """Stop the ML service"""
        logger.info("Stopping ML service...")
        self.running = False
        
        if self.consumer:
            try:
                self.consumer.close()
            except Exception as e:
                logger.error(f"Error closing consumer: {e}")
        
        if self.producer:
            try:
                self.producer.flush()
                self.producer.close()
            except Exception as e:
                logger.error(f"Error closing producer: {e}")
        
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis client: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_service()
        sys.exit(0)

def main():
    """Main function to run the ML service"""
    ml_service = MLService()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, ml_service.signal_handler)
    signal.signal(signal.SIGTERM, ml_service.signal_handler)
    
    # Start service
    ml_service.start_service()

if __name__ == "__main__":
    main()