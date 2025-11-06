import os
import json
import time
import logging
import psycopg2
import redis
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import signal
import sys
from typing import Dict, List, Optional
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConsumerService:
    def __init__(self):
        # Kafka configuration
        self.kafka_broker = os.getenv('KAFKA_BROKER', 'kafka:9092')
        self.telemetry_topic = os.getenv('KAFKA_TELEMETRY_TOPIC', 'engine-telemetry')
        self.predictions_topic = os.getenv('KAFKA_PREDICTIONS_TOPIC', 'engine-predictions')
        
        # PostgreSQL configuration
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
        
        # Initialize components
        self.telemetry_consumer = None
        self.predictions_consumer = None
        self.redis_client = None
        self.running = False
        
    def _connect_services(self) -> bool:
        """Connect to Kafka, PostgreSQL, and Redis"""
        try:
            # Connect to Kafka consumers
            self.telemetry_consumer = KafkaConsumer(
                self.telemetry_topic,
                bootstrap_servers=[self.kafka_broker],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                group_id='consumer-service-telemetry',
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=5000
            )
            
            self.predictions_consumer = KafkaConsumer(
                self.predictions_topic,
                bootstrap_servers=[self.kafka_broker],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                group_id='consumer-service-predictions',
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=5000
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
            
            # Test connections
            self.redis_client.ping()
            self._test_postgres_connection()
            
            logger.info("Successfully connected to all services")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to services: {e}")
            return False
    
    def _test_postgres_connection(self):
        """Test PostgreSQL connection"""
        with self._get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
    
    @contextmanager
    def _get_db_connection(self):
        """Get PostgreSQL database connection with context manager"""
        conn = None
        try:
            conn = psycopg2.connect(**self.postgres_config)
            conn.autocommit = True
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def _store_telemetry(self, telemetry: Dict) -> bool:
        """Store telemetry data in PostgreSQL and cache in Redis"""
        try:
            # Store in PostgreSQL
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    insert_query = """
                        INSERT INTO telemetry (
                            engine_id, timestamp, cycle,
                            operational_setting_1, operational_setting_2, operational_setting_3,
                            sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6,
                            sensor_7, sensor_8, sensor_9, sensor_10, sensor_11, sensor_12,
                            sensor_13, sensor_14, sensor_15, sensor_16, sensor_17, sensor_18,
                            sensor_19, sensor_20, sensor_21
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        telemetry['engine_id'],
                        datetime.fromisoformat(telemetry['timestamp'].replace('Z', '+00:00')),
                        telemetry.get('cycle', 0),
                        telemetry.get('operational_setting_1', 0),
                        telemetry.get('operational_setting_2', 0),
                        telemetry.get('operational_setting_3', 0),
                        telemetry.get('sensor_1', 0),
                        telemetry.get('sensor_2', 0),
                        telemetry.get('sensor_3', 0),
                        telemetry.get('sensor_4', 0),
                        telemetry.get('sensor_5', 0),
                        telemetry.get('sensor_6', 0),
                        telemetry.get('sensor_7', 0),
                        telemetry.get('sensor_8', 0),
                        telemetry.get('sensor_9', 0),
                        telemetry.get('sensor_10', 0),
                        telemetry.get('sensor_11', 0),
                        telemetry.get('sensor_12', 0),
                        telemetry.get('sensor_13', 0),
                        telemetry.get('sensor_14', 0),
                        telemetry.get('sensor_15', 0),
                        telemetry.get('sensor_16', 0),
                        telemetry.get('sensor_17', 0),
                        telemetry.get('sensor_18', 0),
                        telemetry.get('sensor_19', 0),
                        telemetry.get('sensor_20', 0),
                        telemetry.get('sensor_21', 0)
                    )
                    
                    cursor.execute(insert_query, values)
            
            # Cache in Redis (keep last 100 telemetry records per engine)
            redis_key = f"telemetry:{telemetry['engine_id']}"
            self.redis_client.lpush(redis_key, json.dumps(telemetry))
            self.redis_client.ltrim(redis_key, 0, 99)  # Keep last 100 records
            self.redis_client.expire(redis_key, 3600)  # 1 hour TTL
            
            # Cache latest telemetry for quick access
            latest_key = f"latest_telemetry:{telemetry['engine_id']}"
            self.redis_client.setex(latest_key, 300, json.dumps(telemetry))  # 5 minutes TTL
            
            logger.debug(f"Stored telemetry for {telemetry['engine_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing telemetry: {e}")
            return False
    
    def _store_prediction(self, prediction: Dict) -> bool:
        """Store prediction data in PostgreSQL and cache in Redis"""
        try:
            # Store in PostgreSQL
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    insert_query = """
                        INSERT INTO predictions (
                            engine_id, timestamp, predicted_rul, confidence, anomaly_score, is_anomaly
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        prediction['engine_id'],
                        datetime.fromisoformat(prediction['timestamp'].replace('Z', '+00:00')),
                        prediction.get('predicted_rul'),
                        prediction.get('confidence', 0.0),
                        prediction.get('anomaly_score', 0.0),
                        prediction.get('is_anomaly', False)
                    )
                    
                    cursor.execute(insert_query, values)
            
            # Check for anomalies and create alerts
            if prediction.get('is_anomaly', False):
                self._create_alert(
                    prediction['engine_id'],
                    'anomaly',
                    'high',
                    f"Anomaly detected with score {prediction.get('anomaly_score', 0):.3f}",
                    prediction['timestamp']
                )
            
            # Check for low RUL and create alerts
            predicted_rul = prediction.get('predicted_rul', 0)
            if predicted_rul and predicted_rul < 50:  # Less than 50 cycles remaining
                severity = 'critical' if predicted_rul < 20 else 'medium'
                self._create_alert(
                    prediction['engine_id'],
                    'low_rul',
                    severity,
                    f"Low RUL predicted: {predicted_rul} cycles remaining",
                    prediction['timestamp']
                )
            
            # Cache latest prediction
            latest_key = f"latest_prediction:{prediction['engine_id']}"
            self.redis_client.setex(latest_key, 300, json.dumps(prediction))  # 5 minutes TTL
            
            logger.debug(f"Stored prediction for {prediction['engine_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return False
    
    def _create_alert(self, engine_id: str, alert_type: str, severity: str, message: str, timestamp: str) -> bool:
        """Create an alert in the database"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    insert_query = """
                        INSERT INTO alerts (engine_id, alert_type, severity, message, timestamp)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        engine_id,
                        alert_type,
                        severity,
                        message,
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    )
                    
                    cursor.execute(insert_query, values)
            
            # Cache alert for quick access
            alert_data = {
                'engine_id': engine_id,
                'alert_type': alert_type,
                'severity': severity,
                'message': message,
                'timestamp': timestamp,
                'acknowledged': False
            }
            
            alerts_key = f"alerts:{engine_id}"
            self.redis_client.lpush(alerts_key, json.dumps(alert_data))
            self.redis_client.ltrim(alerts_key, 0, 49)  # Keep last 50 alerts
            self.redis_client.expire(alerts_key, 86400)  # 24 hours TTL
            
            # Cache in global alerts list
            self.redis_client.lpush("global_alerts", json.dumps(alert_data))
            self.redis_client.ltrim("global_alerts", 0, 199)  # Keep last 200 alerts
            self.redis_client.expire("global_alerts", 86400)  # 24 hours TTL
            
            logger.info(f"Created {severity} alert for {engine_id}: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return False
    
    def _process_telemetry_messages(self):
        """Process telemetry messages from Kafka"""
        try:
            for message in self.telemetry_consumer:
                if not self.running:
                    break
                
                telemetry = message.value
                self._store_telemetry(telemetry)
                
        except Exception as e:
            logger.error(f"Error processing telemetry messages: {e}")
    
    def _process_prediction_messages(self):
        """Process prediction messages from Kafka"""
        try:
            for message in self.predictions_consumer:
                if not self.running:
                    break
                
                prediction = message.value
                self._store_prediction(prediction)
                
        except Exception as e:
            logger.error(f"Error processing prediction messages: {e}")
    
    def _update_system_health(self):
        """Update system health metrics in Redis"""
        try:
            # Get counts from database
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Count total telemetry records
                    cursor.execute("SELECT COUNT(*) FROM telemetry WHERE timestamp > NOW() - INTERVAL '1 hour'")
                    telemetry_count = cursor.fetchone()[0]
                    
                    # Count total predictions
                    cursor.execute("SELECT COUNT(*) FROM predictions WHERE timestamp > NOW() - INTERVAL '1 hour'")
                    predictions_count = cursor.fetchone()[0]
                    
                    # Count active alerts
                    cursor.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = false AND timestamp > NOW() - INTERVAL '24 hours'")
                    active_alerts_count = cursor.fetchone()[0]
                    
                    # Count engines with recent data
                    cursor.execute("SELECT COUNT(DISTINCT engine_id) FROM telemetry WHERE timestamp > NOW() - INTERVAL '5 minutes'")
                    active_engines_count = cursor.fetchone()[0]
            
            # Store health metrics in Redis
            health_data = {
                'telemetry_count_1h': telemetry_count,
                'predictions_count_1h': predictions_count,
                'active_alerts_count': active_alerts_count,
                'active_engines_count': active_engines_count,
                'last_updated': datetime.now().isoformat()
            }
            
            self.redis_client.setex("system_health", 60, json.dumps(health_data))
            
            logger.debug("Updated system health metrics")
            
        except Exception as e:
            logger.error(f"Error updating system health: {e}")
    
    def start_service(self):
        """Start the consumer service"""
        logger.info("Starting Consumer Service...")
        
        # Connect to services
        if not self._connect_services():
            logger.error("Failed to connect to services. Exiting.")
            return
        
        self.running = True
        
        try:
            logger.info("Consumer Service started successfully")
            
            # Run consumers in parallel (simplified approach)
            # In production, you'd use threading or asyncio
            last_health_update = time.time()
            
            while self.running:
                # Process telemetry messages
                try:
                    for message in self.telemetry_consumer:
                        if not self.running:
                            break
                        telemetry = message.value
                        self._store_telemetry(telemetry)
                        break  # Process one message at a time
                except:
                    pass  # Timeout is expected
                
                # Process prediction messages
                try:
                    for message in self.predictions_consumer:
                        if not self.running:
                            break
                        prediction = message.value
                        self._store_prediction(prediction)
                        break  # Process one message at a time
                except:
                    pass  # Timeout is expected
                
                # Update system health every 30 seconds
                current_time = time.time()
                if current_time - last_health_update > 30:
                    self._update_system_health()
                    last_health_update = current_time
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        except Exception as e:
            logger.error(f"Service error: {e}")
        finally:
            self.stop_service()
    
    def stop_service(self):
        """Stop the consumer service"""
        logger.info("Stopping consumer service...")
        self.running = False
        
        if self.telemetry_consumer:
            try:
                self.telemetry_consumer.close()
            except Exception as e:
                logger.error(f"Error closing telemetry consumer: {e}")
        
        if self.predictions_consumer:
            try:
                self.predictions_consumer.close()
            except Exception as e:
                logger.error(f"Error closing predictions consumer: {e}")
        
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
    """Main function to run the consumer service"""
    consumer_service = ConsumerService()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, consumer_service.signal_handler)
    signal.signal(signal.SIGTERM, consumer_service.signal_handler)
    
    # Start service
    consumer_service.start_service()

if __name__ == "__main__":
    main()