import os
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kafka import KafkaProducer
from kafka.errors import KafkaError
import signal
import sys
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EngineDataSimulator:
    def __init__(self):
        self.kafka_broker = os.getenv('KAFKA_BROKER', 'kafka:9092')
        self.telemetry_topic = os.getenv('KAFKA_TELEMETRY_TOPIC', 'engine-telemetry')
        self.simulation_interval = int(os.getenv('SIMULATION_INTERVAL', '2'))
        self.num_engines = int(os.getenv('NUM_ENGINES', '10'))
        
        # Sensor column names based on C-MAPSS dataset
        self.sensor_columns = [
            'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
            'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
            'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
            'sensor_19', 'sensor_20', 'sensor_21'
        ]
        
        # Initialize Kafka producer
        self.producer = None
        self.running = False
        
        # Engine states
        self.engine_states = {}
        self.baseline_data = None
        
        # Load baseline data from C-MAPSS dataset
        self._load_baseline_data()
        self._initialize_engine_states()
        
    def _load_baseline_data(self):
        """Load baseline sensor data from C-MAPSS dataset"""
        try:
            # Load training data to establish baseline patterns
            train_file = '/app/data/train_FD001.txt'
            if os.path.exists(train_file):
                self.baseline_data = pd.read_csv(
                    train_file, 
                    sep=' ', 
                    header=None,
                    names=['unit_number', 'time_cycles'] + self.sensor_columns,
                    usecols=range(26)  # Only use first 26 columns
                )
                logger.info(f"Loaded baseline data with {len(self.baseline_data)} records")
            else:
                logger.warning("Baseline data file not found, using synthetic data")
                self._create_synthetic_baseline()
                
        except Exception as e:
            logger.error(f"Error loading baseline data: {e}")
            self._create_synthetic_baseline()
    
    def _create_synthetic_baseline(self):
        """Create synthetic baseline data if real data is not available"""
        logger.info("Creating synthetic baseline data")
        np.random.seed(42)
        
        # Create synthetic data with realistic engine sensor ranges
        synthetic_data = {
            'operational_setting_1': np.random.normal(0, 0.01, 1000),
            'operational_setting_2': np.random.normal(0, 0.01, 1000),
            'operational_setting_3': np.random.normal(100, 5, 1000),
            'sensor_1': np.random.normal(518.67, 2, 1000),
            'sensor_2': np.random.normal(642, 5, 1000),
            'sensor_3': np.random.normal(1590, 20, 1000),
            'sensor_4': np.random.normal(1400, 15, 1000),
            'sensor_5': np.random.normal(14.62, 0.1, 1000),
            'sensor_6': np.random.normal(21.61, 0.2, 1000),
            'sensor_7': np.random.normal(554, 3, 1000),
            'sensor_8': np.random.normal(2388, 10, 1000),
            'sensor_9': np.random.normal(9050, 50, 1000),
            'sensor_10': np.random.normal(1.3, 0.05, 1000),
            'sensor_11': np.random.normal(47.3, 1, 1000),
            'sensor_12': np.random.normal(522, 3, 1000),
            'sensor_13': np.random.normal(2388, 10, 1000),
            'sensor_14': np.random.normal(8130, 40, 1000),
            'sensor_15': np.random.normal(8.4, 0.2, 1000),
            'sensor_16': np.random.normal(0.03, 0.005, 1000),
            'sensor_17': np.random.normal(392, 5, 1000),
            'sensor_18': np.random.normal(2388, 10, 1000),
            'sensor_19': np.random.normal(100, 1, 1000),
            'sensor_20': np.random.normal(39, 1, 1000),
            'sensor_21': np.random.normal(23.4, 0.3, 1000)
        }
        
        self.baseline_data = pd.DataFrame(synthetic_data)
    
    def _initialize_engine_states(self):
        """Initialize state for each engine"""
        for i in range(1, self.num_engines + 1):
            engine_id = f"ENG{i:03d}"
            
            # Get baseline values from dataset
            if self.baseline_data is not None and len(self.baseline_data) > 0:
                baseline_idx = np.random.randint(0, len(self.baseline_data))
                baseline_values = self.baseline_data.iloc[baseline_idx].to_dict()
            else:
                baseline_values = {col: 0.0 for col in self.sensor_columns}
            
            self.engine_states[engine_id] = {
                'cycle': 0,
                'baseline_values': baseline_values,
                'degradation_rate': np.random.uniform(0.0001, 0.001),  # Random degradation rate
                'noise_factor': np.random.uniform(0.8, 1.2),  # Individual engine variation
                'last_update': datetime.now()
            }
            
        logger.info(f"Initialized states for {len(self.engine_states)} engines")
    
    def _connect_kafka(self) -> bool:
        """Connect to Kafka broker"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_broker],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=5,
                retry_backoff_ms=1000,
                request_timeout_ms=30000
            )
            logger.info(f"Connected to Kafka broker: {self.kafka_broker}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def _generate_telemetry(self, engine_id: str) -> Dict:
        """Generate realistic telemetry data for an engine"""
        state = self.engine_states[engine_id]
        state['cycle'] += 1
        
        current_time = datetime.now()
        
        # Calculate degradation based on cycles
        degradation_factor = 1 + (state['cycle'] * state['degradation_rate'])
        
        telemetry = {
            'engine_id': engine_id,
            'timestamp': current_time.isoformat(),
            'cycle': state['cycle'],
        }
        
        # Generate sensor readings with degradation and noise
        for sensor in self.sensor_columns:
            baseline_value = state['baseline_values'].get(sensor, 0.0)
            
            # Apply degradation (some sensors increase, some decrease with wear)
            if sensor in ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_11', 'sensor_12']:
                # Temperature and pressure sensors typically increase with degradation
                degraded_value = baseline_value * degradation_factor
            elif sensor in ['sensor_7', 'sensor_8', 'sensor_13', 'sensor_14']:
                # Efficiency-related sensors typically decrease with degradation
                degraded_value = baseline_value / degradation_factor
            else:
                # Other sensors have mixed behavior
                sign = 1 if np.random.random() > 0.5 else -1
                degraded_value = baseline_value * (1 + sign * (degradation_factor - 1) * 0.5)
            
            # Add noise
            noise = np.random.normal(0, abs(degraded_value) * 0.01 * state['noise_factor'])
            final_value = degraded_value + noise
            
            telemetry[sensor] = round(final_value, 4)
        
        state['last_update'] = current_time
        return telemetry
    
    def _send_telemetry(self, telemetry: Dict) -> bool:
        """Send telemetry data to Kafka"""
        try:
            future = self.producer.send(
                self.telemetry_topic,
                key=telemetry['engine_id'],
                value=telemetry
            )
            
            # Wait for acknowledgment with timeout
            record_metadata = future.get(timeout=10)
            logger.debug(f"Sent telemetry for {telemetry['engine_id']} to partition {record_metadata.partition}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send telemetry for {telemetry['engine_id']}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending telemetry: {e}")
            return False
    
    def start_simulation(self):
        """Start the data simulation"""
        logger.info("Starting Aircraft Engine Data Simulation")
        
        # Connect to Kafka
        if not self._connect_kafka():
            logger.error("Failed to connect to Kafka. Exiting.")
            return
        
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                
                # Generate and send telemetry for all engines
                for engine_id in self.engine_states.keys():
                    if not self.running:
                        break
                        
                    telemetry = self._generate_telemetry(engine_id)
                    success = self._send_telemetry(telemetry)
                    
                    if success:
                        logger.debug(f"Generated telemetry for {engine_id}, cycle {telemetry['cycle']}")
                    else:
                        logger.warning(f"Failed to send telemetry for {engine_id}")
                
                # Calculate sleep time to maintain consistent interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.simulation_interval - elapsed_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Log status every 30 seconds
                if self.engine_states['ENG001']['cycle'] % (30 // self.simulation_interval) == 0:
                    logger.info(f"Simulation running - Cycle: {self.engine_states['ENG001']['cycle']}")
                    
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self.stop_simulation()
    
    def stop_simulation(self):
        """Stop the simulation and cleanup resources"""
        logger.info("Stopping simulation...")
        self.running = False
        
        if self.producer:
            try:
                self.producer.flush()
                self.producer.close()
                logger.info("Kafka producer closed successfully")
            except Exception as e:
                logger.error(f"Error closing Kafka producer: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_simulation()
        sys.exit(0)

def main():
    """Main function to run the simulator"""
    simulator = EngineDataSimulator()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, simulator.signal_handler)
    signal.signal(signal.SIGTERM, simulator.signal_handler)
    
    # Start simulation
    simulator.start_simulation()

if __name__ == "__main__":
    main()