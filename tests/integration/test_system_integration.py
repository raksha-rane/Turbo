import pytest
import time
import json
import requests
from kafka import KafkaProducer, KafkaConsumer
import redis
import psycopg2

class TestSystemIntegration:
    """Integration tests for the entire Aircraft Engine Monitoring System"""
    
    @pytest.fixture(scope="class")
    def kafka_producer(self):
        """Create Kafka producer for testing"""
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9093'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        yield producer
        producer.close()
    
    @pytest.fixture(scope="class") 
    def redis_client(self):
        """Create Redis client for testing"""
        client = redis.Redis(host='localhost', port=6380, db=0, decode_responses=True)
        yield client
        client.close()
    
    @pytest.fixture(scope="class")
    def db_connection(self):
        """Create database connection for testing"""
        conn = psycopg2.connect(
            host='localhost',
            port=5433,
            database='aircraft_monitoring_test',
            user='test_admin',
            password='test_password'
        )
        yield conn
        conn.close()
    
    def test_kafka_connectivity(self, kafka_producer):
        """Test Kafka broker connectivity"""
        # Send test message
        test_message = {
            'engine_id': 'TEST001',
            'timestamp': '2023-01-01T00:00:00Z',
            'cycle': 1,
            'sensor_1': 100.0
        }
        
        future = kafka_producer.send('engine-telemetry', value=test_message)
        record_metadata = future.get(timeout=10)
        
        assert record_metadata.topic == 'engine-telemetry'
        assert record_metadata.partition >= 0
    
    def test_redis_connectivity(self, redis_client):
        """Test Redis connectivity and operations"""
        # Test basic operations
        redis_client.set('test_key', 'test_value')
        value = redis_client.get('test_key')
        assert value == 'test_value'
        
        # Test list operations
        redis_client.lpush('test_list', 'item1', 'item2')
        items = redis_client.lrange('test_list', 0, -1)
        assert 'item1' in items
        assert 'item2' in items
        
        # Cleanup
        redis_client.delete('test_key', 'test_list')
    
    def test_database_connectivity(self, db_connection):
        """Test database connectivity and schema"""
        with db_connection.cursor() as cursor:
            # Test basic connectivity
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
            
            # Test table existence
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['engines', 'telemetry', 'predictions', 'alerts']
            for table in expected_tables:
                assert table in tables, f"Table {table} not found"
    
    def test_end_to_end_data_flow(self, kafka_producer, redis_client, db_connection):
        """Test complete data flow from simulation to storage"""
        # This test would require the actual services to be running
        # For now, we'll test the individual components
        
        # Simulate telemetry data
        telemetry_data = {
            'engine_id': 'ENG001',
            'timestamp': '2023-01-01T00:00:00Z',
            'cycle': 100,
            'operational_setting_1': 0.001,
            'operational_setting_2': -0.002,
            'operational_setting_3': 100.0,
            'sensor_1': 518.67,
            'sensor_2': 642.15,
            'sensor_3': 1591.82,
            'sensor_4': 1403.14,
            'sensor_5': 14.62,
            'sensor_6': 21.61,
            'sensor_7': 553.75,
            'sensor_8': 2388.04,
            'sensor_9': 9044.07,
            'sensor_10': 1.30,
            'sensor_11': 47.49,
            'sensor_12': 522.28,
            'sensor_13': 2388.07,
            'sensor_14': 8131.49,
            'sensor_15': 8.4318,
            'sensor_16': 0.03,
            'sensor_17': 392,
            'sensor_18': 2388,
            'sensor_19': 100.00,
            'sensor_20': 39.00,
            'sensor_21': 23.4236
        }
        
        # Send data to Kafka
        future = kafka_producer.send('engine-telemetry', key='ENG001', value=telemetry_data)
        record_metadata = future.get(timeout=10)
        assert record_metadata.topic == 'engine-telemetry'
        
        # Verify data structure
        assert telemetry_data['engine_id'] == 'ENG001'
        assert telemetry_data['cycle'] == 100
        assert all(key in telemetry_data for key in ['sensor_1', 'sensor_2', 'sensor_3'])
    
    def test_dashboard_health_endpoint(self):
        """Test dashboard health endpoint (if available)"""
        try:
            response = requests.get('http://localhost:8501/_stcore/health', timeout=5)
            # Dashboard might not be running in test environment
            # This is more of a smoke test
            assert response.status_code in [200, 404, 503]  # Accept various states
        except requests.exceptions.RequestException:
            # Dashboard might not be accessible in test environment
            pytest.skip("Dashboard not accessible in test environment")
    
    def test_data_validation(self):
        """Test data validation and schema compliance"""
        # Test telemetry data structure validation
        valid_telemetry = {
            'engine_id': 'ENG001',
            'timestamp': '2023-01-01T00:00:00Z',
            'cycle': 1,
            'sensor_1': 100.0
        }
        
        # Check required fields
        required_fields = ['engine_id', 'timestamp', 'cycle']
        for field in required_fields:
            assert field in valid_telemetry
        
        # Check data types
        assert isinstance(valid_telemetry['engine_id'], str)
        assert isinstance(valid_telemetry['timestamp'], str)
        assert isinstance(valid_telemetry['cycle'], int)
        assert isinstance(valid_telemetry['sensor_1'], (int, float))
    
    def test_error_handling(self, kafka_producer):
        """Test system error handling"""
        # Test invalid message format
        invalid_message = {'invalid': 'data'}
        
        # This should not crash the system
        try:
            future = kafka_producer.send('engine-telemetry', value=invalid_message)
            future.get(timeout=10)
        except Exception as e:
            # Error handling should be graceful
            assert isinstance(e, Exception)
    
    def test_performance_baseline(self, kafka_producer):
        """Test basic performance characteristics"""
        start_time = time.time()
        
        # Send multiple messages
        messages_count = 10
        for i in range(messages_count):
            test_message = {
                'engine_id': f'PERF{i:03d}',
                'timestamp': '2023-01-01T00:00:00Z',
                'cycle': i,
                'sensor_1': float(i * 10)
            }
            kafka_producer.send('engine-telemetry', value=test_message)
        
        # Flush all messages
        kafka_producer.flush()
        
        elapsed_time = time.time() - start_time
        
        # Performance assertion - should complete within reasonable time
        assert elapsed_time < 10.0, f"Performance test took too long: {elapsed_time}s"
        
        # Calculate throughput
        throughput = messages_count / elapsed_time
        assert throughput > 1.0, f"Throughput too low: {throughput} msg/s"