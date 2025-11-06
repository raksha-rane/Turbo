-- Initialize database schema for Aircraft Engine Monitoring System

-- Create engines table
CREATE TABLE IF NOT EXISTS engines (
    id SERIAL PRIMARY KEY,
    engine_id VARCHAR(50) UNIQUE NOT NULL,
    model VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create telemetry data table
CREATE TABLE IF NOT EXISTS telemetry (
    id SERIAL PRIMARY KEY,
    engine_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    cycle INTEGER NOT NULL,
    operational_setting_1 FLOAT,
    operational_setting_2 FLOAT,
    operational_setting_3 FLOAT,
    sensor_1 FLOAT,
    sensor_2 FLOAT,
    sensor_3 FLOAT,
    sensor_4 FLOAT,
    sensor_5 FLOAT,
    sensor_6 FLOAT,
    sensor_7 FLOAT,
    sensor_8 FLOAT,
    sensor_9 FLOAT,
    sensor_10 FLOAT,
    sensor_11 FLOAT,
    sensor_12 FLOAT,
    sensor_13 FLOAT,
    sensor_14 FLOAT,
    sensor_15 FLOAT,
    sensor_16 FLOAT,
    sensor_17 FLOAT,
    sensor_18 FLOAT,
    sensor_19 FLOAT,
    sensor_20 FLOAT,
    sensor_21 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    engine_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    predicted_rul INTEGER,
    confidence FLOAT,
    anomaly_score FLOAT,
    is_anomaly BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    engine_id VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT,
    timestamp TIMESTAMP NOT NULL,
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (engine_id) REFERENCES engines(engine_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_telemetry_engine_timestamp ON telemetry(engine_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_engine_timestamp ON predictions(engine_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_engine_timestamp ON alerts(engine_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);

-- Insert sample engines
INSERT INTO engines (engine_id, model) VALUES 
    ('ENG001', 'CFM56-7B'),
    ('ENG002', 'CFM56-7B'),
    ('ENG003', 'CFM56-7B'),
    ('ENG004', 'CFM56-7B'),
    ('ENG005', 'CFM56-7B'),
    ('ENG006', 'CFM56-7B'),
    ('ENG007', 'CFM56-7B'),
    ('ENG008', 'CFM56-7B'),
    ('ENG009', 'CFM56-7B'),
    ('ENG010', 'CFM56-7B')
ON CONFLICT (engine_id) DO NOTHING;