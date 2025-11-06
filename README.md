# Real-Time Aircraft Engine Monitoring & Predictive Maintenance System

A comprehensive, production-ready system for monitoring aircraft engine health in real-time using machine learning for predictive maintenance. Built with modern DevOps practices including containerization, CI/CD pipelines, and automated testing.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data          │    │     Kafka       │    │   ML Service   │
│   Simulator     ├────┤   (KRaft Mode)  ├────┤   (RUL Pred.)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                │                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │◄───┤    Consumer     │    │      Redis      │
│   Dashboard     │    │    Service      │    │     (Cache)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   PostgreSQL    │
                       │   (Database)    │
                       └─────────────────┘
```

## Features

### 🔄 Real-Time Data Processing
- **High-frequency telemetry ingestion** via Apache Kafka (KRaft mode)
- **26 sensor monitoring** based on NASA C-MAPSS dataset structure
- **Real-time data streaming** with configurable intervals

### 🤖 Machine Learning & Analytics
- **RUL (Remaining Useful Life) prediction** using Random Forest algorithms
- **Anomaly detection** with Isolation Forest
- **Automated model retraining** via CI/CD pipelines
- **Confidence scoring** for predictions

### 📊 Interactive Dashboard
- **Real-time visualization** with Streamlit
- **Engine health monitoring** with status indicators
- **Sensor data charts** and trend analysis
- **Alert management** with severity levels
- **System health metrics** and monitoring

### 🚀 DevOps & Automation
- **Containerized microservices** with Docker
- **CI/CD pipeline** with Jenkins
- **Automated testing** (unit, integration, security)
- **Infrastructure as Code** with Docker Compose
- **Health checks** and monitoring

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Git
- 8GB+ RAM recommended
- Jenkins (for CI/CD)

### 1. Clone Repository
```bash
git clone <your-repository-url>
cd aircraft-engine-monitoring
```

### 2. Start the System
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Access Dashboard
- Dashboard: http://localhost:8501
- Kafka: localhost:9092
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### 4. Monitor System Health
```bash
# Check individual service health
docker-compose exec kafka kafka-broker-api-versions.sh --bootstrap-server localhost:9092
docker-compose exec postgres pg_isready -U admin -d aircraft_monitoring
docker-compose exec redis redis-cli ping

# View real-time logs
docker-compose logs -f data-simulator
docker-compose logs -f ml-service
```

## Service Details

### Data Simulator Service
- **Purpose**: Generates realistic engine telemetry data
- **Data Source**: Based on NASA C-MAPSS dataset patterns
- **Output**: 26 sensors + operational settings per engine
- **Frequency**: Configurable (default: 2 seconds)
- **Engines**: 10 simulated engines with individual characteristics

### ML Service
- **Purpose**: Provides RUL predictions and anomaly detection
- **Models**: Random Forest (RUL), Isolation Forest (Anomaly)
- **Features**: Statistical features from sensor time series
- **Training**: Automated on C-MAPSS dataset
- **Output**: Predictions with confidence scores

### Consumer Service
- **Purpose**: Processes and stores all data streams
- **Storage**: PostgreSQL for persistence, Redis for caching
- **Alerts**: Automatic alert generation for anomalies/low RUL
- **Monitoring**: System health metrics collection

### Dashboard Service
- **Purpose**: Real-time visualization and monitoring
- **Technology**: Streamlit with Plotly charts
- **Features**: Engine selection, time range filtering, auto-refresh
- **Alerts**: Real-time alert display with severity levels

## Configuration

### Environment Variables (.env)
```bash
# Kafka Configuration
KAFKA_BROKER=kafka:9092
KAFKA_TELEMETRY_TOPIC=engine-telemetry
KAFKA_PREDICTIONS_TOPIC=engine-predictions

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_DB=aircraft_monitoring
POSTGRES_USER=admin
POSTGRES_PASSWORD=secure_password

# ML Configuration
MODEL_UPDATE_INTERVAL=3600
PREDICTION_THRESHOLD=0.8

# Simulation Configuration
SIMULATION_INTERVAL=2
NUM_ENGINES=10
```

### Scaling Configuration
```bash
# Scale specific services
docker-compose up -d --scale ml-service=3
docker-compose up -d --scale consumer-service=2

# Monitor resource usage
docker stats
```

## Development

### Local Development Setup
```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run code quality checks
black services/
flake8 services/
pylint services/

# Run unit tests
pytest services/ -v --cov=services
```

### Adding New Sensors
1. Update `sensor_columns` in data simulator
2. Modify database schema in `infrastructure/init.sql`
3. Update ML feature extraction
4. Add visualization to dashboard

### Custom ML Models
1. Implement model in `services/ml-service/`
2. Update training pipeline
3. Modify prediction output format
4. Add model evaluation metrics

## CI/CD Pipeline

### Jenkins Pipeline Stages
1. **Code Quality**: Linting, security scans, Dockerfile analysis
2. **Testing**: Unit tests, integration tests, coverage reports
3. **Building**: Docker image creation and tagging
4. **ML Training**: Optional model retraining
5. **Deployment**: Environment-specific deployments
6. **Health Checks**: Post-deployment validation

### Pipeline Configuration
```bash
# Trigger pipeline
# Push to main branch or create pull request

# Manual triggers
curl -X POST https://jenkins.example.com/job/aircraft-monitoring/build

# Parameters
DEPLOY_ENVIRONMENT=production
RETRAIN_MODELS=true
RUN_INTEGRATION_TESTS=true
```

## Monitoring & Alerting

### System Metrics
- **Throughput**: Messages per second
- **Latency**: End-to-end processing time
- **Availability**: Service uptime
- **Errors**: Error rates and types
- **Resources**: CPU, memory, disk usage

### Alert Types
- **Critical**: RUL < 20 cycles, system failures
- **Medium**: RUL < 50 cycles, performance degradation
- **Low**: Minor anomalies, maintenance reminders

### Log Analysis
```bash
# Centralized logging
docker-compose logs --tail=100 -f

# Service-specific logs
docker-compose logs ml-service
docker-compose logs consumer-service

# Error filtering
docker-compose logs | grep ERROR
```

## Production Deployment

### Security Considerations
- Change default passwords in production
- Use secrets management (Kubernetes secrets, Vault)
- Enable TLS/SSL for external connections
- Implement authentication for dashboard
- Regular security updates

### Performance Optimization
- **Kafka**: Increase partitions for higher throughput
- **Database**: Add indexes, connection pooling
- **ML Service**: Model caching, batch predictions
- **Dashboard**: Data pagination, caching

### High Availability
```yaml
# docker-compose.prod.yml
services:
  kafka:
    deploy:
      replicas: 3
  ml-service:
    deploy:
      replicas: 2
  consumer-service:
    deploy:
      replicas: 2
```

## Troubleshooting

### Common Issues

**Services not starting**
```bash
# Check Docker daemon
sudo systemctl status docker

# Check resource usage
docker system df
docker system prune

# Rebuild images
docker-compose build --no-cache
```

**Kafka connection issues**
```bash
# Check Kafka topics
docker exec kafka kafka-topics.sh --bootstrap-server localhost:9092 --list

# Create topics manually
docker exec kafka kafka-topics.sh --bootstrap-server localhost:9092 \
  --create --topic engine-telemetry --partitions 3 --replication-factor 1
```

**Database connection issues**
```bash
# Check database
docker exec postgres psql -U admin -d aircraft_monitoring -c "SELECT 1;"

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

**Dashboard not loading**
```bash
# Check dashboard logs
docker-compose logs dashboard

# Verify dependencies
docker exec dashboard pip list

# Restart dashboard
docker-compose restart dashboard
```

## Data Schema

### Telemetry Data
- **engine_id**: Engine identifier (ENG001-ENG010)
- **timestamp**: UTC timestamp
- **cycle**: Operational cycle number
- **operational_setting_1-3**: Flight conditions
- **sensor_1-21**: Various engine sensors (temperature, pressure, etc.)

### Prediction Data
- **engine_id**: Engine identifier
- **predicted_rul**: Remaining useful life in cycles
- **confidence**: Prediction confidence (0-1)
- **anomaly_score**: Anomaly detection score (0-1)
- **is_anomaly**: Boolean anomaly flag

## API Reference

### Internal APIs
- **Kafka Topics**: `engine-telemetry`, `engine-predictions`, `engine-alerts`
- **Redis Keys**: `latest_telemetry:{engine_id}`, `prediction:{engine_id}`
- **Database Tables**: `engines`, `telemetry`, `predictions`, `alerts`

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards
- Follow PEP 8 for Python code
- Use Black for code formatting
- Add type hints where appropriate
- Write comprehensive tests
- Update documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA C-MAPSS dataset for training data
- Apache Kafka community
- Streamlit team for excellent visualization framework
- Open source contributors

## Support

For questions and support:
- Create GitHub issue for bugs/features
- Check troubleshooting section
- Review system logs for debugging

---

**Built with modern DevOps practices for production reliability and scalability.**