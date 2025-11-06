# Models Directory

This directory contains trained machine learning models and related configuration files for the Aircraft Engine Monitoring System.

## Files Overview

### Trained Models (Generated during training)
- `rul_model.pkl` - Random Forest model for Remaining Useful Life prediction
- `anomaly_model.pkl` - Isolation Forest model for anomaly detection  
- `scaler.pkl` - StandardScaler for feature normalization
- `metadata.json` - Model training metadata and performance metrics
- `evaluation_report.json` - Comprehensive model evaluation results

### Configuration Files
- `model_config.json` - Model parameters and configuration settings
- `train_models.py` - Model training script
- `evaluate_models.py` - Model evaluation and validation script

## Model Details

### RUL Prediction Model
- **Algorithm**: Random Forest Regressor
- **Purpose**: Predict remaining useful life in operational cycles
- **Input**: Engineered features from sensor telemetry
- **Output**: Predicted RUL (integer cycles) with confidence score

### Anomaly Detection Model  
- **Algorithm**: Isolation Forest
- **Purpose**: Detect abnormal engine behavior patterns
- **Input**: Same engineered features as RUL model
- **Output**: Anomaly score and binary anomaly flag

## Feature Engineering

The models use the following types of features:

1. **Raw Sensor Data** (21 sensors + 3 operational settings)
   - Temperature sensors (sensor_2, sensor_3, sensor_4, sensor_11, sensor_12)
   - Pressure sensors (sensor_1, sensor_5, sensor_6)
   - Efficiency sensors (sensor_7, sensor_9, sensor_14)
   - Performance sensors (sensor_8, sensor_13, sensor_15-21)

2. **Rolling Statistics** (window size: 5)
   - Rolling mean for trend analysis
   - Rolling standard deviation for variability

3. **Trend Features** (window size: 10)
   - Deviation from longer-term rolling mean
   - Captures degradation patterns

## Training Process

1. **Data Loading**: NASA C-MAPSS dataset or synthetic data generation
2. **Feature Engineering**: Calculate rolling statistics and trends
3. **Data Preparation**: Create RUL labels and feature matrices
4. **Model Training**: Train RUL and anomaly models with hyperparameter tuning
5. **Validation**: Evaluate model performance on held-out data
6. **Persistence**: Save models, scaler, and metadata

## Usage

### Training New Models
```bash
# From the models directory
python train_models.py
```

### Evaluating Models
```bash
# From the models directory  
python evaluate_models.py
```

### Loading Models in Application
```python
import joblib
import json

# Load models
rul_model = joblib.load('rul_model.pkl')
anomaly_model = joblib.load('anomaly_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load metadata
with open('metadata.json', 'r') as f:
    metadata = json.load(f)
```

## Model Performance Targets

### RUL Prediction
- **R² Score**: > 0.7
- **RMSE**: < 50 cycles
- **MAE**: < 30 cycles

### Anomaly Detection
- **Anomaly Rate**: 5-20% of samples
- **False Positive Rate**: < 10%

## Model Updates

Models are automatically retrained:
- **Scheduled**: Every 24 hours via CI/CD pipeline
- **On-demand**: Triggered by poor performance metrics
- **Data-driven**: When new training data becomes available

## Monitoring

Model performance is continuously monitored through:
- **Prediction accuracy**: Comparing predictions with actual RUL
- **Drift detection**: Monitoring feature distributions
- **Anomaly rates**: Tracking anomaly detection patterns
- **Confidence scores**: Monitoring prediction confidence levels

## Troubleshooting

### Common Issues

1. **Models not found**: Run `train_models.py` to generate initial models
2. **Poor performance**: Check training data quality and feature engineering
3. **Memory issues**: Reduce model complexity or batch size
4. **Slow predictions**: Consider model optimization or caching

### Model Validation

Before deploying new models, validate:
- Performance metrics meet thresholds
- Feature importance makes domain sense
- Anomaly detection rates are reasonable
- Prediction distributions are stable