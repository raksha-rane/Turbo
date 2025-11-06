# Model Training Script for Aircraft Engine RUL Prediction
# Based on NASA C-MAPSS Dataset

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, data_path='/app/data'):
        self.data_path = data_path
        self.models_path = '/app/models'
        
        # Ensure models directory exists
        os.makedirs(self.models_path, exist_ok=True)
        
        # Feature columns from C-MAPSS dataset
        self.sensor_columns = [
            'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
            'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
            'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
            'sensor_19', 'sensor_20', 'sensor_21'
        ]
        
        # Models
        self.rul_model = None
        self.anomaly_model = None
        self.scaler = None
        
    def load_cmapss_data(self):
        """Load NASA C-MAPSS dataset"""
        print("Loading NASA C-MAPSS dataset...")
        
        # Try to load real data first
        train_files = ['train_FD001.txt', 'train_FD002.txt', 'train_FD003.txt', 'train_FD004.txt']
        test_files = ['test_FD001.txt', 'test_FD002.txt', 'test_FD003.txt', 'test_FD004.txt']
        rul_files = ['RUL_FD001.txt', 'RUL_FD002.txt', 'RUL_FD003.txt', 'RUL_FD004.txt']
        
        all_train_data = []
        all_test_data = []
        all_rul_data = []
        
        for i, (train_file, test_file, rul_file) in enumerate(zip(train_files, test_files, rul_files)):
            train_path = os.path.join(self.data_path, train_file)
            test_path = os.path.join(self.data_path, test_file)
            rul_path = os.path.join(self.data_path, rul_file)
            
            if os.path.exists(train_path):
                # Load training data
                train_df = pd.read_csv(
                    train_path,
                    sep=' ',
                    header=None,
                    names=['unit_number', 'time_cycles'] + self.sensor_columns,
                    usecols=range(26)
                )
                train_df['dataset'] = f'FD00{i+1}'
                all_train_data.append(train_df)
                
                # Load test data if available
                if os.path.exists(test_path):
                    test_df = pd.read_csv(
                        test_path,
                        sep=' ',
                        header=None,
                        names=['unit_number', 'time_cycles'] + self.sensor_columns,  
                        usecols=range(26)
                    )
                    test_df['dataset'] = f'FD00{i+1}'
                    all_test_data.append(test_df)
                
                # Load RUL data if available
                if os.path.exists(rul_path):
                    rul_df = pd.read_csv(rul_path, header=None, names=['rul'])
                    rul_df['dataset'] = f'FD00{i+1}'
                    all_rul_data.append(rul_df)
        
        if all_train_data:
            train_data = pd.concat(all_train_data, ignore_index=True)
            test_data = pd.concat(all_test_data, ignore_index=True) if all_test_data else None
            rul_data = pd.concat(all_rul_data, ignore_index=True) if all_rul_data else None
            
            print(f"Loaded {len(train_data)} training records from {len(all_train_data)} datasets")
            return train_data, test_data, rul_data
        else:
            print("No C-MAPSS data found, generating synthetic data...")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """Generate synthetic data for model training"""
        print("Generating synthetic training data...")
        np.random.seed(42)
        
        train_data = []
        test_data = []
        rul_data = []
        
        # Generate data for 100 synthetic engines
        for engine_id in range(1, 101):
            max_cycles = np.random.randint(150, 400)
            
            for cycle in range(1, max_cycles + 1):
                # Simulate degradation over time
                degradation_factor = 1 + (cycle / max_cycles) * np.random.uniform(0.05, 0.15)
                noise_factor = np.random.uniform(0.95, 1.05)
                
                # Generate sensor readings with realistic patterns
                row = {
                    'unit_number': engine_id,
                    'time_cycles': cycle,
                    'operational_setting_1': np.random.normal(0, 0.01),
                    'operational_setting_2': np.random.normal(0, 0.01),
                    'operational_setting_3': np.random.normal(100, 5),
                    'sensor_1': np.random.normal(518.67, 2) * degradation_factor * noise_factor,
                    'sensor_2': np.random.normal(642, 5) * degradation_factor * noise_factor,
                    'sensor_3': np.random.normal(1590, 20) * degradation_factor * noise_factor,
                    'sensor_4': np.random.normal(1400, 15) * degradation_factor * noise_factor,
                    'sensor_5': np.random.normal(14.62, 0.1) * noise_factor,
                    'sensor_6': np.random.normal(21.61, 0.2) * noise_factor,
                    'sensor_7': np.random.normal(554, 3) / degradation_factor * noise_factor,
                    'sensor_8': np.random.normal(2388, 10) * noise_factor,
                    'sensor_9': np.random.normal(9050, 50) / degradation_factor * noise_factor,
                    'sensor_10': np.random.normal(1.3, 0.05) * noise_factor,
                    'sensor_11': np.random.normal(47.3, 1) * degradation_factor * noise_factor,
                    'sensor_12': np.random.normal(522, 3) * degradation_factor * noise_factor,
                    'sensor_13': np.random.normal(2388, 10) * noise_factor,
                    'sensor_14': np.random.normal(8130, 40) / degradation_factor * noise_factor,
                    'sensor_15': np.random.normal(8.4, 0.2) * noise_factor,
                    'sensor_16': np.random.normal(0.03, 0.005) * noise_factor,
                    'sensor_17': np.random.normal(392, 5) * noise_factor,
                    'sensor_18': np.random.normal(2388, 10) * noise_factor,
                    'sensor_19': np.random.normal(100, 1) * noise_factor,
                    'sensor_20': np.random.normal(39, 1) * noise_factor,
                    'sensor_21': np.random.normal(23.4, 0.3) * noise_factor,
                    'dataset': 'synthetic'
                }
                train_data.append(row)
            
            # Create test data (last portion of each engine's life)
            test_start = max(1, int(max_cycles * 0.8))
            test_cycles = np.random.randint(test_start, max_cycles)
            
            # Add test record
            test_row = row.copy()
            test_row['time_cycles'] = test_cycles
            test_data.append(test_row)
            
            # RUL for test data
            rul_data.append({'rul': max_cycles - test_cycles, 'dataset': 'synthetic'})
        
        train_df = pd.DataFrame(train_data)  
        test_df = pd.DataFrame(test_data)
        rul_df = pd.DataFrame(rul_data)
        
        print(f"Generated {len(train_df)} training records for {len(set(train_df['unit_number']))} engines")
        return train_df, test_df, rul_df
    
    def engineer_features(self, df):
        """Engineer features from raw sensor data"""
        print("Engineering features...")
        
        features_df = df.copy()
        
        # Group by engine and dataset
        for (engine_id, dataset) in features_df[['unit_number', 'dataset']].drop_duplicates().values:
            mask = (features_df['unit_number'] == engine_id) & (features_df['dataset'] == dataset)
            engine_data = features_df[mask].copy()
            
            if len(engine_data) >= 5:  # Need minimum data for rolling statistics
                for sensor in self.sensor_columns:
                    if sensor in engine_data.columns:
                        # Rolling statistics
                        rolling_mean = engine_data[sensor].rolling(window=5, min_periods=1).mean()
                        rolling_std = engine_data[sensor].rolling(window=5, min_periods=1).std().fillna(0)
                        
                        # Assign back to main dataframe
                        features_df.loc[mask, f'{sensor}_rolling_mean'] = rolling_mean
                        features_df.loc[mask, f'{sensor}_rolling_std'] = rolling_std
                        
                        # Trend features
                        if len(engine_data) >= 10:
                            rolling_mean_10 = engine_data[sensor].rolling(window=10, min_periods=1).mean()
                            features_df.loc[mask, f'{sensor}_trend'] = engine_data[sensor] - rolling_mean_10
        
        return features_df
    
    def prepare_training_data(self, train_df):
        """Prepare data for model training"""
        print("Preparing training data...")
        
        # Engineer features
        train_df = self.engineer_features(train_df)
        
        # Calculate RUL for each record
        rul_data = []
        for (engine_id, dataset) in train_df[['unit_number', 'dataset']].drop_duplicates().values:
            engine_mask = (train_df['unit_number'] == engine_id) & (train_df['dataset'] == dataset)
            engine_data = train_df[engine_mask].copy()
            max_cycle = engine_data['time_cycles'].max()
            
            # Calculate RUL
            engine_data['rul'] = max_cycle - engine_data['time_cycles']
            rul_data.append(engine_data)
        
        final_df = pd.concat(rul_data, ignore_index=True)
        
        # Select feature columns
        feature_cols = [col for col in final_df.columns 
                       if col not in ['unit_number', 'time_cycles', 'rul', 'dataset']]
        
        X = final_df[feature_cols].fillna(0)
        y = final_df['rul']
        
        print(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        return X, y, feature_cols
    
    def train_rul_model(self, X_train, y_train, X_val, y_val):
        """Train RUL prediction model"""
        print("Training RUL prediction model...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        self.rul_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = self.rul_model.predict(X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)  
        r2 = r2_score(y_val, y_pred)
        
        print(f"RUL Model Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  Best parameters: {grid_search.best_params_}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
    
    def train_anomaly_model(self, X_train):
        """Train anomaly detection model"""
        print("Training anomaly detection model...")
        
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train Isolation Forest
        self.anomaly_model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_model.fit(X_train_scaled)
        
        # Evaluate on training data
        anomaly_scores = self.anomaly_model.decision_function(X_train_scaled)
        anomaly_predictions = self.anomaly_model.predict(X_train_scaled)
        
        normal_count = np.sum(anomaly_predictions == 1)
        anomaly_count = np.sum(anomaly_predictions == -1)
        
        print(f"Anomaly Model Training:")
        print(f"  Normal samples: {normal_count}")
        print(f"  Anomalous samples: {anomaly_count}")
        print(f"  Anomaly rate: {anomaly_count / len(anomaly_predictions):.1%}")
    
    def save_models_and_metadata(self, feature_cols, rul_metrics):
        """Save trained models and metadata"""
        print("Saving models and metadata...")
        
        # Save models
        joblib.dump(self.rul_model, os.path.join(self.models_path, 'rul_model.pkl'))
        joblib.dump(self.anomaly_model, os.path.join(self.models_path, 'anomaly_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_path, 'scaler.pkl'))
        
        # Save metadata
        metadata = {
            'model_version': '1.0',
            'trained_at': datetime.now().isoformat(),
            'feature_columns': feature_cols,
            'sensor_columns': self.sensor_columns,
            'rul_model_performance': rul_metrics,
            'model_types': {
                'rul_model': 'RandomForestRegressor',
                'anomaly_model': 'IsolationForest',
                'scaler': 'StandardScaler'
            },
            'training_info': {
                'dataset_source': 'NASA C-MAPSS / Synthetic',
                'feature_engineering': 'rolling_statistics_and_trends',
                'validation_split': 0.2
            }
        }
        
        with open(os.path.join(self.models_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {self.models_path}")
    
    def train_all_models(self):
        """Main training pipeline"""
        print("=" * 60)
        print("AIRCRAFT ENGINE RUL PREDICTION MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        train_df, test_df, rul_df = self.load_cmapss_data()
        
        # Prepare training data
        X, y, feature_cols = self.prepare_training_data(train_df)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Train RUL model
        rul_metrics = self.train_rul_model(X_train, y_train, X_val, y_val)
        
        # Train anomaly model
        self.train_anomaly_model(X_train)
        
        # Save everything
        self.save_models_and_metadata(feature_cols, rul_metrics)
        
        print("=" * 60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return rul_metrics

if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.train_all_models()