# Model Evaluation and Validation Script

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import os

class ModelEvaluator:
    def __init__(self, models_path='/app/models'):
        self.models_path = models_path
        self.rul_model = None
        self.anomaly_model = None
        self.scaler = None
        self.metadata = None
        
    def load_models(self):
        """Load trained models"""
        try:
            self.rul_model = joblib.load(os.path.join(self.models_path, 'rul_model.pkl'))
            self.anomaly_model = joblib.load(os.path.join(self.models_path, 'anomaly_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.models_path, 'scaler.pkl'))
            
            with open(os.path.join(self.models_path, 'metadata.json'), 'r') as f:
                self.metadata = json.load(f)
            
            print("Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def evaluate_rul_model(self, X_test, y_test):
        """Evaluate RUL prediction model"""
        if not self.rul_model or not self.scaler:
            print("Models not loaded")
            return None
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.rul_model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        max_error = np.max(np.abs(y_test - y_pred))
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'max_error': max_error,
            'predictions': y_pred.tolist(),
            'actual': y_test.tolist()
        }
        
        print(f"RUL Model Evaluation Results:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAPE: {mape:.1f}%")
        print(f"  Max Error: {max_error:.2f}")
        
        return metrics
    
    def evaluate_anomaly_model(self, X_test, y_anomaly=None):
        """Evaluate anomaly detection model"""
        if not self.anomaly_model or not self.scaler:
            print("Models not loaded")
            return None
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get anomaly scores and predictions
        anomaly_scores = self.anomaly_model.decision_function(X_test_scaled)
        anomaly_predictions = self.anomaly_model.predict(X_test_scaled)
        
        # Convert predictions to boolean (True = anomaly)
        is_anomaly = anomaly_predictions == -1
        
        normal_count = np.sum(~is_anomaly)
        anomaly_count = np.sum(is_anomaly)
        
        results = {
            'anomaly_scores': anomaly_scores.tolist(),
            'is_anomaly': is_anomaly.tolist(),
            'normal_count': int(normal_count),
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_count / len(is_anomaly))
        }
        
        print(f"Anomaly Detection Results:")
        print(f"  Normal samples: {normal_count}")
        print(f"  Anomalous samples: {anomaly_count}")
        print(f"  Anomaly rate: {results['anomaly_rate']:.1%}")
        
        return results
    
    def generate_feature_importance(self):
        """Generate feature importance analysis"""
        if not self.rul_model:
            print("RUL model not loaded")
            return None
        
        if hasattr(self.rul_model, 'feature_importances_'):
            feature_names = self.metadata.get('feature_columns', [])
            importances = self.rul_model.feature_importances_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            return importance_df
        else:
            print("Model does not support feature importance")
            return None
    
    def create_evaluation_report(self, rul_metrics, anomaly_results, feature_importance):
        """Create comprehensive evaluation report"""
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_metadata': self.metadata,
            'rul_model_evaluation': rul_metrics,
            'anomaly_model_evaluation': anomaly_results,
            'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else None,
            'evaluation_summary': {
                'rul_model_acceptable': rul_metrics['r2'] > 0.7 if rul_metrics else False,
                'anomaly_detection_reasonable': 0.05 < anomaly_results['anomaly_rate'] < 0.2 if anomaly_results else False
            }
        }
        
        # Save report
        report_path = os.path.join(self.models_path, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {report_path}")
        return report

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    
    if not evaluator.load_models():
        print("Failed to load models")
        return
    
    print("Model evaluation completed. Use this script with test data for full evaluation.")

if __name__ == "__main__":
    main()