import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ClimateRiskModel:
    def __init__(self, disaster_category: str):
        self.category = disaster_category
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling"""
        # Select relevant features for the specific disaster category
        feature_columns = [
            'year', 'month', 'day_of_year', 'is_weekend',
            'is_spring', 'is_summer', 'is_fall', 'is_winter',
            'duration_hours', 'path_length_km',
            'state_fips', 'magnitude', 'tor_f_scale'
        ]
        
        # Add category-specific features
        if self.category == 'tornado':
            feature_columns.extend(['tor_length', 'tor_width'])
        elif self.category == 'flooding':
            feature_columns.append('flood_cause')
        
        # Filter features that exist in the dataset
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in available_features:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df[available_features].fillna(0)

    def train_models(self, X: pd.DataFrame, y: Dict[str, pd.Series]) -> Dict[str, object]:
        """Train separate models for each impact type"""
        models = {}
        
        for impact_type, target in y.items():
            print(f"Training {impact_type} model for {self.category}...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models and select best performer
            model_results = {}
            
            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            lr_pred = lr.predict(X_test_scaled)
            model_results['Linear'] = {
                'model': lr,
                'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
                'r2': r2_score(y_test, lr_pred)
            }
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            model_results['RandomForest'] = {
                'model': rf,
                'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'r2': r2_score(y_test, rf_pred)
            }
            
            # Select best model based on R²
            best_model_name = max(model_results.keys(),
                                key=lambda x: model_results[x]['r2'])
            best_model = model_results[best_model_name]['model']
            
            models[impact_type] = {
                'model': best_model,
                'scaler': scaler,
                'performance': model_results[best_model_name]
            }
            
            print(f"Best model for {impact_type}: {best_model_name}")
            print(f"R² Score: {model_results[best_model_name]['r2']:.4f}")
        
        return models

    def predict_impact(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict impact for given features"""
        predictions = {}
        
        for impact_type, model_info in self.models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            X_scaled = scaler.transform(X)
            predictions[impact_type] = model.predict(X_scaled)
        
        return predictions
