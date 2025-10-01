"""
Statistical Models Module

This module provides statistical modeling functionality for climate disaster risk prediction,
including feature preparation, model training, and impact prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ClimateRiskModel:
    """
    Handles statistical modeling for climate disaster risk prediction.

    Trains and evaluates multiple models for different impact types (deaths, injuries, damage)
    and selects the best performing model for each impact type.
    """

    def __init__(self, disaster_category: str):
        """Initialize the climate risk model for a specific disaster category."""
        self.category = disaster_category
        self.models = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling with enhanced feature engineering."""
        # Base features for all categories
        base_features = [
            'year', 'YEAR', 'BEGIN_DAY', 'STATE_FIPS', 'MAGNITUDE', 'TOR_F_SCALE'
        ]

        # Add category-specific features
        if self.category.lower() in ['tornado', 'severe_weather']:
            base_features.extend(['TOR_LENGTH', 'TOR_WIDTH'])
        elif self.category.lower() in ['flooding', 'flood']:
            base_features.append('FLOOD_CAUSE')

        # Enhanced features from preprocessing
        enhanced_features = [
            'month', 'quarter', 'day_of_year', 'is_weekend',
            'is_spring', 'is_summer', 'is_fall', 'is_winter',
            'years_since_1950', 'year_squared', 'year_cubed',
            'year_month', 'year_season',
            'duration_hours', 'path_length_km',
            'geo_cluster', 'lat_rounded', 'lon_rounded'
        ]

        # Historical pattern features
        historical_features = [
            'deaths_direct_3yr_avg', 'deaths_direct_5yr_avg', 'deaths_direct_10yr_avg',
            'injuries_direct_3yr_avg', 'injuries_direct_5yr_avg', 'injuries_direct_10yr_avg',
            'damage_property_3yr_avg', 'damage_property_5yr_avg', 'damage_property_10yr_avg',
            'damage_crops_3yr_avg', 'damage_crops_5yr_avg', 'damage_crops_10yr_avg',
            'state_total_damage', 'state_event_count', 'state_avg_damage'
        ]

        # Combine all feature sets
        all_features = base_features + enhanced_features + historical_features

        # Filter features that exist in the dataset
        available_features = [col for col in all_features if col in df.columns]

        print(f"📊 Preparing {len(available_features)} features for {self.category}")

        # Encode categorical variables
        le = LabelEncoder()
        for col in available_features:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = le.fit_transform(df[col].astype(str))

        # Fill any remaining NaN values
        result = df[available_features].fillna(0)

        print(f"✅ Feature preparation complete: {result.shape}")
        return result

    def train_models(self, X: pd.DataFrame, y: Dict[str, pd.Series]) -> Dict[str, object]:
        """Train separate models for each impact type with optimized hyperparameters."""
        models = {}

        for impact_type, target in y.items():
            print(f"Training optimized {impact_type} model for {self.category}...")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, target, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Target-specific model selection and hyperparameters
            if 'injuries' in impact_type.lower():
                # Injuries need special handling - often sparse and different patterns
                models_to_try = [
                    ('RandomForest', RandomForestRegressor(
                        n_estimators=200, max_depth=10, min_samples_split=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1
                    )),
                    ('GradientBoosting', GradientBoostingRegressor(
                        n_estimators=150, max_depth=6, learning_rate=0.1,
                        subsample=0.8, random_state=42
                    )),
                    ('Ridge', Ridge(alpha=1.0, random_state=42))
                ]
            elif 'damage' in impact_type.lower():
                # Damage models - optimize for high performance
                models_to_try = [
                    ('RandomForest', RandomForestRegressor(
                        n_estimators=300, max_depth=15, min_samples_split=3,
                        min_samples_leaf=1, random_state=42, n_jobs=-1
                    )),
                    ('GradientBoosting', GradientBoostingRegressor(
                        n_estimators=200, max_depth=8, learning_rate=0.05,
                        subsample=0.9, random_state=42
                    )),
                    ('Linear', LinearRegression())
                ]
            else:
                # Deaths and other targets
                models_to_try = [
                    ('RandomForest', RandomForestRegressor(
                        n_estimators=250, max_depth=12, min_samples_split=4,
                        min_samples_leaf=2, random_state=42, n_jobs=-1
                    )),
                    ('GradientBoosting', GradientBoostingRegressor(
                        n_estimators=180, max_depth=7, learning_rate=0.08,
                        subsample=0.85, random_state=42
                    ))
                ]

            # Evaluate models using cross-validation
            best_model = None
            best_score = -np.inf
            model_results = {}

            for model_name, model in models_to_try:
                try:
                    # Use cross-validation for more robust evaluation
                    scores = cross_val_score(model, X_train_scaled, y_train,
                                           cv=3, scoring='r2', n_jobs=-1)

                    avg_score = np.mean(scores)

                    # Train on full training set for final evaluation
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    model_results[model_name] = {
                        'model': model,
                        'cv_r2': avg_score,
                        'test_r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                    }

                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model

                except Exception as e:
                    print(f"   ⚠️  {model_name} failed: {e}")
                    continue

            if best_model is not None:
                # Final training on full dataset
                X_scaled = scaler.fit_transform(X)
                best_model.fit(X_scaled, target)

                # Select best model name based on CV performance
                best_model_name = max(model_results.keys(),
                                    key=lambda x: model_results[x]['cv_r2'])

                models[impact_type] = {
                    'model': best_model,
                    'scaler': scaler,
                    'performance': model_results[best_model_name]
                }

                print(f"✅ Best model for {impact_type}: {best_model_name}")
                print(f"   CV R²: {model_results[best_model_name]['cv_r2']:.4f}")
                print(f"   Test R²: {model_results[best_model_name]['test_r2']:.4f}")
            else:
                print(f"❌ No models succeeded for {impact_type}")

        self.models = models
        return models

    def predict_impact(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict impact values for given features using trained models."""
        predictions = {}

        for impact_type, model_info in self.models.items():
            model = model_info['model']
            scaler = model_info['scaler']

            X_scaled = scaler.transform(X)
            predictions[impact_type] = model.predict(X_scaled)

        return predictions
