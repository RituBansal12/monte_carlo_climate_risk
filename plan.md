# Monte Carlo Climate Risk Prediction System - Implementation Plan

## Project Overview

This project implements a Monte Carlo simulation system to predict the probability and impact of climate-related disasters on human life, property, and infrastructure. The system will use historical NOAA storm events data and climate data to model risk scenarios for different disaster categories across US states.

### Key Objectives

1. **Risk Prediction**: Predict probabilities of deaths, injuries, property damage, and crop damage for different climate disaster categories
2. **State-Level Analysis**: Provide state-specific risk assessments and comparative analysis
3. **Category-Specific Models**: Develop separate prediction models for different disaster types (tornadoes, floods, hurricanes, etc.)
4. **Monte Carlo Simulation**: Use probabilistic modeling to generate risk scenarios and confidence intervals
5. **Exploratory Data Analysis**: Comprehensive EDA to understand patterns, trends, and relationships in the data

## Data Architecture

### Existing Datasets

1. **SED_details_1950-2025.parquet** - Primary storm events data with impact metrics
2. **SED_locations_1950-2025.parquet** - Geographic location data for storm events
3. **SED_fatalities_1950-2025.parquet** - Detailed fatality information
4. **epiNOAA_cty_scaled_1950_2025_ALL.parquet** - County-level climate data
5. **events-US-1980-2024-Q4.csv** - Billion-dollar weather and climate disaster events

### Data Flow Architecture

```
Raw Data → Data Cleaning → Feature Engineering → EDA → Model Training → Monte Carlo Simulation → Results & Visualization
```

## Implementation Phases

### Phase 1: Project Setup and Data Pipeline

#### 1.1 Environment Setup
```bash
# Create virtual environment
python -m venv climate_risk_env
source climate_risk_env/bin/activate  # On Windows: climate_risk_env\Scripts\activate

# Install core dependencies
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
pip install pyarrow fastparquet geopy folium
pip install streamlit dash  # For web dashboard
```

#### 1.2 Project Structure
```
monte_carlo_climate_risk/
├── data/                          # Raw and processed data
│   ├── raw/                      # Original datasets
│   ├── processed/                # Cleaned and feature-engineered data
│   └── results/                  # Model outputs and simulation results
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   ├── __init__.py
│   │   ├── cleaning.py           # Data cleaning functions
│   │   ├── feature_engineering.py # Feature creation
│   │   └── validation.py         # Data quality checks
│   ├── models/                   # Modeling modules
│   │   ├── __init__.py
│   │   ├── statistical_models.py # Regression and statistical models
│   │   ├── monte_carlo.py        # Monte Carlo simulation engine
│   │   └── risk_assessment.py    # Risk calculation functions
│   ├── visualization/            # Visualization modules
│   │   ├── __init__.py
│   │   ├── eda_plots.py          # Exploratory data analysis plots
│   │   ├── risk_maps.py          # Geographic risk visualization
│   │   └── results_dashboard.py  # Interactive dashboard
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── helpers.py            # General helper functions
│       └── constants.py          # Project constants
├── notebooks/                    # Jupyter notebooks for analysis
├── config/                       # Configuration files
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── main.py                       # Main execution script
```

#### 1.3 Data Pipeline Implementation

**File: `src/data/cleaning.py`**
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

class DataCleaner:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def clean_sed_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean SED details dataset with comprehensive preprocessing"""
        # Handle missing values
        numeric_columns = ['injuries_direct', 'injuries_indirect',
                          'deaths_direct', 'deaths_indirect',
                          'damage_property', 'damage_crops']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)

        # Convert damage to numeric values (handle K, M, B suffixes)
        df = self._convert_damage_to_numeric(df)

        # Standardize date formats
        df['begin_date_time'] = pd.to_datetime(df['begin_date_time'],
                                              format='%m/%d/%Y %H:%M:%S',
                                              errors='coerce')

        # Validate coordinate data
        df = self._validate_coordinates(df)

        return df

    def _convert_damage_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert damage strings like '10.00K' to numeric values"""
        def convert_damage(value):
            if pd.isna(value) or value == 0:
                return 0.0

            value_str = str(value).upper().replace(',', '')
            multiplier = 1

            if value_str.endswith('K'):
                multiplier = 1000
                value_str = value_str[:-1]
            elif value_str.endswith('M'):
                multiplier = 1000000
                value_str = value_str[:-1]
            elif value_str.endswith('B'):
                multiplier = 1000000000
                value_str = value_str[:-1]

            try:
                return float(value_str) * multiplier
            except ValueError:
                return 0.0

        df['damage_property'] = df['damage_property'].apply(convert_damage)
        df['damage_crops'] = df['damage_crops'].apply(convert_damage)

        return df

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean coordinate data"""
        # Remove invalid coordinates
        df = df[df['begin_lat'].notna() & df['begin_lon'].notna()]
        df = df[df['end_lat'].notna() & df['end_lon'].notna()]

        # Validate coordinate ranges
        df = df[(df['begin_lat'].between(-90, 90)) &
                (df['begin_lon'].between(-180, 180))]

        return df
```

**File: `src/data/feature_engineering.py`**
```python
import pandas as pd
import numpy as np
from typing import Dict, List
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.disaster_categories = {
            'severe_weather': ['Thunderstorm Wind', 'Hail', 'Lightning', 'High Wind'],
            'flooding': ['Flash Flood', 'Flood', 'Coastal Flood'],
            'winter_weather': ['Winter Storm', 'Blizzard', 'Ice Storm', 'Heavy Snow'],
            'tropical': ['Hurricane', 'Tropical Storm', 'Tropical Depression'],
            'tornado': ['Tornado', 'Funnel Cloud'],
            'fire': ['Wildfire'],
            'other': ['Drought', 'Dust Storm', 'Extreme Cold/Wind Chill']
        }

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df['year'] = df['begin_date_time'].dt.year
        df['month'] = df['begin_date_time'].dt.month
        df['quarter'] = df['begin_date_time'].dt.quarter
        df['day_of_year'] = df['begin_date_time'].dt.dayofyear
        df['is_weekend'] = df['begin_date_time'].dt.weekday.isin([5, 6]).astype(int)

        # Seasonal indicators
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

        return df

    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geography-based features"""
        # Calculate event duration
        df['duration_hours'] = (df['end_date_time'] - df['begin_date_time']).dt.total_seconds() / 3600

        # Calculate path length for tornadoes and other path-based events
        df['path_length_km'] = df.apply(
            lambda row: geodesic(
                (row['begin_lat'], row['begin_lon']),
                (row['end_lat'], row['end_lon'])
            ).kilometers if row['begin_lat'] != row['end_lat'] else 0,
            axis=1
        )

        return df

    def create_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create impact severity features"""
        # Total impact scores
        df['total_deaths'] = df['deaths_direct'] + df['deaths_indirect']
        df['total_injuries'] = df['injuries_direct'] + df['injuries_indirect']
        df['total_damage'] = df['damage_property'] + df['damage_crops']

        # Impact severity categories
        df['severity_score'] = (df['total_deaths'] * 10 +
                               df['total_injuries'] * 2 +
                               np.log1p(df['total_damage']) / 1000000)

        # Binary impact indicators
        df['has_fatalities'] = (df['total_deaths'] > 0).astype(int)
        df['has_injuries'] = (df['total_injuries'] > 0).astype(int)
        df['has_damage'] = (df['total_damage'] > 0).astype(int)

        return df

    def categorize_disasters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize events into broader disaster categories"""
        df['disaster_category'] = 'other'

        for category, events in self.disaster_categories.items():
            mask = df['event_type'].isin(events)
            df.loc[mask, 'disaster_category'] = category

        return df
```

### Phase 2: Exploratory Data Analysis (EDA)

#### 2.1 State-Level Analysis

**File: `src/visualization/eda_plots.py`**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class StateEDA:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    def plot_state_risk_heatmap(self, save_path: str = None):
        """Create heatmap of disaster frequency by state"""
        state_risk = self.data_processor.get_state_risk_summary()

        fig = px.choropleth(
            state_risk,
            locations='state_code',
            locationmode="USA-states",
            color='total_events',
            color_continuous_scale="Reds",
            scope="usa",
            title="Disaster Frequency by State (1950-2025)",
            labels={'total_events': 'Number of Events'}
        )

        if save_path:
            fig.write_html(save_path)
        return fig

    def plot_state_impact_comparison(self, save_path: str = None):
        """Compare total impact (deaths, injuries, damage) across states"""
        state_impacts = self.data_processor.get_state_impact_summary()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Deaths', 'Total Injuries', 'Property Damage ($M)', 'Crop Damage ($M)'),
            specs=[[{"type": "choropleth"}, {"type": "choropleth"}],
                   [{"type": "choropleth"}, {"type": "choropleth"}]]
        )

        # Add choropleth maps for each impact type
        for i, impact_type in enumerate(['total_deaths', 'total_injuries', 'property_damage', 'crop_damage']):
            row = i // 2 + 1
            col = i % 2 + 1

            fig.add_choropleth(
                locations=state_impacts['state_code'],
                locationmode="USA-states",
                z=state_impacts[impact_type],
                colorscale="Reds",
                showscale=True if i == 0 else False,
                row=row, col=col
            )

        fig.update_layout(title="State-Level Climate Impact Comparison", height=800)
        if save_path:
            fig.write_html(save_path)
        return fig

    def plot_temporal_trends_by_state(self, state_code: str, save_path: str = None):
        """Plot temporal trends for a specific state"""
        state_data = self.data_processor.get_state_temporal_data(state_code)

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Annual Event Count', 'Annual Deaths', 'Annual Damage ($M)'),
            shared_xaxes=True
        )

        # Events over time
        yearly_events = state_data.groupby('year').size()
        fig.add_trace(
            go.Scatter(x=yearly_events.index, y=yearly_events.values, mode='lines+markers'),
            row=1, col=1
        )

        # Deaths over time
        yearly_deaths = state_data.groupby('year')['total_deaths'].sum()
        fig.add_trace(
            go.Scatter(x=yearly_deaths.index, y=yearly_deaths.values, mode='lines+markers'),
            row=2, col=1
        )

        # Damage over time
        yearly_damage = state_data.groupby('year')['total_damage'].sum() / 1e6
        fig.add_trace(
            go.Scatter(x=yearly_damage.index, y=yearly_damage.values, mode='lines+markers'),
            row=3, col=1
        )

        fig.update_layout(title=f"Temporal Trends - {state_code}", height=600)
        if save_path:
            fig.write_html(save_path)
        return fig
```

#### 2.2 Disaster Category Analysis

**File: `src/visualization/category_analysis.py`**
```python
class CategoryEDA:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    def plot_category_impact_distribution(self, save_path: str = None):
        """Analyze impact distribution across disaster categories"""
        category_impacts = self.data_processor.get_category_impact_summary()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Deaths by Category', 'Average Injuries by Category',
                          'Average Property Damage by Category', 'Average Crop Damage by Category')
        )

        categories = category_impacts.index.tolist()

        # Deaths
        fig.add_trace(
            go.Bar(x=categories, y=category_impacts['avg_deaths'], name='Deaths'),
            row=1, col=1
        )

        # Injuries
        fig.add_trace(
            go.Bar(x=categories, y=category_impacts['avg_injuries'], name='Injuries'),
            row=1, col=2
        )

        # Property Damage
        fig.add_trace(
            go.Bar(x=categories, y=category_impacts['avg_property_damage'], name='Property Damage'),
            row=2, col=1
        )

        # Crop Damage
        fig.add_trace(
            go.Bar(x=categories, y=category_impacts['avg_crop_damage'], name='Crop Damage'),
            row=2, col=2
        )

        fig.update_layout(title="Impact Distribution by Disaster Category", height=800)
        if save_path:
            fig.write_html(save_path)
        return fig

    def plot_seasonal_patterns_by_category(self, save_path: str = None):
        """Analyze seasonal patterns for each disaster category"""
        seasonal_data = self.data_processor.get_seasonal_patterns()

        fig = px.line(
            seasonal_data,
            x='month',
            y='event_count',
            color='disaster_category',
            facet_col='disaster_category',
            facet_col_wrap=2,
            title="Seasonal Patterns by Disaster Category"
        )

        if save_path:
            fig.write_html(save_path)
        return fig
```

### Phase 3: Statistical Modeling and Monte Carlo Simulation

#### 3.1 Statistical Models by Category

**File: `src/models/statistical_models.py`**
```python
import pandas as pd
import numpy as np
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
            print(f"R² Score: {model_results[best_model_name]['r2']".4f"}")

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
```

#### 3.2 Monte Carlo Simulation Engine

**File: `src/models/monte_carlo.py`**
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.simulation_results = {}

    def run_climate_risk_simulation(self,
                                  state_code: str,
                                  disaster_category: str,
                                  time_horizon: int,
                                  model: ClimateRiskModel) -> Dict[str, pd.DataFrame]:
        """
        Run Monte Carlo simulation for climate risk prediction

        Parameters:
        - state_code: Two-letter state code
        - disaster_category: Type of disaster to simulate
        - time_horizon: Number of years to simulate
        - model: Trained ClimateRiskModel for the category
        """

        print(f"Running Monte Carlo simulation for {disaster_category} in {state_code}")

        # Generate future scenarios
        future_scenarios = self._generate_future_scenarios(
            state_code, disaster_category, time_horizon
        )

        # Run simulations
        simulation_results = {}

        for year in range(1, time_horizon + 1):
            year_scenarios = future_scenarios[future_scenarios['year'] == year]

            if len(year_scenarios) == 0:
                continue

            # Prepare features for prediction
            X = model.prepare_features(year_scenarios)

            # Run Monte Carlo iterations
            yearly_results = self._run_yearly_simulation(X, model, year)

            simulation_results[f'year_{year}'] = yearly_results

        return simulation_results

    def _generate_future_scenarios(self,
                                 state_code: str,
                                 disaster_category: str,
                                 time_horizon: int) -> pd.DataFrame:
        """Generate future climate scenarios based on historical patterns"""

        # Get historical data for the state and category
        historical_data = self._get_historical_scenarios(state_code, disaster_category)

        scenarios = []

        for year in range(1, time_horizon + 1):
            # Sample from historical distribution for each month
            for month in range(1, 13):
                # Get historical events for this month
                month_data = historical_data[historical_data['month'] == month]

                if len(month_data) == 0:
                    # Skip months with no historical events
                    continue

                # Sample number of events from historical distribution
                n_events = np.random.poisson(month_data.groupby('year').size().mean())

                for event in range(n_events):
                    # Sample event characteristics from historical data
                    scenario = self._sample_event_scenario(month_data)
                    scenario['year'] = year
                    scenario['month'] = month
                    scenario['simulated'] = True

                    scenarios.append(scenario)

        return pd.DataFrame(scenarios)

    def _sample_event_scenario(self, month_data: pd.DataFrame) -> Dict:
        """Sample a single event scenario from historical data"""
        scenario = {}

        # Sample from each feature's distribution
        for column in month_data.columns:
            if column not in ['year', 'month', 'simulated']:
                if month_data[column].dtype in ['int64', 'float64']:
                    # Sample from normal distribution fitted to historical data
                    mu = month_data[column].mean()
                    sigma = month_data[column].std()

                    if sigma > 0:
                        scenario[column] = np.random.normal(mu, sigma)
                    else:
                        scenario[column] = mu
                else:
                    # Sample from categorical distribution
                    scenario[column] = np.random.choice(month_data[column].unique())

        return scenario

    def _run_yearly_simulation(self,
                             X: pd.DataFrame,
                             model: ClimateRiskModel,
                             year: int) -> pd.DataFrame:
        """Run Monte Carlo simulation for a single year"""

        results = []

        for sim in range(self.n_simulations):
            # Add random noise to features to simulate variability
            X_noisy = X.copy()

            # Add small random perturbations to numeric features
            for col in X_noisy.columns:
                if X_noisy[col].dtype in ['int64', 'float64']:
                    noise_level = X_noisy[col].std() * 0.1  # 10% noise
                    noise = np.random.normal(0, noise_level, len(X_noisy))
                    X_noisy[col] += noise

            # Predict impacts for this simulation
            predictions = model.predict_impact(X_noisy)

            # Store results
            sim_result = {'simulation': sim, 'year': year}
            for impact_type, values in predictions.items():
                sim_result[f'{impact_type}_predicted'] = values.sum()

            results.append(sim_result)

        return pd.DataFrame(results)

    def calculate_risk_metrics(self, simulation_results: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate risk metrics from simulation results"""

        all_results = []
        for year_key, year_df in simulation_results.items():
            year_df['year'] = int(year_key.split('_')[1])
            all_results.append(year_df)

        combined_results = pd.concat(all_results, ignore_index=True)

        risk_metrics = {}

        impact_types = ['deaths', 'injuries', 'property_damage', 'crop_damage']

        for impact in impact_types:
            col_name = f'{impact}_predicted'

            if col_name in combined_results.columns:
                values = combined_results[col_name]

                risk_metrics[impact] = {
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'p5': values.quantile(0.05),
                    'p25': values.quantile(0.25),
                    'p75': values.quantile(0.75),
                    'p95': values.quantile(0.95),
                    'probability_positive': (values > 0).mean()
                }

        return risk_metrics

    def plot_risk_distributions(self, simulation_results: Dict[str, pd.DataFrame],
                              save_path: str = None):
        """Plot risk distributions from simulation results"""

        all_results = []
        for year_key, year_df in simulation_results.items():
            year_df['year'] = int(year_key.split('_')[1])
            all_results.append(year_df)

        combined_results = pd.concat(all_results, ignore_index=True)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Deaths Distribution', 'Injuries Distribution',
                          'Property Damage Distribution', 'Crop Damage Distribution')
        )

        impact_types = ['deaths', 'injuries', 'property_damage', 'crop_damage']
        colors = ['blue', 'orange', 'green', 'red']

        for i, (impact, color) in enumerate(zip(impact_types, colors)):
            col_name = f'{impact}_predicted'

            if col_name in combined_results.columns:
                row = i // 2 + 1
                col = i % 2 + 1

                fig.add_trace(
                    go.Histogram(
                        x=combined_results[col_name],
                        name=impact.title(),
                        marker_color=color,
                        opacity=0.7
                    ),
                    row=row, col=col
                )

        fig.update_layout(
            title="Monte Carlo Simulation Results - Risk Distributions",
            height=800,
            showlegend=False
        )

        if save_path:
            fig.write_html(save_path)

        return fig
```

### Phase 4: Implementation and Execution Pipeline

#### 4.1 Main Execution Script

**File: `main.py`**
```python
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data.cleaning import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.models.statistical_models import ClimateRiskModel
from src.models.monte_carlo import MonteCarloSimulator
from src.visualization.eda_plots import StateEDA, CategoryEDA

class ClimateRiskPipeline:
    def __init__(self, config_path: str = 'config/pipeline_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.data_cleaner = DataCleaner(self.config.get('cleaning', {}))
        self.feature_engineer = FeatureEngineer()
        self.monte_carlo = MonteCarloSimulator(
            n_simulations=self.config.get('monte_carlo', {}).get('n_simulations', 10000)
        )

    def run_full_pipeline(self):
        """Execute the complete climate risk analysis pipeline"""

        print("Starting Climate Risk Analysis Pipeline...")
        start_time = datetime.now()

        # Phase 1: Data Loading and Cleaning
        print("Phase 1: Loading and cleaning data...")
        data = self._load_and_clean_data()

        # Phase 2: Feature Engineering
        print("Phase 2: Engineering features...")
        data = self._engineer_features(data)

        # Phase 3: Exploratory Data Analysis
        print("Phase 3: Performing exploratory data analysis...")
        self._perform_eda(data)

        # Phase 4: Model Training
        print("Phase 4: Training statistical models...")
        models = self._train_models(data)

        # Phase 5: Monte Carlo Simulation
        print("Phase 5: Running Monte Carlo simulations...")
        simulation_results = self._run_simulations(models)

        # Phase 6: Results Analysis and Visualization
        print("Phase 6: Analyzing results and creating visualizations...")
        self._analyze_results(simulation_results)

        end_time = datetime.now()
        print(f"Pipeline completed in {end_time - start_time}")

        return {
            'data': data,
            'models': models,
            'simulation_results': simulation_results
        }

    def _load_and_clean_data(self) -> Dict[str, pd.DataFrame]:
        """Load and clean all datasets"""
        data = {}

        # Load SED details (main dataset)
        print("Loading SED details data...")
        data['sed_details'] = pd.read_parquet('data/SED_details_1950-2025.parquet')
        data['sed_details'] = self.data_cleaner.clean_sed_details(data['sed_details'])

        # Load other datasets
        data['sed_locations'] = pd.read_parquet('data/SED_locations_1950-2025.parquet')
        data['sed_fatalities'] = pd.read_parquet('data/SED_fatalities_1950-2025.parquet')
        data['climate_data'] = pd.read_parquet('data/epiNOAA_cty_scaled_1950_2025_ALL.parquet')
        data['billion_dollar_events'] = pd.read_csv('data/events-US-1980-2024-Q4.csv')

        return data

    def _engineer_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Engineer features for modeling"""
        # Engineer features for main SED details dataset
        data['sed_details'] = self.feature_engineer.create_temporal_features(data['sed_details'])
        data['sed_details'] = self.feature_engineer.create_geographic_features(data['sed_details'])
        data['sed_details'] = self.feature_engineer.create_impact_features(data['sed_details'])
        data['sed_details'] = self.feature_engineer.categorize_disasters(data['sed_details'])

        return data

    def _perform_eda(self, data: Dict[str, pd.DataFrame]):
        """Perform exploratory data analysis"""
        eda = StateEDA(data)
        category_eda = CategoryEDA(data)

        # Create state-level visualizations
        eda.plot_state_risk_heatmap('results/state_risk_heatmap.html')
        eda.plot_state_impact_comparison('results/state_impact_comparison.html')

        # Create category-level visualizations
        category_eda.plot_category_impact_distribution('results/category_impact_distribution.html')
        category_eda.plot_seasonal_patterns_by_category('results/seasonal_patterns.html')

    def _train_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, ClimateRiskModel]:
        """Train models for each disaster category"""
        models = {}

        # Get unique disaster categories
        categories = data['sed_details']['disaster_category'].unique()

        for category in categories:
            print(f"Training model for {category} category...")
            model = ClimateRiskModel(category)

            # Filter data for this category
            category_data = data['sed_details'][data['sed_details']['disaster_category'] == category]

            # Prepare features
            X = model.prepare_features(category_data)

            # Define target variables
            y = {
                'deaths': category_data['total_deaths'],
                'injuries': category_data['total_injuries'],
                'property_damage': category_data['damage_property'],
                'crop_damage': category_data['damage_crops']
            }

            # Train models
            models[category] = model.train_models(X, y)

        return models

    def _run_simulations(self, models: Dict[str, ClimateRiskModel]) -> Dict:
        """Run Monte Carlo simulations for key states and categories"""
        simulation_results = {}

        # Define simulation scenarios
        scenarios = [
            {'state': 'TX', 'category': 'tornado', 'horizon': 10},
            {'state': 'FL', 'category': 'hurricane', 'horizon': 10},
            {'state': 'CA', 'category': 'wildfire', 'horizon': 10},
            {'state': 'NY', 'category': 'severe_weather', 'horizon': 10},
            {'state': 'LA', 'category': 'flooding', 'horizon': 10}
        ]

        for scenario in scenarios:
            state = scenario['state']
            category = scenario['category']
            horizon = scenario['horizon']

            if category in models:
                results = self.monte_carlo.run_climate_risk_simulation(
                    state, category, horizon, models[category]
                )
                simulation_results[f'{state}_{category}'] = results

        return simulation_results

    def _analyze_results(self, simulation_results: Dict):
        """Analyze simulation results and create final visualizations"""
        # Calculate risk metrics for each simulation
        risk_assessments = {}

        for scenario_key, results in simulation_results.items():
            risk_metrics = self.monte_carlo.calculate_risk_metrics(results)
            risk_assessments[scenario_key] = risk_metrics

            # Create risk distribution plots
            self.monte_carlo.plot_risk_distributions(
                results,
                f'results/risk_distributions_{scenario_key}.html'
            )

        # Save risk assessments
        with open('results/risk_assessments.json', 'w') as f:
            json.dump(risk_assessments, f, indent=2, default=str)

        # Create summary report
        self._create_summary_report(risk_assessments)

    def _create_summary_report(self, risk_assessments: Dict):
        """Create a comprehensive summary report"""
        report = {
            'execution_date': datetime.now().isoformat(),
            'simulation_parameters': self.config.get('monte_carlo', {}),
            'risk_assessments': risk_assessments,
            'summary_statistics': self._calculate_summary_statistics(risk_assessments)
        }

        with open('results/summary_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def _calculate_summary_statistics(self, risk_assessments: Dict) -> Dict:
        """Calculate summary statistics across all scenarios"""
        # Implementation for calculating aggregate statistics
        return {}

if __name__ == "__main__":
    pipeline = ClimateRiskPipeline()
    results = pipeline.run_full_pipeline()
```

### Phase 5: Configuration and Deployment

#### 5.1 Configuration Files

**File: `config/pipeline_config.json`**
```json
{
  "data": {
    "raw_data_path": "data/",
    "processed_data_path": "data/processed/",
    "results_path": "results/"
  },
  "cleaning": {
    "remove_outliers": true,
    "outlier_threshold": 3.0,
    "fill_missing_numeric": "median",
    "coordinate_validation": true
  },
  "features": {
    "temporal_features": true,
    "geographic_features": true,
    "impact_features": true,
    "seasonal_indicators": true
  },
  "modeling": {
    "test_size": 0.2,
    "random_state": 42,
    "cross_validation_folds": 5
  },
  "monte_carlo": {
    "n_simulations": 10000,
    "random_seed": 42,
    "confidence_levels": [0.05, 0.25, 0.75, 0.95],
    "time_horizon": 10
  },
  "visualization": {
    "plot_style": "plotly",
    "color_scheme": "viridis",
    "interactive_plots": true,
    "save_formats": ["html", "png"]
  }
}
```

### Phase 6: Testing and Validation

#### 6.1 Unit Tests

**File: `tests/test_data_cleaning.py`**
```python
import pytest
import pandas as pd
import numpy as np
from src.data.cleaning import DataCleaner

class TestDataCleaner:
    def test_damage_conversion(self):
        """Test damage string to numeric conversion"""
        cleaner = DataCleaner({})

        test_data = pd.DataFrame({
            'damage_property': ['10.00K', '5.50M', '1.25B', '0.00K'],
            'damage_crops': ['100.00K', '2.00M', '0', None]
        })

        result = cleaner._convert_damage_to_numeric(test_data)

        expected_property = [10000.0, 5500000.0, 1250000000.0, 0.0]
        expected_crops = [100000.0, 2000000.0, 0.0, 0.0]

        assert result['damage_property'].tolist() == expected_property
        assert result['damage_crops'].tolist() == expected_crops
```

### Phase 7: Documentation and Deployment

#### 7.1 Usage Instructions

1. **Setup**: Install dependencies and activate virtual environment
2. **Configuration**: Modify `config/pipeline_config.json` for your specific needs
3. **Execution**: Run `python main.py` to execute the full pipeline
4. **Results**: View results in the `results/` directory

#### 7.2 Key Deliverables

- **Interactive Dashboards**: HTML files with interactive visualizations
- **Risk Assessment Reports**: JSON files with detailed risk metrics
- **Statistical Models**: Trained models for each disaster category
- **Monte Carlo Results**: Probability distributions for future risk scenarios
- **Documentation**: Comprehensive documentation of methodology and results

This plan provides a complete roadmap for implementing a sophisticated Monte Carlo climate risk prediction system with extensive exploratory data analysis and state-specific risk assessments.
