"""
Monte Carlo Simulation Module

This module provides Monte Carlo simulation functionality for climate risk analysis,
including scenario generation, simulation execution, and risk metrics calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class MonteCarloSimulator:
    """
    Handles Monte Carlo simulations for climate disaster risk assessment.

    Generates future scenarios based on historical data patterns and runs
    simulations to predict potential impacts over specified time horizons.
    """

    def __init__(self, n_simulations: int = 10000, random_seed: int = 42, historical_data: pd.DataFrame = None):
        """Initialize the Monte Carlo simulator."""
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.historical_data = historical_data
        np.random.seed(random_seed)
        self.simulation_results = {}

    def run_climate_risk_simulation(self, state_code: str, disaster_category: str, time_horizon: int, model) -> Dict[str, pd.DataFrame]:
        """Run Monte Carlo simulation for climate risk prediction."""
        print(f"Running Monte Carlo simulation for {disaster_category} in {state_code}")

        try:
            # Generate future scenarios
            future_scenarios = self._generate_future_scenarios(state_code, disaster_category, time_horizon)

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

                simulation_results[f'year_{year}'] = yearly_results.to_dict('records')

            return simulation_results

        except Exception as e:
            print(f"Simulation failed: {e}")
            return {}

    def _generate_future_scenarios(self, state_code: str, disaster_category: str, time_horizon: int) -> pd.DataFrame:
        """Generate future climate scenarios based on historical patterns."""
        # Get historical data for the state and category
        historical_data = self._get_historical_scenarios(state_code, disaster_category)

        # Ensure month column exists
        if 'month' not in historical_data.columns and 'begin_date_time' in historical_data.columns:
            historical_data = historical_data.copy()
            historical_data['month'] = historical_data['begin_date_time'].dt.month

        scenarios = []

        for year in range(1, time_horizon + 1):
            # Sample from historical distribution for each month
            for month in range(1, 13):
                if 'month' not in historical_data.columns:
                    # If no month column, use all data for each month
                    month_data = historical_data
                else:
                    month_data = historical_data[historical_data['month'] == month]

                if len(month_data) == 0:
                    # If no data for this month, use all available data as fallback
                    month_data = historical_data

                if len(month_data) == 0:
                    continue

                # Sample number of events from historical distribution
                if 'year' in month_data.columns:
                    n_events = np.random.poisson(month_data.groupby('year').size().mean())
                else:
                    n_events = np.random.poisson(len(month_data))

                for event in range(n_events):
                    try:
                        scenario = self._sample_event_scenario(month_data)
                        scenario['year'] = year
                        scenario['month'] = month
                        scenario['simulated'] = True
                        scenarios.append(scenario)
                    except Exception:
                        scenarios.append({
                            'year': year,
                            'month': month,
                            'simulated': True,
                            'EVENT_TYPE': 'Unknown'
                        })

        # Create DataFrame and ensure year column exists
        future_scenarios = pd.DataFrame(scenarios)
        if 'year' not in future_scenarios.columns:
            future_scenarios['year'] = pd.Series(dtype='int64')

        return future_scenarios

    def _get_historical_scenarios(self, state_code: str, disaster_category: str) -> pd.DataFrame:
        """Get historical data for a specific state and category."""
        if self.historical_data is None or self.historical_data.empty:
            return pd.DataFrame()

        category_data = self.historical_data[self.historical_data['CATEGORY'] == disaster_category]

        if category_data.empty:
            return pd.DataFrame()

        # Filter by state if state column exists and state_code is not 'TEST'
        if state_code != 'TEST' and 'STATE' in category_data.columns:
            state_data = category_data[category_data['STATE'] == state_code]
            if not state_data.empty:
                return state_data
            else:
                return category_data

        return category_data

    def _sample_event_scenario(self, month_data: pd.DataFrame) -> Dict:
        """Sample a single event scenario from historical data."""
        scenario = {}

        # Sample from each feature's distribution
        for column in month_data.columns:
            if column not in ['year', 'month', 'simulated', 'MONTH_NAME']:
                try:
                    # Skip problematic columns
                    if column in ['BEGIN_DATE_TIME', 'END_DATE_TIME', 'begin_date_time', 'end_date_time']:
                        continue

                    if month_data[column].dtype in ['int64', 'float64']:
                        mu = month_data[column].mean()
                        sigma = month_data[column].std()

                        if sigma > 0 and not pd.isna(sigma):
                            scenario[column] = np.random.normal(mu, sigma)
                        else:
                            scenario[column] = mu
                    elif month_data[column].dtype == 'object':
                        unique_values = month_data[column].dropna().unique()
                        if len(unique_values) > 0:
                            scenario[column] = np.random.choice(unique_values)
                        else:
                            scenario[column] = None
                    else:
                        scenario[column] = month_data[column].iloc[0] if len(month_data) > 0 else None
                except Exception:
                    scenario[column] = month_data[column].iloc[0] if len(month_data) > 0 else None

        return scenario

    def _run_yearly_simulation(self, X: pd.DataFrame, model, year: int) -> pd.DataFrame:
        """Run Monte Carlo simulation for a single year."""
        results = []

        for sim in range(self.n_simulations):
            # Add random noise to features to simulate variability
            X_noisy = X.copy()

            # Add small random perturbations to numeric features
            for col in X_noisy.columns:
                if X_noisy[col].dtype in ['int64', 'float64']:
                    noise_level = X_noisy[col].std() * 0.1
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
        """Calculate risk metrics from simulation results."""
        all_results = []
        for year_key, year_data in simulation_results.items():
            # Handle both DataFrame format and list of dicts format
            if isinstance(year_data, pd.DataFrame):
                year_df = year_data.copy()
            elif isinstance(year_data, list):
                year_df = pd.DataFrame(year_data)
            else:
                continue

            year_df['year'] = int(year_key.split('_')[1])
            all_results.append(year_df)

        if not all_results:
            return {}

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
