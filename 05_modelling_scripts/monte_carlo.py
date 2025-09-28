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
                                  model: object) -> Dict[str, pd.DataFrame]:
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
                             model: object,
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
