import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from cleaning_scripts.data_cleaning import DataCleaner
from cleaning_scripts.feature_engineering import FeatureEngineer
from modelling_scripts.statistical_models import ClimateRiskModel
from modelling_scripts.monte_carlo import MonteCarloSimulator

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
        
        # Phase 3: Model Training
        print("Phase 3: Training statistical models...")
        models = self._train_models(data)
        
        # Phase 4: Monte Carlo Simulation
        print("Phase 4: Running Monte Carlo simulations...")
        simulation_results = self._run_simulations(models)
        
        # Phase 5: Results Analysis
        print("Phase 5: Analyzing results...")
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

    def _train_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
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

    def _run_simulations(self, models: Dict[str, object]) -> Dict:
        """Run Monte Carlo simulations for key states and categories"""
        simulation_results = {}
        
        # Define simulation scenarios
        scenarios = [
            {'state': 'TX', 'category': 'tornado', 'horizon': 5},
            {'state': 'FL', 'category': 'tropical', 'horizon': 5},
            {'state': 'CA', 'category': 'fire', 'horizon': 5}
        ]
        
        for scenario in scenarios:
            state = scenario['state']
            category = scenario['category']
            horizon = scenario['horizon']
            
            if category in models:
                # Filter state data for this category
                state_category_data = self._get_state_category_data(state, category, models)
                
                if len(state_category_data) > 0:
                    results = self.monte_carlo.run_climate_risk_simulation(
                        state, category, horizon, models[category]
                    )
                    simulation_results[f'{state}_{category}'] = results
        
        return simulation_results

    def _get_state_category_data(self, state_code: str, category: str, models: Dict[str, object]):
        """Get historical data for a specific state and category"""
        # This is a placeholder - in a real implementation, you would
        # filter the data by state code and category
        # For now, we'll use a subset of the category data
        return pd.DataFrame()  # Placeholder

    def _analyze_results(self, simulation_results: Dict):
        """Analyze simulation results and create summary"""
        # Calculate risk metrics for each simulation
        risk_assessments = {}
        
        for scenario_key, results in simulation_results.items():
            risk_metrics = self.monte_carlo.calculate_risk_metrics(results)
            risk_assessments[scenario_key] = risk_metrics
        
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
