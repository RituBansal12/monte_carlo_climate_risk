#!/usr/bin/env python3
"""Comprehensive Climate Risk Analysis Pipeline"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List
import warnings
import time
import sys

warnings.filterwarnings('ignore')

from cleaning_scripts.data_cleaning import DataCleaner
from cleaning_scripts.data_processing import preprocess_climate_data, validate_modeling_data
from modelling_scripts.statistical_models import ClimateRiskModel
from modelling_scripts.monte_carlo import MonteCarloSimulator


class ComprehensiveClimateRiskPipeline:
    """Main pipeline for comprehensive climate risk analysis."""

    def __init__(self, config_path: str = 'config/pipeline_config.json'):
        """Initialize with configuration settings."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.data_cleaner = DataCleaner(self.config.get('cleaning', {}))

    def run_comprehensive_pipeline(self):
        """Execute complete climate risk analysis pipeline."""
        print("Starting Comprehensive Climate Risk Analysis Pipeline...")
        print("Analyzing 75 years of climate disaster data...")
        start_time = datetime.now()

        # Phase 1: Data Loading and Processing
        print("Phase 1: Loading and processing climate data...")
        data = self._load_and_process_data()

        # Phase 2: Multi-Scenario Risk Analysis
        print("Phase 2: Running multi-scenario Monte Carlo simulations...")
        comprehensive_results = self._run_multi_scenario_analysis(data)

        # Phase 3: Results Compilation
        print("Phase 3: Compiling comprehensive risk metrics...")
        self._compile_comprehensive_results(comprehensive_results)

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"Analysis completed in {duration}")
        print("Results saved to: results/comprehensive_risk_analysis.json")

        return comprehensive_results

    def _load_and_process_data(self) -> Dict[str, pd.DataFrame]:
        """Load and process datasets."""
        data = {}
        data['sed_details'] = pd.read_parquet('data/SED_details_1950-2025.parquet')
        print(f"Loaded dataset with {len(data['sed_details'])} records")

        # Preprocess data
        print("Preprocessing climate data...")
        data['sed_details'] = preprocess_climate_data(data['sed_details'])

        # Apply cleaning
        data['sed_details'] = self.data_cleaner.clean_sed_details(data['sed_details'])
        print("Data cleaning completed")

        return data

    def _run_multi_scenario_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Run comprehensive multi-scenario risk analysis."""
        comprehensive_results = {
            'metadata': {
                'execution_timestamp': datetime.now().isoformat(),
                'dataset_size': len(data['sed_details']),
                'analysis_version': '1.0'
            },
            'scenarios': {},
            'summary': {}
        }

        # Define scenarios with enhanced category names
        scenarios = [
            {'state': 'FLORIDA', 'category': 'Flooding', 'horizon': 1, 'description': 'Florida Flooding (1-year horizon)'},
            {'state': 'TEXAS', 'category': 'Severe Weather', 'horizon': 3, 'description': 'Texas Severe Weather (3-year horizon)'},
            {'state': 'TEXAS', 'category': 'Flooding', 'horizon': 1, 'description': 'Texas Flooding (1-year horizon)'}
        ]

        print(f"Created {len(scenarios)} simulation scenarios")

        # Train models for categories using enhanced EVENT_TYPE mapping
        print("Training models for available categories...")
        trained_model_objects = {}

        # Use enhanced category mapping based on EVENT_TYPE
        category_mapping = {
            'Flooding': ['Flood', 'Flash Flood', 'Coastal Flood'],
            'Severe Weather': ['Thunderstorm', 'Severe Thunderstorm', 'Tornado', 'Hail', 'Wind']
        }

        for category_name, event_types in category_mapping.items():
            try:
                model = ClimateRiskModel(category_name)
                category_data = data['sed_details'][data['sed_details']['EVENT_TYPE'].isin(event_types)]

                # Validate data quality with enhanced criteria
                validation = validate_modeling_data(
                    category_data,
                    min_records=50,  # Higher threshold for better models
                    min_nonzero_targets=10  # More non-zero examples needed
                )

                if not validation['sufficient_data']:
                    print(f"Insufficient data for {category_name}: {validation['recommendations']}")
                    continue

                print(f"Training model for {category_name} with {len(category_data)} records")

                # Prepare features and targets with enhanced processing
                X = model.prepare_features(category_data)

                y = {}
                for target in ['DEATHS_DIRECT', 'INJURIES_DIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS']:
                    if target in category_data.columns:
                        if target == 'DAMAGE_PROPERTY':
                            y['property_damage'] = category_data[target]
                        elif target == 'DAMAGE_CROPS':
                            y['crop_damage'] = category_data[target]
                        elif target == 'DEATHS_DIRECT':
                            y['deaths'] = category_data[target]
                        elif target == 'INJURIES_DIRECT':
                            y['injuries'] = category_data[target]

                if y:
                    trained_models = model.train_models(X, y)
                    trained_model_objects[category_name] = model
                    print(f"✅ Model trained for {category_name} with {len(trained_models)} targets")

            except Exception as e:
                print(f"❌ Model training failed for {category_name}: {e}")

        print(f"Successfully trained {len(trained_model_objects)} enhanced models")

        # Run simulations
        print(f"Running {len(scenarios)} Monte Carlo simulations...")

        for i, scenario in enumerate(scenarios, 1):
            print(f"  {i}. {scenario['description']}")

            try:
                results = self._run_single_scenario(scenario, trained_model_objects, data)
                if results:
                    scenario_key = f"{scenario['state']}_{scenario['category']}_{scenario['horizon']}yr"
                    comprehensive_results['scenarios'][scenario_key] = {'results': results}
            except Exception as e:
                print(f"Scenario {scenario['description']} failed: {e}")
                continue

        return comprehensive_results

    def _run_single_scenario(self, scenario: Dict, models: Dict, data: Dict) -> Dict:
        """Run Monte Carlo simulation for a single scenario."""
        if scenario['category'] not in models:
            return None

        try:
            # Create Monte Carlo simulator
            monte_carlo = MonteCarloSimulator(
                n_simulations=1000,
                historical_data=data['sed_details']
            )

            # Run simulation
            simulation_results = monte_carlo.run_climate_risk_simulation(
                scenario['state'], scenario['category'], scenario['horizon'], models[scenario['category']]
            )

            # Calculate risk metrics
            risk_metrics = monte_carlo.calculate_risk_metrics(simulation_results)

            return {
                'simulation_results': simulation_results,
                'risk_metrics': risk_metrics,
                'scenario_summary': self._summarize_scenario_results(risk_metrics, scenario)
            }

        except Exception as e:
            print(f"Scenario failed: {e}")
            return None

    def _summarize_scenario_results(self, risk_metrics: Dict, scenario: Dict) -> Dict:
        """Create human-readable summary of scenario results."""
        summary = {
            'scenario': scenario,
            'key_findings': {},
            'risk_recommendations': {}
        }

        for impact_type, metrics in risk_metrics.items():
            mean_val = metrics.get('mean', 0)
            p95_val = metrics.get('p95', 0)
            prob_positive = metrics.get('probability_positive', 0)

            summary['key_findings'][impact_type] = {
                'expected_impact': mean_val,
                'worst_case_95confidence': p95_val,
                'probability_any_impact': prob_positive
            }

            # Generate recommendations
            if impact_type == 'deaths' and mean_val > 1:
                summary['risk_recommendations'][impact_type] = "HIGH PRIORITY: Significant life safety risk"
            elif impact_type == 'property_damage' and mean_val > 1000000:
                summary['risk_recommendations'][impact_type] = "HIGH PRIORITY: Major property damage risk"
            elif prob_positive > 0.5:
                summary['risk_recommendations'][impact_type] = "MODERATE PRIORITY: Frequent impact events"
            else:
                summary['risk_recommendations'][impact_type] = "LOW PRIORITY: Minimal risk"

        return summary

    def _compile_comprehensive_results(self, comprehensive_results: Dict):
        """Compile and save comprehensive analysis results."""
        print("Compiling final results...")

        # Calculate summary statistics with enhanced categories
        all_scenarios = comprehensive_results['scenarios']
        if all_scenarios:
            categories_found = set()
            states_found = set()

            for scenario_key in all_scenarios.keys():
                parts = scenario_key.split('_')
                if len(parts) >= 2:
                    states_found.add(parts[0])
                    categories_found.add(parts[1])

            summary = {
                'total_scenarios_analyzed': len(all_scenarios),
                'categories_covered': len(categories_found),
                'states_covered': len(states_found),
                'enhanced_features_used': True,
                'analysis_timestamp': datetime.now().isoformat()
            }
        else:
            summary = {
                'total_scenarios_analyzed': 0,
                'categories_covered': 0,
                'states_covered': 0,
                'enhanced_features_used': False,
                'analysis_timestamp': datetime.now().isoformat()
            }

        comprehensive_results['summary'] = summary

        # Save results
        with open('results/comprehensive_risk_analysis.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)

        print("Comprehensive risk analysis saved!")


if __name__ == "__main__":
    pipeline = ComprehensiveClimateRiskPipeline()
    results = pipeline.run_comprehensive_pipeline()

    print("\n" + "="*80)
    print("COMPREHENSIVE RISK ANALYSIS SUMMARY")
    print("="*80)
    print(f"Scenarios Analyzed: {results['summary']['total_scenarios_analyzed']}")
    print(f"States Covered: {results['summary']['states_covered']}")
    print(f"Categories Analyzed: {results['summary']['categories_covered']}")
    print(f"Enhanced Features: {'✅ ENABLED' if results['summary'].get('enhanced_features_used', False) else '❌ DISABLED'}")
    print(f"Results File: results/comprehensive_risk_analysis.json")
    print("="*80)
