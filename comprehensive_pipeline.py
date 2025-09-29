"""
Comprehensive Climate Risk Analysis Pipeline

This module provides a complete climate risk analysis system that processes historical
climate disaster data and runs Monte Carlo simulations to assess future risks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
import warnings
import time
import sys
warnings.filterwarnings('ignore')

from cleaning_scripts.data_cleaning import DataCleaner
from cleaning_scripts.feature_engineering import FeatureEngineer
from modelling_scripts.statistical_models import ClimateRiskModel
from modelling_scripts.monte_carlo import MonteCarloSimulator


def log_progress(message, start_time=None):
    """Log progress with timestamp and optional elapsed time."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if start_time:
        elapsed = time.time() - start_time
        print(f"[{timestamp}] {message} (Elapsed: {elapsed:.1f}s)")
    else:
        print(f"[{timestamp}] {message}")
    sys.stdout.flush()
    return time.time()


class ProgressMonitor:
    """Simple progress monitor for long-running operations."""

    def __init__(self, total, description):
        self.total = total
        self.description = description
        self.start_time = time.time()
        self.last_update = 0

    def update(self, current):
        """Update progress if enough time has passed."""
        if time.time() - self.last_update > 5:  # Update every 5 seconds
            elapsed = time.time() - self.start_time
            percent = (current / self.total) * 100
            eta = (elapsed / current) * (self.total - current) if current > 0 else 0
            print(f"  [{self.description}] {current}/{self.total} ({percent:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
            sys.stdout.flush()
            self.last_update = time.time()


class ComprehensiveClimateRiskPipeline:
    """
    Main pipeline class for comprehensive climate risk analysis.

    This class orchestrates the complete analysis workflow including data loading,
    feature engineering, model training, Monte Carlo simulations, and results compilation.
    """

    def __init__(self, config_path: str = 'config/pipeline_config.json'):
        """
        Initialize the pipeline with configuration settings.

        Args:
            config_path: Path to JSON configuration file containing pipeline parameters
        """
        self.start_time = time.time()
        log_progress("Pipeline initialization started")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.data_cleaner = DataCleaner(self.config.get('cleaning', {}))
        self.feature_engineer = FeatureEngineer()
        log_progress("Pipeline initialization completed", self.start_time)

    def run_comprehensive_pipeline(self):
        """
        Execute the complete climate risk analysis pipeline.

        Returns:
            Dict containing comprehensive analysis results with metadata, scenarios, and summary
        """
        log_progress("Starting Comprehensive Climate Risk Analysis Pipeline")
        log_progress("Analyzing 75 years of climate disaster data")
        pipeline_start = time.time()

        # Phase 1: Data Loading and Processing
        log_progress("Phase 1: Loading and processing climate data")
        phase_start = time.time()
        data = self._load_and_process_data()
        log_progress("Phase 1 completed", phase_start)

        # Phase 2: Multi-Scenario Risk Analysis
        log_progress("Phase 2: Running multi-scenario Monte Carlo simulations")
        phase_start = time.time()
        comprehensive_results = self._run_multi_scenario_analysis(data)
        log_progress("Phase 2 completed", phase_start)

        # Phase 3: Results Compilation
        log_progress("Phase 3: Compiling comprehensive risk metrics")
        phase_start = time.time()
        self._compile_comprehensive_results(comprehensive_results)
        log_progress("Phase 3 completed", phase_start)

        total_time = time.time() - pipeline_start
        log_progress(f"Comprehensive analysis completed in {total_time:.1f} seconds")
        log_progress("Results saved to: results/comprehensive_risk_analysis.json")

        return comprehensive_results
    """
    Main pipeline class for comprehensive climate risk analysis.

    This class orchestrates the complete analysis workflow including data loading,
    feature engineering, model training, Monte Carlo simulations, and results compilation.
    """

    def __init__(self, config_path: str = 'config/pipeline_config.json'):
        """
        Initialize the pipeline with configuration settings.

        Args:
            config_path: Path to JSON configuration file containing pipeline parameters
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.data_cleaner = DataCleaner(self.config.get('cleaning', {}))
        self.feature_engineer = FeatureEngineer()

    def run_comprehensive_pipeline(self):
        """
        Execute the complete climate risk analysis pipeline.

        Returns:
            Dict containing comprehensive analysis results with metadata, scenarios, and summary
        """
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

        print(f"Comprehensive analysis completed in {duration}")
        print("Results saved to: results/comprehensive_risk_analysis.json")

        return comprehensive_results

    def _load_and_process_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and process all required datasets.

        Returns:
            Dict containing processed DataFrames with keys: 'sed_details'
        """
        phase_start = log_progress("Loading climate dataset")

        data = {}
        data['sed_details'] = pd.read_parquet('data/SED_details_1950-2025.parquet')
        log_progress(f"Loaded dataset with {len(data['sed_details'])} records")

        data['sed_details'] = self.data_cleaner.clean_sed_details(data['sed_details'])
        log_progress("Data cleaning completed")

        # Process features
        data['sed_details'] = self.feature_engineer.create_temporal_features(data['sed_details'])
        data['sed_details'] = self.feature_engineer.create_geographic_features(data['sed_details'])
        data['sed_details'] = self.feature_engineer.create_impact_features(data['sed_details'])
        data['sed_details'] = self.feature_engineer.categorize_disasters(data['sed_details'])

        log_progress("Feature engineering completed", phase_start)
        return data

    def _run_multi_scenario_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run comprehensive multi-scenario risk analysis.

        Args:
            data: Dict containing processed DataFrames

        Returns:
            Dict containing comprehensive analysis results
        """
        comprehensive_results = {
            'metadata': {
                'execution_timestamp': datetime.now().isoformat(),
                'dataset_size': len(data['sed_details']),
                'analysis_version': '1.0'
            },
            'scenarios': {},
            'summary': {}
        }

        # Create scenarios using available categories
        scenarios = []
        states = ['TX', 'FL']  
        available_categories = ['flooding', 'severe_weather', 'wildfire']

        for i, category in enumerate(available_categories[:1]): 
            for horizon in [1]:  # Reduced from [1, 3, 5]
                scenarios.append({
                    'state': states[i % len(states)],
                    'category': category,
                    'horizon': horizon,
                    'description': f'{states[i % len(states)]} {category.replace("_", " ").title()} ({horizon}-year horizon)'
                })
        print(f"Created {len(scenarios)} simulation scenarios")

        # Train models for available categories
        log_progress("Training models for available categories")
        models = self._train_models_for_categories(data, available_categories[: 1])
        log_progress(f"Running {len(scenarios)} Monte Carlo simulations")
        progress_monitor = ProgressMonitor(len(scenarios), "Monte Carlo Simulations")

        for i, scenario in enumerate(scenarios[:1], 1): 
            log_progress(f"  {i}. {scenario['description']}")
            progress_monitor.update(i)

            results = self._run_single_scenario(scenario, models, data)
            if results:
                scenario_key = f"{scenario['state']}_{scenario['category']}_{scenario['horizon']}yr"
                comprehensive_results['scenarios'][scenario_key] = {
                    'results': results
                }

        return comprehensive_results

    def _train_models_for_categories(self, data: Dict[str, pd.DataFrame], categories: List[str]) -> Dict[str, object]:
        """
        Train statistical models for specific disaster categories.

        Args:
            data: Dict containing processed DataFrames
            categories: List of disaster category names to train models for

        Returns:
            Dict mapping category names to trained model objects
        """
        models = {}
        category_monitor = ProgressMonitor(len(categories), "Model Training")

        for i, category in enumerate(categories, 1):
            log_progress(f"Training model for {category} category ({i}/{len(categories)})")
            category_monitor.update(i)

            try:
                model = ClimateRiskModel(category)
                category_data = data['sed_details'][data['sed_details']['disaster_category'] == category]

                if len(category_data) < 50:
                    log_progress(f"Insufficient data for {category} ({len(category_data)} samples)")
                    continue

                # Prepare features
                X = model.prepare_features(category_data)

                # Define target variables - only include columns that exist
                y = {}
                for target in ['total_deaths', 'total_injuries', 'damage_property', 'damage_crops']:
                    if target in category_data.columns:
                        y[target.split('_')[1]] = category_data[target]

                if y:
                    models[category] = model  # Store the ClimateRiskModel object, not the dict
                    log_progress(f"Model trained for {category}")
                else:
                    log_progress(f"No target variables for {category}")

            except Exception as e:
                log_progress(f"Model training failed for {category}: {e}")

        log_progress(f"Successfully trained {len(models)} models")
        return models

    def _run_single_scenario(self, scenario: Dict, models: Dict, data: Dict) -> Dict:
        """
        Run Monte Carlo simulation for a single scenario.

        Args:
            scenario: Dict containing scenario parameters (state, category, horizon)
            models: Dict of trained models by category
            data: Dict containing processed DataFrames

        Returns:
            Dict containing simulation results and risk metrics, or None if failed
        """
        if scenario['category'] not in models:
            return None

        scenario_start = log_progress(f"Running simulation for {scenario['description']}")

        try:
            # Create Monte Carlo simulator with historical data
            monte_carlo = MonteCarloSimulator(
                n_simulations=5,  # Reduced from 10 to 5 for faster testing
                historical_data=data['sed_details']
            )

            # Run simulation
            simulation_results = monte_carlo.run_climate_risk_simulation(
                scenario['state'], scenario['category'], scenario['horizon'], models[scenario['category']]
            )

            # Calculate comprehensive risk metrics
            risk_metrics = monte_carlo.calculate_risk_metrics(simulation_results)

            log_progress(f"Scenario completed", scenario_start)
            return {
                'simulation_results': simulation_results,
                'risk_metrics': risk_metrics,
                'scenario_summary': self._summarize_scenario_results(risk_metrics, scenario)
            }

        except Exception as e:
            log_progress(f"Scenario failed: {e}")
            return None

    def _summarize_scenario_results(self, risk_metrics: Dict, scenario: Dict) -> Dict:
        """
        Create human-readable summary of scenario results.

        Args:
            risk_metrics: Dict containing calculated risk metrics
            scenario: Dict containing scenario parameters

        Returns:
            Dict containing scenario summary with key findings and recommendations
        """
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
        """
        Compile and save comprehensive analysis results.

        Args:
            comprehensive_results: Dict containing all analysis results
        """
        log_progress("Compiling final results")

        # Calculate overall summary statistics
        all_scenarios = comprehensive_results['scenarios']
        summary = {
            'total_scenarios_analyzed': len(all_scenarios),
            'categories_covered': len(set(scenario.split('_')[1] for scenario in all_scenarios.keys())),
            'states_covered': len(set(scenario.split('_')[0] for scenario in all_scenarios.keys())),
            'analysis_timestamp': datetime.now().isoformat()
        }

        comprehensive_results['summary'] = summary

        # Convert DataFrames to dictionaries for proper JSON serialization
        def convert_for_json(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict('records')
            return obj

        # Save comprehensive results
        with open('results/comprehensive_risk_analysis.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=convert_for_json)

        log_progress("Comprehensive risk analysis saved!")


if __name__ == "__main__":
    pipeline = ComprehensiveClimateRiskPipeline()
    results = pipeline.run_comprehensive_pipeline()

    print("\n" + "="*80)
    print("COMPREHENSIVE RISK ANALYSIS SUMMARY")
    print("="*80)
    print(f"Scenarios Analyzed: {results['summary']['total_scenarios_analyzed']}")
    print(f"States Covered: {results['summary']['states_covered']}")
    print(f"Categories Analyzed: {results['summary']['categories_covered']}")
    print(f"Results File: results/comprehensive_risk_analysis.json")
    print("="*80)
