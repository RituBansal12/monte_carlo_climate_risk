import pandas as pd
import numpy as np
import json
from cleaning_scripts._1_data_cleaning import DataCleaner
from cleaning_scripts._2_feature_engineering import FeatureEngineer
from modelling_scripts._1_statistical_models import ClimateRiskModel
from modelling_scripts._2_monte_carlo import MonteCarloSimulator

def test_pipeline():
    """Test the climate risk pipeline with a subset of data"""
    
    print("Testing Climate Risk Pipeline...")
    
    # Load and clean data
    print("1. Loading and cleaning data...")
    df = pd.read_parquet('data/SED_details_1950-2025.parquet')
    
    # Clean the data
    cleaner = DataCleaner({})
    cleaned_df = cleaner.clean_sed_details(df)
    
    # Engineer features
    engineer = FeatureEngineer()
    engineered_df = engineer.create_temporal_features(cleaned_df)
    engineered_df = engineer.create_geographic_features(engineered_df)
    engineered_df = engineer.create_impact_features(engineered_df)
    engineered_df = engineer.categorize_disasters(engineered_df)
    
    print(f"Processed {len(engineered_df)} storm events")
    
    # Focus on tornado category for testing
    tornado_data = engineered_df[engineered_df['disaster_category'] == 'tornado']
    print(f"Found {len(tornado_data)} tornado events for modeling")
    
    if len(tornado_data) > 100:  # Need enough data for training
        # Train a simple model
        print("2. Training model...")
        model = ClimateRiskModel('tornado')
        
        # Prepare features
        X = model.prepare_features(tornado_data)
        print(f"Feature matrix shape: {X.shape}")
        
        # Define targets
        y = {
            'deaths': tornado_data['total_deaths'],
            'injuries': tornado_data['total_injuries'],
            'property_damage': tornado_data['damage_property'],
            'crop_damage': tornado_data['damage_crops']
        }
        
        # Train models
        trained_models = model.train_models(X, y)
        
        # Run a small Monte Carlo simulation
        print("3. Running Monte Carlo simulation...")
        simulator = MonteCarloSimulator(n_simulations=1000)  # Smaller for testing
        
        # Create a small test scenario
        test_scenario = tornado_data.head(10).copy()
        test_scenario['year'] = 2025
        test_scenario['month'] = 5
        
        # Run simulation
        X_test = model.prepare_features(test_scenario)
        simulation_results = simulator._run_yearly_simulation(X_test, model, 2025)
        
        print(f"Simulation completed with {len(simulation_results)} results")
        
        # Calculate risk metrics
        risk_metrics = simulator.calculate_risk_metrics({'year_1': simulation_results})
        
        print("Risk Metrics:")
        for impact, metrics in risk_metrics.items():
            print(f"  {impact.title()}:")
            print(f"    Mean: {metrics['mean']:.2f}")
            print(f"    95th percentile: {metrics['p95']:.2f}")
            print(f"    Probability of impact: {metrics['probability_positive']:.2%}")
        
        # Save results
        with open('results/test_results.json', 'w') as f:
            json.dump({
                'risk_metrics': risk_metrics,
                'simulation_shape': simulation_results.shape
            }, f, indent=2)
        
        print("4. Test completed successfully!")
        print("Results saved to results/test_results.json")
        
    else:
        print("Not enough tornado data for testing. Need at least 100 events.")

if __name__ == "__main__":
    test_pipeline()
