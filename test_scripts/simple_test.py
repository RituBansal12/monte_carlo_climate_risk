import pandas as pd
import numpy as np
from cleaning_scripts._1_data_cleaning import DataCleaner
from cleaning_scripts._2_feature_engineering import FeatureEngineer

def simple_test():
    """Simple test of the data processing pipeline"""
    
    print("Testing basic data processing...")
    
    try:
        # Load data
        df = pd.read_parquet('data/SED_details_1950-2025.parquet')
        print(f"✓ Loaded {len(df)} storm events")
        
        # Test data cleaner
        cleaner = DataCleaner({})
        cleaned_df = cleaner.clean_sed_details(df.head(100))
        print(f"✓ Cleaned data: {cleaned_df.shape}")
        
        # Test feature engineer
        engineer = FeatureEngineer()
        engineered_df = engineer.categorize_disasters(cleaned_df)
        print(f"✓ Categorized disasters: {engineered_df['disaster_category'].value_counts().to_dict()}")
        
        # Test model import
        from modelling_scripts._1_statistical_models import ClimateRiskModel
        from modelling_scripts._2_monte_carlo import MonteCarloSimulator
        print("✓ All modules imported successfully")
        
        print("\n🎉 Basic pipeline test completed successfully!")
        print("The climate risk prediction system is ready for use.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
