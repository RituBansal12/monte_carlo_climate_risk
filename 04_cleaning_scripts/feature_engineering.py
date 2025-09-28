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
