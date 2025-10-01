#!/usr/bin/env python3
"""Data Preprocessing and Feature Engineering Module"""

import pandas as pd
import numpy as np
from typing import Dict
from geopy.distance import geodesic


def convert_target_to_numeric(value):
    """Convert target values to numeric, handling various formats"""
    if pd.isna(value) or value == '' or value is None:
        return 0.0

    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.lower() in ['none', 'null', 'na', 'n/a']:
            return 0.0

        try:
            clean_value = value.replace('$', '').replace(',', '').replace('K', '000').replace('M', '000000')
            return float(clean_value)
        except ValueError:
            return 0.0

    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def preprocess_target_columns(df, target_columns):
    """Convert target columns to numeric types"""
    df_processed = df.copy()

    for col in target_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(convert_target_to_numeric)

    return df_processed


def map_category_codes(category_code):
    """Map numeric category codes to descriptive names"""
    category_map = {
        '1': 'flooding',
        '2': 'severe_weather',
        '3': 'winter_weather',
        '4': 'tropical_storm',
        '5': 'fire'
    }

    if category_code is None or pd.isna(category_code):
        return 'unknown'

    return category_map.get(str(category_code), f'category_{category_code}')


def create_temporal_features(df):
    """Create time-based features from date information"""
    if 'begin_date_time' not in df.columns:
        return df

    df['year'] = df['begin_date_time'].dt.year
    df['month'] = df['begin_date_time'].dt.month
    df['quarter'] = df['begin_date_time'].dt.quarter
    df['day_of_year'] = df['begin_date_time'].dt.dayofyear
    df['is_weekend'] = df['begin_date_time'].dt.weekday.isin([5, 6]).astype(int)

    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

    return df


def create_geographic_features(df):
    """Create geography-based features"""
    if 'begin_date_time' in df.columns and 'end_date_time' in df.columns:
        df['duration_hours'] = (df['end_date_time'] - df['begin_date_time']).dt.total_seconds() / 3600

    coord_columns = ['begin_lat', 'begin_lon', 'end_lat', 'end_lon']
    if all(col in df.columns for col in coord_columns):
        df['path_length_km'] = df.apply(
            lambda row: geodesic(
                (row['begin_lat'], row['begin_lon']),
                (row['end_lat'], row['end_lon'])
            ).kilometers if row['begin_lat'] != row['end_lat'] else 0,
            axis=1
        )

    return df


def create_impact_features(df):
    """Create impact severity features"""
    if 'deaths_direct' in df.columns and 'deaths_indirect' in df.columns:
        df['total_deaths'] = df['deaths_direct'] + df['deaths_indirect']
    else:
        df['total_deaths'] = 0

    if 'injuries_direct' in df.columns and 'injuries_indirect' in df.columns:
        df['total_injuries'] = df['injuries_direct'] + df['injuries_indirect']
    else:
        df['total_injuries'] = 0

    if 'damage_property' in df.columns and 'damage_crops' in df.columns:
        df['total_damage'] = df['damage_property'] + df['damage_crops']
    else:
        df['total_damage'] = 0

    df['severity_score'] = (df['total_deaths'] * 10 +
                           df['total_injuries'] * 2 +
                           np.log1p(df['total_damage']) / 1000000)

    df['has_fatalities'] = (df['total_deaths'] > 0).astype(int)
    df['has_injuries'] = (df['total_injuries'] > 0).astype(int)
    df['has_damage'] = (df['total_damage'] > 0).astype(int)

    return df


def categorize_by_event_type(df):
    """Categorize events using the EVENT_TYPE column for better categorization"""
    event_to_category = {
        'flooding': ['Flash Flood', 'Flood', 'Coastal Flood', 'River Flood', 'Urban Flood'],
        'severe_weather': ['Thunderstorm Wind', 'Hail', 'Lightning', 'High Wind', 'Strong Wind'],
        'winter_weather': ['Winter Storm', 'Blizzard', 'Ice Storm', 'Heavy Snow', 'Lake-Effect Snow'],
        'tropical_storm': ['Hurricane', 'Tropical Storm', 'Tropical Depression', 'Hurricane (Typhoon)'],
        'tornado': ['Tornado', 'Funnel Cloud', 'Waterspout'],
        'fire': ['Wildfire', 'Grass Fire', 'Forest Fire'],
        'other': ['Drought', 'Dust Storm', 'Extreme Cold/Wind Chill', 'Heat', 'Cold/Wind Chill']
    }

    df['CATEGORY'] = 'other'  # Default category

    if 'EVENT_TYPE' in df.columns:
        for category, events in event_to_category.items():
            mask = df['EVENT_TYPE'].isin(events)
            df.loc[mask, 'CATEGORY'] = category

    return df


def validate_modeling_data(df, state=None, category=None, min_records=50, min_nonzero_targets=10):
    """Validate if data is sufficient for modeling"""
    validation = {
        'sufficient_data': False,
        'total_records': len(df),
        'target_analysis': {},
        'recommendations': []
    }

    filtered_df = df.copy()
    if state:
        filtered_df = filtered_df[filtered_df['STATE'] == state]
    if category:
        filtered_df = filtered_df[filtered_df['CATEGORY'] == category]

    validation['filtered_records'] = len(filtered_df)

    if len(filtered_df) < min_records:
        validation['recommendations'].append(f"Insufficient records: {len(filtered_df)} < {min_records}")
        return validation

    target_columns = ['DEATHS_DIRECT', 'INJURIES_DIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS']

    for col in target_columns:
        if col in filtered_df.columns:
            non_zero = (filtered_df[col] > 0).sum()
            validation['target_analysis'][col] = {
                'total': len(filtered_df),
                'non_zero': non_zero,
                'percentage': non_zero / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            }

            if non_zero < min_nonzero_targets:
                validation['recommendations'].append(f"Insufficient non-zero {col}: {non_zero} < {min_nonzero_targets}")

    viable_targets = [info for info in validation['target_analysis'].values() if info['non_zero'] >= min_nonzero_targets]
    validation['sufficient_data'] = len(viable_targets) > 0 and len(filtered_df) >= min_records

    return validation


def preprocess_climate_data(df):
    """Complete preprocessing and feature engineering pipeline"""
    print("Starting comprehensive data preprocessing...")

    # Convert target columns to numeric
    target_columns = ['DEATHS_DIRECT', 'INJURIES_DIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS']
    df = preprocess_target_columns(df, target_columns)

    # Create categories from EVENT_TYPE if available (more detailed than CATEGORY)
    if 'EVENT_TYPE' in df.columns:
        df = categorize_by_event_type(df)
    elif 'CATEGORY' in df.columns:
        # Fallback to CATEGORY mapping if EVENT_TYPE not available
        df['CATEGORY'] = df['CATEGORY'].apply(map_category_codes)
        # Don't filter out unknown categories - keep all data
        # df = df[df['CATEGORY'] != 'unknown']

    # Handle state column
    if 'STATE' in df.columns:
        df['STATE'] = df['STATE'].fillna('UNKNOWN').astype(str).str.upper()

    # Create features
    df = create_temporal_features(df)
    df = create_geographic_features(df)
    df = create_impact_features(df)

    print(f"Preprocessing complete. Final shape: {df.shape}")
    return df
