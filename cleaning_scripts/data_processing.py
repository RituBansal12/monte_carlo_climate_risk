#!/usr/bin/env python3
"""Data Preprocessing and Feature Engineering Module"""

import pandas as pd
import numpy as np
from typing import Dict
from geopy.distance import geodesic


def convert_target_to_numeric(value):
    """Convert target values to numeric, handling various formats including K/M/B suffixes"""
    if pd.isna(value) or value == '' or value is None:
        return 0.0

    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.lower() in ['none', 'null', 'na', 'n/a']:
            return 0.0

        try:
            # Handle K, M, B suffixes
            value_upper = value.upper()
            if 'K' in value_upper:
                multiplier = 1000
                clean_value = value_upper.replace('K', '')
            elif 'M' in value_upper:
                multiplier = 1000000
                clean_value = value_upper.replace('M', '')
            elif 'B' in value_upper:
                multiplier = 1000000000
                clean_value = value_upper.replace('B', '')
            else:
                multiplier = 1
                clean_value = value_upper

            # Remove common formatting characters
            clean_value = clean_value.replace('$', '').replace(',', '')
            return float(clean_value) * multiplier
        except ValueError:
            return 0.0

    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def clean_damage_columns(df, damage_columns=['DAMAGE_PROPERTY', 'DAMAGE_CROPS']):
    """Enhanced cleaning for damage columns with K/M/B suffix handling"""
    df_cleaned = df.copy()

    for col in damage_columns:
        if col in df_cleaned.columns:
            print(f"🧹 Enhanced cleaning of {col} column...")

            # Convert to string first to handle mixed types
            df_cleaned[col] = df_cleaned[col].astype(str)

            # Clean and convert to numeric
            df_cleaned[col] = df_cleaned[col].apply(convert_target_to_numeric)
            print(f"   Converted {col} to numeric (sample: {df_cleaned[col].iloc[0] if len(df_cleaned) > 0 else 'N/A'})")

    return df_cleaned


def preprocess_target_columns(df, target_columns):
    """Enhanced target column preprocessing with comprehensive cleaning"""
    df_processed = df.copy()

    # First clean damage columns with enhanced method
    damage_cols = [col for col in target_columns if 'DAMAGE' in col]
    if damage_cols:
        df_processed = clean_damage_columns(df_processed, damage_cols)

    # Clean remaining target columns
    other_targets = [col for col in target_columns if col not in damage_cols]
    for col in other_targets:
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


def create_enhanced_temporal_features(df):
    """Create comprehensive time-based features"""
    # Look for date column with various possible names
    date_column = None
    for col in df.columns:
        if 'date' in col.lower() and 'time' in col.lower():
            date_column = col
            break
        elif 'begin_date' in col.lower():
            date_column = col
            break

    if date_column is None:
        print("Warning: No date column found for temporal features")
        return df

    print(f"Using date column: {date_column}")

    # Basic temporal features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['quarter'] = df[date_column].dt.quarter
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['is_weekend'] = df[date_column].dt.weekday.isin([5, 6]).astype(int)

    # Enhanced seasonal features
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

    # Trend features
    current_year = 2024
    df['years_since_1950'] = df['year'] - 1950
    df['year_squared'] = df['year'] ** 2
    df['year_cubed'] = df['year'] ** 3

    # Interaction features
    if 'year' in df.columns and 'month' in df.columns:
        df['year_month'] = df['year'] * 100 + df['month']
        df['year_season'] = df['year'] + df['month'] / 12

    return df


def create_enhanced_geographic_features(df):
    """Create comprehensive geography-based features with clustering"""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Warning: scikit-learn not available for geographic clustering")
        KMeans = None

    # Look for date columns
    begin_date_col = None
    end_date_col = None

    for col in df.columns:
        if 'begin' in col.lower() and ('date' in col.lower() or 'time' in col.lower()):
            begin_date_col = col
        elif 'end' in col.lower() and ('date' in col.lower() or 'time' in col.lower()):
            end_date_col = col

    if begin_date_col and end_date_col:
        df['duration_hours'] = (df[end_date_col] - df[begin_date_col]).dt.total_seconds() / 3600

    coord_columns = ['begin_lat', 'begin_lon', 'end_lat', 'end_lon']
    if all(col in df.columns for col in coord_columns):
        df['path_length_km'] = df.apply(
            lambda row: geodesic(
                (row['begin_lat'], row['begin_lon']),
                (row['end_lat'], row['end_lon'])
            ).kilometers if row['begin_lat'] != row['end_lat'] else 0,
            axis=1
        )

        # Enhanced geographic features
        if KMeans is not None and len(df) > 100:
            try:
                # Create geographic clusters
                geo_data = df[['begin_lat', 'begin_lon']].dropna()
                if len(geo_data) > 50:
                    scaler = StandardScaler()
                    scaled_coords = scaler.fit_transform(geo_data)
                    n_clusters = min(20, max(5, len(geo_data) // 100))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_coords)

                    # Map clusters back to original dataframe
                    cluster_series = pd.Series(index=geo_data.index, data=clusters)
                    df['geo_cluster'] = cluster_series.reindex(df.index, fill_value=-1)

                    # Rounded coordinates for grouping
                    df['lat_rounded'] = (df['begin_lat'] * 10).round() / 10
                    df['lon_rounded'] = (df['begin_lon'] * 10).round() / 10

            except Exception as e:
                print(f"Warning: Geographic clustering failed: {e}")

    return df


def create_historical_pattern_features(df):
    """Create historical pattern features based on state-level trends"""
    if 'STATE' not in df.columns:
        return df

    # Multi-year rolling statistics for targets
    target_cols = ['DEATHS_DIRECT', 'INJURIES_DIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS']

    for window in [3, 5, 10]:
        for col in target_cols:
            if col in df.columns:
                df[f'{col.lower()}_{window}yr_avg'] = df.groupby('STATE')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

    # State-level risk scores
    if 'DAMAGE_PROPERTY' in df.columns:
        df['state_total_damage'] = df.groupby('STATE')['DAMAGE_PROPERTY'].transform('sum')
        df['state_event_count'] = df.groupby('STATE')['DAMAGE_PROPERTY'].transform('count')
        df['state_avg_damage'] = df['state_total_damage'] / df['state_event_count']

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
    """Complete preprocessing and feature engineering pipeline with enhanced features"""
    print("Starting comprehensive data preprocessing...")

    # Convert target columns to numeric first
    target_columns = ['DEATHS_DIRECT', 'INJURIES_DIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS']
    df = preprocess_target_columns(df, target_columns)

    # Convert date columns to datetime format first
    date_columns = [col for col in df.columns if 'date' in col.lower() and 'time' in col.lower()]
    if date_columns:
        for date_col in date_columns:
            print(f"Converting {date_col} to datetime...")
            df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y %H:%M:%S', errors='coerce')

    # Create categories from EVENT_TYPE if available (more detailed than CATEGORY)
    if 'EVENT_TYPE' in df.columns:
        df = categorize_by_event_type(df)
    elif 'CATEGORY' in df.columns:
        # Fallback to CATEGORY mapping if EVENT_TYPE not available
        df['CATEGORY'] = df['CATEGORY'].apply(map_category_codes)

    # Handle state column
    if 'STATE' in df.columns:
        df['STATE'] = df['STATE'].fillna('UNKNOWN').astype(str).str.upper()

    # Create enhanced features (now that dates are properly formatted)
    print("Creating enhanced features...")
    df = create_enhanced_temporal_features(df)
    df = create_enhanced_geographic_features(df)
    df = create_historical_pattern_features(df)
    df = create_impact_features(df)

    print(f"Preprocessing complete. Final shape: {df.shape}")
    return df
