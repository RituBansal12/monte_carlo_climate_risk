"""
Feature Engineering Module

This module provides feature engineering functionality for climate disaster data,
including temporal features, geographic features, impact features, and disaster categorization.
"""

import pandas as pd
import numpy as np
from typing import Dict
from geopy.distance import geodesic


class FeatureEngineer:
    """
    Handles feature engineering for climate disaster datasets.

    Creates temporal, geographic, and impact-based features from raw disaster data.
    Also categorizes disasters into broader categories for analysis.
    """

    def __init__(self):
        """
        Initialize the feature engineer with disaster category definitions.
        """
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
        """
        Create time-based features from date information.

        Extracts year, month, quarter, seasonal indicators, and weekend flags.

        Args:
            df: DataFrame with date information

        Returns:
            DataFrame with temporal features added
        """
        # Create begin_date_time from components if it doesn't exist
        if 'begin_date_time' not in df.columns:
            df = self._create_datetime_column(df)

        if 'begin_date_time' in df.columns:
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

    def _create_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create begin_date_time column from date components.

        Converts BEGIN_YEARMONTH, BEGIN_DAY, BEGIN_TIME into a proper datetime column.

        Args:
            df: DataFrame with raw date components

        Returns:
            DataFrame with datetime column added
        """
        try:
            # Check if we have the required components
            if 'BEGIN_YEARMONTH' in df.columns and 'BEGIN_DAY' in df.columns and 'BEGIN_TIME' in df.columns:
                # Convert BEGIN_YEARMONTH to year and month
                df['BEGIN_YEARMONTH'] = pd.to_numeric(df['BEGIN_YEARMONTH'], errors='coerce')
                df['year'] = df['BEGIN_YEARMONTH'] // 100
                df['month'] = df['BEGIN_YEARMONTH'] % 100
                df['day'] = df['BEGIN_DAY']

                df['BEGIN_TIME'] = df['BEGIN_TIME'].astype(str).str.zfill(4)
                df['hour'] = df['BEGIN_TIME'].str[:2].astype(int)
                df['minute'] = df['BEGIN_TIME'].str[2:].astype(int)

                # Create datetime column
                df['begin_date_time'] = pd.to_datetime(
                    df['year'].astype(str) + '-' +
                    df['month'].astype(str).str.zfill(2) + '-' +
                    df['day'].astype(str).str.zfill(2) + ' ' +
                    df['hour'].astype(str).str.zfill(2) + ':' +
                    df['minute'].astype(str).str.zfill(2),
                    format='%Y-%m-%d %H:%M',
                    errors='coerce'
                )

                # Drop temporary columns
                df = df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1, errors='ignore')

        except Exception as e:
            print(f"Warning: Could not create datetime column: {e}")

        return df

    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create geography-based features.

        Calculates event duration and path length for geographic events.

        Args:
            df: DataFrame with coordinate information

        Returns:
            DataFrame with geographic features added
        """
        # Calculate event duration only if both date columns exist
        if 'begin_date_time' in df.columns and 'end_date_time' in df.columns:
            df['duration_hours'] = (df['end_date_time'] - df['begin_date_time']).dt.total_seconds() / 3600

        # Calculate path length for tornadoes and other path-based events
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

    def create_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create impact severity features.

        Calculates total deaths, injuries, damage, and creates impact severity scores.

        Args:
            df: DataFrame with impact data

        Returns:
            DataFrame with impact features added
        """
        # Total impact scores - only calculate if columns exist
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
        """
        Categorize events into broader disaster categories.

        Maps specific event types to broader disaster categories for analysis.

        Args:
            df: DataFrame with event type information

        Returns:
            DataFrame with disaster_category column added
        """
        df['disaster_category'] = 'other'

        # Check for event type column (try both lowercase and uppercase)
        event_col = None
        if 'event_type' in df.columns:
            event_col = 'event_type'
        elif 'EVENT_TYPE' in df.columns:
            event_col = 'EVENT_TYPE'

        if event_col:
            for category, events in self.disaster_categories.items():
                mask = df[event_col].isin(events)
                df.loc[mask, 'disaster_category'] = category

        return df

    def _create_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create begin_date_time column from date components"""
        try:
            # Check if we have the required components
            if 'BEGIN_YEARMONTH' in df.columns and 'BEGIN_DAY' in df.columns and 'BEGIN_TIME' in df.columns:
                # Convert BEGIN_YEARMONTH to year and month
                df['BEGIN_YEARMONTH'] = pd.to_numeric(df['BEGIN_YEARMONTH'], errors='coerce')
                df['year'] = df['BEGIN_YEARMONTH'] // 100
                df['month'] = df['BEGIN_YEARMONTH'] % 100
                df['day'] = df['BEGIN_DAY']

                df['BEGIN_TIME'] = df['BEGIN_TIME'].astype(str).str.zfill(4)
                df['hour'] = df['BEGIN_TIME'].str[:2].astype(int)
                df['minute'] = df['BEGIN_TIME'].str[2:].astype(int)

                # Create datetime column
                df['begin_date_time'] = pd.to_datetime(
                    df['year'].astype(str) + '-' +
                    df['month'].astype(str).str.zfill(2) + '-' +
                    df['day'].astype(str).str.zfill(2) + ' ' +
                    df['hour'].astype(str).str.zfill(2) + ':' +
                    df['minute'].astype(str).str.zfill(2),
                    format='%Y-%m-%d %H:%M',
                    errors='coerce'
                )

                # Drop temporary columns
                df = df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1, errors='ignore')

        except Exception as e:
            print(f"Warning: Could not create datetime column: {e}")

        return df

    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'begin_date_time' in df.columns and 'end_date_time' in df.columns:
            df['duration_hours'] = (df['end_date_time'] - df['begin_date_time']).dt.total_seconds() / 3600

        # Calculate path length for tornadoes and other path-based events
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

    def create_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create impact severity features"""
        # Total impact scores - only calculate if columns exist
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

        # Check for event type column (try both lowercase and uppercase)
        event_col = None
        if 'event_type' in df.columns:
            event_col = 'event_type'
        elif 'EVENT_TYPE' in df.columns:
            event_col = 'EVENT_TYPE'

        if event_col:
            for category, events in self.disaster_categories.items():
                mask = df[event_col].isin(events)
                df.loc[mask, 'disaster_category'] = category

        return df
