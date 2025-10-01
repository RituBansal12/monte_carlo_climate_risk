"""
Data Cleaning Module

This module provides comprehensive data cleaning and preprocessing functionality
for climate disaster datasets, including damage value conversion and coordinate validation.
"""

import pandas as pd
import numpy as np
from typing import Dict


class DataCleaner:
    """
    Handles data cleaning and preprocessing for climate disaster datasets.

    This class provides methods to clean numeric data, convert damage values,
    validate coordinates, and prepare data for feature engineering.
    """

    def __init__(self, config: Dict):
        """Initialize the data cleaner with configuration settings."""
        self.config = config

    def clean_sed_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean SED details dataset with comprehensive preprocessing.

        This method handles missing values, converts damage strings to numeric,
        standardizes dates, and validates coordinates.

        Args:
            df: Raw SED details DataFrame

        Returns:
            Cleaned and preprocessed DataFrame
        """
        # Handle missing values for numeric columns that exist in the dataset
        possible_numeric_columns = ['injuries_direct', 'injuries_indirect',
                                   'deaths_direct', 'deaths_indirect',
                                   'damage_property', 'damage_crops']

        # Only process columns that actually exist in the dataset
        numeric_columns = [col for col in possible_numeric_columns if col in df.columns]

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)

        # Convert damage to numeric values (handle K, M, B suffixes)
        df = self._convert_damage_to_numeric(df)

        # Standardize date formats
        if 'begin_date_time' in df.columns:
            df['begin_date_time'] = pd.to_datetime(df['begin_date_time'],
                                                  format='%m/%d/%Y %H:%M:%S',
                                                  errors='coerce')

        # Validate coordinate data
        df = self._validate_coordinates(df)

        return df

    def _convert_damage_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert damage strings like '10.00K' to numeric values.

        Handles K (thousand), M (million), and B (billion) suffixes.

        Args:
            df: DataFrame containing damage columns

        Returns:
            DataFrame with numeric damage values
        """
        def convert_damage(value):
            if pd.isna(value) or value == 0:
                return 0.0

            value_str = str(value).upper().replace(',', '')
            multiplier = 1

            if value_str.endswith('K'):
                multiplier = 1000
                value_str = value_str[:-1]
            elif value_str.endswith('M'):
                multiplier = 1000000
                value_str = value_str[:-1]
            elif value_str.endswith('B'):
                multiplier = 1000000000
                value_str = value_str[:-1]

            try:
                return float(value_str) * multiplier
            except ValueError:
                return 0.0

        # Only process damage columns that exist in the dataset
        damage_columns = ['damage_property', 'damage_crops']
        for col in damage_columns:
            if col in df.columns:
                df[col] = df[col].apply(convert_damage)

        return df

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean coordinate data.

        Removes rows with missing or invalid coordinates and validates ranges.

        Args:
            df: DataFrame containing coordinate columns

        Returns:
            DataFrame with valid coordinates only
        """
        # Remove invalid coordinates only if columns exist
        coord_columns = ['begin_lat', 'begin_lon', 'end_lat', 'end_lon']
        existing_coord_columns = [col for col in coord_columns if col in df.columns]

        if len(existing_coord_columns) >= 4:  # All coordinate columns exist
            df = df[df['begin_lat'].notna() & df['begin_lon'].notna()]
            df = df[df['end_lat'].notna() & df['end_lon'].notna()]

            # Validate coordinate ranges
            df = df[(df['begin_lat'].between(-90, 90)) &
                    (df['begin_lon'].between(-180, 180))]

        return df
