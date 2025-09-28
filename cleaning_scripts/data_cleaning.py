import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

class DataCleaner:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def clean_sed_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean SED details dataset with comprehensive preprocessing"""
        # Handle missing values
        numeric_columns = ['injuries_direct', 'injuries_indirect',
                          'deaths_direct', 'deaths_indirect',
                          'damage_property', 'damage_crops']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)

        # Convert damage to numeric values (handle K, M, B suffixes)
        df = self._convert_damage_to_numeric(df)

        # Standardize date formats
        df['begin_date_time'] = pd.to_datetime(df['begin_date_time'],
                                              format='%m/%d/%Y %H:%M:%S',
                                              errors='coerce')

        # Validate coordinate data
        df = self._validate_coordinates(df)

        return df

    def _convert_damage_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert damage strings like '10.00K' to numeric values"""
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

        df['damage_property'] = df['damage_property'].apply(convert_damage)
        df['damage_crops'] = df['damage_crops'].apply(convert_damage)

        return df

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean coordinate data"""
        # Remove invalid coordinates
        df = df[df['begin_lat'].notna() & df['begin_lon'].notna()]
        df = df[df['end_lat'].notna() & df['end_lon'].notna()]

        # Validate coordinate ranges
        df = df[(df['begin_lat'].between(-90, 90)) &
                (df['begin_lon'].between(-180, 180))]

        return df
