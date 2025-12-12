"""
Data preprocessing pipeline for voter embeddings.
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessedData:
    """Container for preprocessed data"""
    features: torch.Tensor
    labels: Optional[torch.Tensor]
    feature_names: List[str]
    scaler: StandardScaler
    encoders: Dict[str, LabelEncoder]


class VoterPreprocessor:
    """Preprocess voter data for neural network training"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.feature_names = []

    def encode_demographics(self,
                           df: pd.DataFrame,
                           categorical_cols: List[str],
                           numerical_cols: List[str],
                           onehot_cols: Optional[List[str]] = None) -> torch.Tensor:
        """
        Encode demographic features.

        Args:
            df: DataFrame with demographic data
            categorical_cols: Columns to label encode
            numerical_cols: Columns to normalize
            onehot_cols: Columns to one-hot encode (optional)

        Returns:
            Tensor of encoded features
        """
        encoded_features = []
        self.feature_names = []

        # Label encode categorical columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))
            else:
                encoded = self.label_encoders[col].transform(df[col].fillna('Unknown'))

            encoded_features.append(encoded.reshape(-1, 1))
            self.feature_names.append(col)

        # One-hot encode specified columns
        if onehot_cols:
            for col in onehot_cols:
                if col not in self.onehot_encoders:
                    self.onehot_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = self.onehot_encoders[col].fit_transform(df[[col]].fillna('Unknown'))
                else:
                    encoded = self.onehot_encoders[col].transform(df[[col]].fillna('Unknown'))

                encoded_features.append(encoded)
                # Add feature names for each one-hot category
                categories = self.onehot_encoders[col].categories_[0]
                self.feature_names.extend([f"{col}_{cat}" for cat in categories])

        # Normalize numerical columns
        if numerical_cols:
            numerical_data = df[numerical_cols].fillna(0).values
            if not hasattr(self.scaler, 'mean_'):
                normalized = self.scaler.fit_transform(numerical_data)
            else:
                normalized = self.scaler.transform(numerical_data)

            encoded_features.append(normalized)
            self.feature_names.extend(numerical_cols)

        # Concatenate all features
        if encoded_features:
            features_array = np.concatenate(encoded_features, axis=1)
            return torch.FloatTensor(features_array)
        else:
            return torch.FloatTensor([])

    def encode_party_affiliation(self, party_series: pd.Series) -> torch.Tensor:
        """
        Encode party affiliation as one-hot vector.

        Args:
            party_series: Series with party labels (DEMOCRAT, REPUBLICAN, OTHER, etc.)

        Returns:
            One-hot encoded tensor
        """
        # Simplify party labels
        simplified = party_series.str.upper().replace({
            'DEMOCRAT': 'DEMOCRAT',
            'REPUBLICAN': 'REPUBLICAN',
            'LIBERTARIAN': 'OTHER',
            'GREEN': 'OTHER',
            'INDEPENDENT': 'INDEPENDENT'
        }).fillna('OTHER')

        # One-hot encode
        if 'party' not in self.onehot_encoders:
            self.onehot_encoders['party'] = OneHotEncoder(
                categories=[['DEMOCRAT', 'REPUBLICAN', 'INDEPENDENT', 'OTHER']],
                sparse_output=False,
                handle_unknown='ignore'
            )
            encoded = self.onehot_encoders['party'].fit_transform(simplified.values.reshape(-1, 1))
        else:
            encoded = self.onehot_encoders['party'].transform(simplified.values.reshape(-1, 1))

        return torch.FloatTensor(encoded)

    def normalize_vote_shares(self, vote_shares: pd.DataFrame) -> torch.Tensor:
        """
        Normalize vote shares to sum to 1 (convert to probability distribution).

        Args:
            vote_shares: DataFrame with vote share columns

        Returns:
            Normalized vote share tensor
        """
        vote_array = vote_shares.values
        # Ensure non-negative
        vote_array = np.maximum(vote_array, 0)
        # Normalize to sum to 1
        row_sums = vote_array.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        normalized = vote_array / row_sums

        return torch.FloatTensor(normalized)

    def create_temporal_features(self, dates: pd.Series) -> torch.Tensor:
        """
        Create temporal features from dates.

        Args:
            dates: Series of datetime objects

        Returns:
            Tensor with temporal features (year, month, day_of_year, etc.)
        """
        dates = pd.to_datetime(dates)

        features = np.stack([
            dates.dt.year.values,
            dates.dt.month.values,
            dates.dt.day_of_year.values,
            np.sin(2 * np.pi * dates.dt.day_of_year / 365),  # Cyclical encoding
            np.cos(2 * np.pi * dates.dt.day_of_year / 365)
        ], axis=1)

        return torch.FloatTensor(features)

    def aggregate_district_level(self,
                                 election_results: pd.DataFrame,
                                 group_by: List[str] = ['year', 'state']) -> pd.DataFrame:
        """
        Aggregate election results to district/state level.

        Args:
            election_results: DataFrame with election results
            group_by: Columns to group by

        Returns:
            Aggregated DataFrame
        """
        # Calculate vote shares by party
        agg_df = election_results.groupby(group_by + ['party']).agg({
            'votes': 'sum',
            'total_votes': 'first'  # Total votes should be same within group
        }).reset_index()

        # Calculate vote share percentage
        agg_df['vote_share'] = (agg_df['votes'] / agg_df['total_votes']) * 100

        # Pivot to wide format
        pivot_df = agg_df.pivot_table(
            index=group_by,
            columns='party',
            values='vote_share',
            fill_value=0
        ).reset_index()

        return pivot_df


class SurveyPreprocessor:
    """Preprocess survey data specifically"""

    @staticmethod
    def interpolate_missing_dates(df: pd.DataFrame,
                                  date_col: str = 'date',
                                  freq: str = 'M') -> pd.DataFrame:
        """
        Interpolate survey data for missing dates.

        Args:
            df: DataFrame with survey data
            date_col: Name of date column
            freq: Frequency for interpolation ('M' = monthly, 'D' = daily)

        Returns:
            DataFrame with interpolated values
        """
        df = df.set_index(date_col).sort_index()

        # Create complete date range
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)

        # Reindex and interpolate
        df_interpolated = df.reindex(full_range).interpolate(method='linear')

        return df_interpolated.reset_index().rename(columns={'index': date_col})

    @staticmethod
    def align_survey_and_elections(survey_df: pd.DataFrame,
                                   election_df: pd.DataFrame,
                                   survey_date_col: str = 'date',
                                   election_date_col: str = 'year') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align survey data with election dates.

        Args:
            survey_df: DataFrame with survey data (with dates)
            election_df: DataFrame with election results (with years)
            survey_date_col: Name of survey date column
            election_date_col: Name of election date column

        Returns:
            Tuple of (aligned surveys, aligned elections)
        """
        # Convert election years to datetime
        election_df['date'] = pd.to_datetime(election_df[election_date_col].astype(str) + '-11-01')

        # For each election, find nearest survey
        aligned_surveys = []
        aligned_elections = []

        for _, election_row in election_df.iterrows():
            election_date = election_row['date']

            # Find nearest survey date
            survey_df['time_diff'] = abs((survey_df[survey_date_col] - election_date).dt.days)
            nearest_survey_idx = survey_df['time_diff'].idxmin()
            nearest_survey = survey_df.loc[nearest_survey_idx]

            aligned_surveys.append(nearest_survey)
            aligned_elections.append(election_row)

        return pd.DataFrame(aligned_surveys), pd.DataFrame(aligned_elections)


def create_training_dataset(survey_data: pd.DataFrame,
                           election_data: pd.DataFrame,
                           preprocessor: VoterPreprocessor) -> Dict[str, torch.Tensor]:
    """
    Create a complete training dataset combining surveys and elections.

    Args:
        survey_data: Preprocessed survey data
        election_data: Preprocessed election data
        preprocessor: Fitted VoterPreprocessor instance

    Returns:
        Dictionary with train/val/test splits
    """
    # This is a placeholder for the full implementation
    # Will be expanded based on specific model requirements
    pass


if __name__ == "__main__":
    # Test preprocessing
    from data_loaders import SurveyLoader, ElectionLoader

    print("Testing preprocessing pipeline...")

    # Load data
    survey_loader = SurveyLoader()
    party_id = survey_loader.load_party_identification()

    # Create DataFrame for testing
    survey_df = pd.DataFrame({
        'date': party_id.date,
        'republicans': party_id.republicans,
        'independents': party_id.independents,
        'democrats': party_id.democrats
    })

    # Test temporal features
    preprocessor = VoterPreprocessor()
    temporal_features = preprocessor.create_temporal_features(survey_df['date'])
    print(f"Temporal features shape: {temporal_features.shape}")

    # Test vote share normalization
    vote_shares = survey_df[['republicans', 'independents', 'democrats']]
    normalized = preprocessor.normalize_vote_shares(vote_shares)
    print(f"Normalized vote shares shape: {normalized.shape}")
    print(f"Sample normalized row (should sum to ~1.0): {normalized[0].sum()}")

    print("\nPreprocessing tests passed!")
