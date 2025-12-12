"""
Data loading modules for survey and election data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SurveyData:
    """Container for survey data"""
    date: pd.Series
    republicans: pd.Series
    independents: pd.Series
    democrats: pd.Series
    source: str

    def to_dict(self):
        """Convert to dictionary for JSON export"""
        return {
            'date': self.date.tolist(),
            'republicans': self.republicans.tolist(),
            'independents': self.independents.tolist(),
            'democrats': self.democrats.tolist(),
            'source': self.source
        }


@dataclass
class LeanerSurveyData:
    """Container for leaner survey data (includes independents who lean toward parties)"""
    date: pd.Series
    republican_leaners: pd.Series
    democratic_leaners: pd.Series
    source: str

    def to_dict(self):
        """Convert to dictionary for JSON export"""
        return {
            'date': self.date.tolist(),
            'republican_leaners': self.republican_leaners.tolist(),
            'democratic_leaners': self.democratic_leaners.tolist(),
            'source': self.source
        }


@dataclass
class ElectionResults:
    """Container for election results"""
    year: pd.Series
    state: pd.Series
    office: pd.Series
    candidate: pd.Series
    party: pd.Series
    votes: pd.Series
    total_votes: pd.Series

    def to_dict(self):
        """Convert to dictionary for JSON export"""
        return {
            'year': self.year.tolist(),
            'state': self.state.tolist(),
            'office': self.office.tolist(),
            'candidate': self.candidate.tolist(),
            'party': self.party.tolist(),
            'votes': self.votes.tolist(),
            'total_votes': self.total_votes.tolist()
        }


class SurveyLoader:
    """Load and process survey data"""

    def __init__(self, data_dir: str = "csv"):
        self.data_dir = Path(data_dir)

    def load_party_identification(self) -> SurveyData:
        """
        Load party identification survey data (data-1aTsg.csv).
        This shows Republicans, Independents, and Democrats over time.
        """
        filepath = self.data_dir / "data-1aTsg.csv"
        df = pd.read_csv(filepath)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Parse dates from the X.1 column (format: "2024 Feb 1-20")
        dates = pd.to_datetime(df['X.1'].str.extract(r'(\d{4}\s+\w+)')[0], format='%Y %b')

        # Convert percentage strings to floats
        republicans = df['Republicans'].astype(float)
        independents = df['Independents'].astype(float)
        democrats = df['Democrats'].astype(float)

        return SurveyData(
            date=dates,
            republicans=republicans,
            independents=independents,
            democrats=democrats,
            source="data-1aTsg.csv"
        )

    def load_leaner_data(self) -> LeanerSurveyData:
        """
        Load leaner survey data (data-gOKi4.csv).
        This shows Republicans/Republican leaners vs Democrats/Democratic leaners.
        """
        filepath = self.data_dir / "data-gOKi4.csv"
        df = pd.read_csv(filepath)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Parse dates
        dates = pd.to_datetime(df['X.1'].str.extract(r'(\d{4}\s+\w+)')[0], format='%Y %b')

        # Convert percentage strings to floats
        republican_leaners = df['Republicans/Republican leaners'].astype(float)
        democratic_leaners = df['Democrats/Democratic leaners'].astype(float)

        return LeanerSurveyData(
            date=dates,
            republican_leaners=republican_leaners,
            democratic_leaners=democratic_leaners,
            source="data-gOKi4.csv"
        )


class ElectionLoader:
    """Load and process election results"""

    def __init__(self, data_dir: str = "csv"):
        self.data_dir = Path(data_dir)

    def load_presidential_results(self,
                                   years: Optional[List[int]] = None,
                                   states: Optional[List[str]] = None) -> ElectionResults:
        """
        Load presidential election results.

        Args:
            years: Filter by specific years (None = all years)
            states: Filter by specific states (None = all states)
        """
        filepath = self.data_dir / "1976-2020-president.csv"
        df = pd.read_csv(filepath)

        # Apply filters
        if years is not None:
            df = df[df['year'].isin(years)]
        if states is not None:
            df = df[df['state'].isin(states)]

        return ElectionResults(
            year=df['year'],
            state=df['state'],
            office=df['office'],
            candidate=df['candidate'],
            party=df['party_simplified'],
            votes=df['candidatevotes'],
            total_votes=df['totalvotes']
        )

    def load_senate_results(self,
                           years: Optional[List[int]] = None,
                           states: Optional[List[str]] = None) -> ElectionResults:
        """Load Senate election results."""
        filepath = self.data_dir / "1976-2020-senate.csv"
        df = pd.read_csv(filepath)

        # Apply filters
        if years is not None:
            df = df[df['year'].isin(years)]
        if states is not None:
            df = df[df['state'].isin(states)]

        return ElectionResults(
            year=df['year'],
            state=df['state'],
            office=df['office'],
            candidate=df['candidate'],
            party=df['party_simplified'],
            votes=df['candidatevotes'],
            total_votes=df['totalvotes']
        )

    def load_house_results(self,
                          years: Optional[List[int]] = None,
                          states: Optional[List[str]] = None,
                          districts: Optional[List[str]] = None) -> ElectionResults:
        """Load House election results."""
        filepath = self.data_dir / "1976-2020-house.csv"
        df = pd.read_csv(filepath)

        # Apply filters
        if years is not None:
            df = df[df['year'].isin(years)]
        if states is not None:
            df = df[df['state'].isin(states)]
        if districts is not None:
            df = df[df['district'].isin(districts)]

        return ElectionResults(
            year=df['year'],
            state=df['state'],
            office=df['office'],
            candidate=df['candidate'],
            party=df['party'],
            votes=df['candidatevotes'],
            total_votes=df['totalvotes']
        )


class DataAggregator:
    """Aggregate election and survey data in various ways"""

    @staticmethod
    def aggregate_by_year(results: ElectionResults) -> pd.DataFrame:
        """Aggregate election results by year"""
        df = pd.DataFrame({
            'year': results.year,
            'party': results.party,
            'votes': results.votes
        })

        return df.groupby(['year', 'party'])['votes'].sum().reset_index()

    @staticmethod
    def aggregate_by_state(results: ElectionResults) -> pd.DataFrame:
        """Aggregate election results by state"""
        df = pd.DataFrame({
            'state': results.state,
            'party': results.party,
            'votes': results.votes
        })

        return df.groupby(['state', 'party'])['votes'].sum().reset_index()

    @staticmethod
    def aggregate_by_year_state(results: ElectionResults) -> pd.DataFrame:
        """Aggregate election results by year and state"""
        df = pd.DataFrame({
            'year': results.year,
            'state': results.state,
            'party': results.party,
            'votes': results.votes
        })

        return df.groupby(['year', 'state', 'party'])['votes'].sum().reset_index()

    @staticmethod
    def calculate_vote_shares(results: ElectionResults) -> pd.DataFrame:
        """Calculate vote share percentages by party"""
        df = pd.DataFrame({
            'year': results.year,
            'state': results.state,
            'party': results.party,
            'votes': results.votes,
            'total_votes': results.total_votes
        })

        df['vote_share'] = (df['votes'] / df['total_votes']) * 100

        return df[['year', 'state', 'party', 'vote_share']]

    @staticmethod
    def pivot_vote_shares(vote_shares: pd.DataFrame) -> pd.DataFrame:
        """Pivot vote shares to wide format (one column per party)"""
        return vote_shares.pivot_table(
            index=['year', 'state'],
            columns='party',
            values='vote_share',
            fill_value=0
        ).reset_index()


def export_for_visualization(output_path: str = "viz_data.json"):
    """Export all data in a format suitable for D3 visualization"""
    import json

    # Load all data
    survey_loader = SurveyLoader()
    election_loader = ElectionLoader()

    party_id = survey_loader.load_party_identification()
    leaners = survey_loader.load_leaner_data()

    presidential = election_loader.load_presidential_results()
    senate = election_loader.load_senate_results()

    # Aggregate election data
    aggregator = DataAggregator()
    pres_by_year = aggregator.aggregate_by_year(presidential).to_dict('records')
    pres_by_state = aggregator.aggregate_by_state(presidential).to_dict('records')
    pres_vote_shares = aggregator.calculate_vote_shares(presidential).to_dict('records')

    senate_by_year = aggregator.aggregate_by_year(senate).to_dict('records')

    # Package everything
    data = {
        'surveys': {
            'party_identification': party_id.to_dict(),
            'leaners': leaners.to_dict()
        },
        'elections': {
            'presidential': {
                'by_year': pres_by_year,
                'by_state': pres_by_state,
                'vote_shares': pres_vote_shares
            },
            'senate': {
                'by_year': senate_by_year
            }
        }
    }

    # Export to JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Exported visualization data to {output_path}")
    return data


if __name__ == "__main__":
    # Test the loaders
    print("Testing Survey Loader...")
    survey_loader = SurveyLoader()
    party_id = survey_loader.load_party_identification()
    print(f"Loaded {len(party_id.date)} party identification records")

    leaners = survey_loader.load_leaner_data()
    print(f"Loaded {len(leaners.date)} leaner records")

    print("\nTesting Election Loader...")
    election_loader = ElectionLoader()
    presidential = election_loader.load_presidential_results(years=[2020])
    print(f"Loaded {len(presidential.year)} presidential records for 2020")

    print("\nExporting data for visualization...")
    export_for_visualization()
