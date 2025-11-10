import pandas as pd
import numpy as np

def _handle_string_column(df, col_name, default=''):
    """Helper function to safely handle string column operations"""
    if col_name in df.columns:
        return df[col_name].fillna('').astype(str).replace('nan', '').str.upper()
    return pd.Series([default] * len(df))

def import_election_data(filepath, election_type):
    """
    Import election data from either old or new CSV format and standardize it.
    """
    if election_type not in ['house', 'senate', 'president']:
        raise ValueError("election_type must be one of: 'house', 'senate', 'president'")
    
    # Read CSV file
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Error reading CSV file: {str(e)}")
    
    # Determine format based on columns
    is_new_format = 'race_id' in df.columns
    
    try:
        if is_new_format:
            df = _process_new_format(df, election_type)
        else:
            df = _process_old_format(df, election_type)
        
        # Common processing for all types
        for col in ['party', 'candidate', 'state']:
            df[col] = _handle_string_column(df, col)
        
        # Type-specific processing
        if election_type == 'president':
            df = _process_president_specific(df)
        elif election_type == 'senate':
            df = _process_senate_specific(df)
        elif election_type == 'house':
            df = _process_house_specific(df)
        
        # Convert vote columns to numeric
        for col in ['candidatevotes', 'totalvotes']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error processing data: {str(e)}")

def _process_new_format(df, election_type):
    """Process new format data"""
    try:
        # Common mappings
        mapping = {
            'state_abbrev': 'state_po',
            'candidate_name': 'candidate',
            'ballot_party': 'party',
            'votes': 'candidatevotes',
            'cycle': 'year'
        }
        
        # Only rename columns that exist
        existing_cols = {k: v for k, v in mapping.items() if k in df.columns}
        df = df.rename(columns=existing_cols)
        
        # Handle string columns
        df['party'] = _handle_string_column(df, 'party')
        df['writein'] = df['party'] == 'W'
        df['mode'] = 'TOTAL'
        df['stage'] = _handle_string_column(df, 'stage', 'GEN')
        df['special'] = _handle_string_column(df, 'special', 'FALSE')
        df['runoff'] = 'FALSE'
        
        # Process district information based on election type
        if election_type == 'house':
            if 'office_seat_name' in df.columns:
                district_extract = df['office_seat_name'].fillna('').astype(str).str.extract(r'District (\d+)', expand=False)
                df['district'] = district_extract.fillna('0').str.zfill(3)
            else:
                df['district'] = '000'
                
        elif election_type == 'senate':
            df['district'] = 'statewide'
            if 'office_seat_name' in df.columns:
                class_extract = df['office_seat_name'].fillna('').astype(str).str.extract(r'Class (\d+)', expand=False)
                df['class'] = class_extract.fillna('0').astype(int)
            else:
                df['class'] = 0
                
        elif election_type == 'president':
            df['district'] = 'statewide'
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error processing new format data: {str(e)}")

def _process_old_format(df, election_type):
    """Process old format data"""
    try:
        if election_type == 'senate':
            if 'class' not in df.columns:
                df['class'] = ((df['year'] - 1976) // 2) % 3 + 1
        
        # Ensure basic required columns exist
        if 'writein' not in df.columns:
            df['writein'] = False
        if 'mode' not in df.columns:
            df['mode'] = 'TOTAL'
        if 'district' not in df.columns:
            df['district'] = 'statewide' if election_type in ['senate', 'president'] else '000'
            
        return df
        
    except Exception as e:
        raise ValueError(f"Error processing old format data: {str(e)}")

def _process_president_specific(df):
    """Add president-specific processing"""
    required_columns = [
        'year', 'state', 'state_po', 'candidate', 'party',
        'writein', 'candidatevotes', 'totalvotes'
    ]
    
    df['district'] = 'statewide'
    df['mode'] = 'TOTAL'
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    return df[required_columns]

def _process_senate_specific(df):
    """Add senate-specific processing"""
    required_columns = [
        'year', 'state', 'state_po', 'district', 'class',
        'stage', 'special', 'candidate', 'party', 'writein',
        'mode', 'candidatevotes', 'totalvotes'
    ]
    
    if all(col in df.columns for col in ['year', 'class']):
        df['year_class'] = ((((df['year']-1976)/2) % 3) + 1).astype(int)
        df['actually_special'] = df['special']
        df.loc[df['class'] == df['year_class'], 'actually_special'] = False
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    return df[required_columns]

def _process_house_specific(df):
    """Add house-specific processing"""
    required_columns = [
        'year', 'state', 'state_po', 'district', 'stage',
        'runoff', 'special', 'candidate', 'party', 'writein',
        'mode', 'candidatevotes', 'totalvotes'
    ]
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    return df[required_columns]

def validate_election_data(df, election_type):
    """Validate the imported election data meets requirements."""
    try:
        base_columns = ['year', 'state', 'state_po', 'candidate', 'party', 
                       'candidatevotes', 'totalvotes']
        
        type_specific_columns = {
            'house': ['district', 'stage', 'runoff', 'special'],
            'senate': ['district', 'class', 'stage', 'special'],
            'president': []
        }
        
        required_columns = base_columns + type_specific_columns[election_type]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        if not pd.to_numeric(df['candidatevotes'], errors='coerce').notnull().all():
            return False, "Invalid non-numeric values in candidatevotes"
        
        if election_type == 'senate':
            if 'class' in df.columns and not df['class'].between(1, 3).all():
                return False, "Invalid Senate class values"
        
        return True, "Validation passed"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def merge_election_data(df_old, df_new, election_type):
    """Merge two election dataframes, keeping newer data in case of duplicates."""
    base_keys = ['year', 'state', 'candidate', 'party']
    type_specific_keys = {
        'house': ['district'],
        'senate': ['class'],
        'president': []
    }
    
    merge_keys = base_keys + type_specific_keys[election_type]
    
    df_combined = pd.concat([df_old, df_new])
    
    df_merged = df_combined.sort_values(merge_keys).drop_duplicates(
        subset=merge_keys,
        keep='last'
    )
    
    sort_cols = ['year', 'state']
    if election_type == 'senate':
        sort_cols.extend(['class', 'candidatevotes'])
    elif election_type == 'house':
        sort_cols.extend(['district', 'candidatevotes'])
    else:  # president
        sort_cols.append('candidatevotes')
        
    df_merged = df_merged.sort_values(sort_cols, ascending=[True] * (len(sort_cols)-1) + [False])
    
    return df_merged
