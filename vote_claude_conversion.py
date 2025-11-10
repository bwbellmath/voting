import pandas as pd
import json
import re

def import_election(input_file, conversion_json_file, election_type):
    """
    Import election data and standardize columns based on JSON mapping.
    
    Parameters:
    input_file (str): Path to input CSV file
    conversion_json_file (str): Path to JSON mapping file
    election_type (str): One of 'house', 'senate', or 'president'
    
    Returns:
    pd.DataFrame: Standardized election data DataFrame
    """
    # Read input data
    df = pd.read_csv(input_file)
    
    # Load conversion mapping
    with open(conversion_json_file, 'r') as f:
        mappings = json.load(f)
    
    mapping_key = f"{election_type}_mapping"
    if mapping_key not in mappings:
        raise ValueError(f"No mapping found for election type: {election_type}")
    
    # Determine format based on columns
    is_new_format = 'race_id' in df.columns
    format_key = 'new_format' if is_new_format else 'old_format'
    column_mapping = mappings[mapping_key][format_key]
    
    # Create new dataframe with standardized columns
    df_standard = pd.DataFrame(index=df.index)
    
    # Process each mapping
    for source_col, target_info in column_mapping.items():
        if isinstance(target_info, str):
            # Simple column rename
            if source_col in df.columns:
                df_standard[target_info] = df[source_col]
        else:
            # Complex transformation
            if target_info['type'] == 'constant':
                # For constant values
                target_col = [k for k, v in column_mapping.items() if v == target_info][0]
                df_standard[target_col] = target_info['value']
            
            elif target_info['type'] == 'extract':
                # For pattern extraction
                if source_col in df.columns:
                    extracted = df[source_col].fillna('').str.extract(target_info['pattern'], expand=False)
                    df_standard[target_info['target']] = extracted.fillna('0')
                    if target_info['target'] == 'district':
                        df_standard[target_info['target']] = df_standard[target_info['target']].str.zfill(3)
            
            elif target_info['type'] == 'derived':
                # For derived values
                if target_info['source'] in df.columns:
                    derived_col = [k for k, v in column_mapping.items() if v == target_info][0]
                    df_standard[derived_col] = (df[target_info['source']] == target_info['condition'])
    
    # Ensure numeric types for vote counts
    if 'candidatevotes' in df_standard.columns:
        df_standard['candidatevotes'] = pd.to_numeric(df_standard['candidatevotes'], errors='coerce').fillna(0)
    if 'totalvotes' in df_standard.columns:
        df_standard['totalvotes'] = pd.to_numeric(df_standard['totalvotes'], errors='coerce').fillna(0)
    
    # Convert strings to uppercase
    string_columns = ['state', 'state_po', 'candidate', 'party']
    for col in string_columns:
        if col in df_standard.columns:
            df_standard[col] = df_standard[col].fillna('').str.upper()
    
    return df_standard
