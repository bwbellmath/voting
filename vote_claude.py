import pandas as pd
import numpy as np

from vote_utils import (
    import_election_data,
    validate_election_data,
    merge_election_data,
    _process_new_format,
    _process_old_format,
    _process_president_specific,
    _process_senate_specific,
    _process_house_specific
)

from vote_claude_conversion import import_election

fi_json = "election_mappings.json"

# Import new datasets
fi_house  = "election-results/election_results_house.csv"
fi_pres   = "election-results/election_results_presidential.csv"
fi_senate = "election-results/election_results_senate.csv"

df_house_new = import_election(fi_house, fi_json, "house")
df_senate_new = import_election(fi_senate, fi_json, "senate")
df_pres_new = import_election(fi_pres, fi_json, "president")

# Import old datasets
fi_house  = "csv/1976-2020-house.csv"
fi_senate = "csv/1976-2020-senate.csv"
fi_pres   = "csv/1976-2020-president.csv"

df_house_old = import_election(fi_house, fi_json, "house")
df_senate_old = import_election(fi_senate, fi_json, "senate")
df_pres_old = import_election(fi_pres, fi_json, "president")

# Merge datasets
df_house_merged = pd.concat([df_house_old, df_house_new]).drop_duplicates(subset=['year', 'state', 'district', 'candidate', 'party'], keep='last')
df_senate_merged = pd.concat([df_senate_old, df_senate_new]).drop_duplicates(subset=['year', 'state', 'district', 'candidate', 'party'], keep='last')
df_pres_merged = pd.concat([df_pres_old, df_pres_new]).drop_duplicates(subset=['year', 'state', 'candidate', 'party'], keep='last')

# now given a dataframe and a

# president: for each year
# dice rolls by state, and elector
# regular voting, state's electors all go for the max candidate
# except maine and nevada, write custom rules
# dump lines to file with elector and simulated choices aggregated
# plot samples of number of electors per candidate, average number of electors, and actual number of electors 

# house : for each year
# dice rolls by state and district
# count up election by party
