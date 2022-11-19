import pandas as pd
import numpy as np
import torch 
import matplotlib.pyplot as plt

HUGE = 10000000000
fi = "csv/1976-2020-president.csv"
pres = pd.read_csv(fi)# stringsAsFactors=False)

# every state gets their number of house reps
fi = "csv/1976-2020-house.csv"
house = pd.read_csv(fi)# stringsAsFactors=False)
house = house.loc[house["state_po"] != "DC"]
#    get unique year, state, district from house.csv
house_districts = house.loc[:, ["year", "state_po", "district"]].drop_duplicates()
house_districts["electors"] = 1
house_electors = house_districts.groupby(["year", "state_po"])["electors"].sum().reset_index()

#    group by year, state to get congress electors

# DC gets 3 electors
pres = pres.merge(house_electors, on=["year", "state_po"], how="outer")
pres.loc[pres["electors"].isnull(), "electors"] = 0
pres["electors"] = pres["electors"].astype(int)
pres["electors"] += 2
pres.loc[pres["state_po"] == "DC", "electors"] += 1
# every state gets 2 for their senators

# each elector will roll a die
# their vote will be according to the state's weight
# TODO : try doing this with the weight from each congressional district?
