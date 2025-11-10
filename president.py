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
pres_electors = house_electors.loc[(house_electors["year"]-1976)%4 == 0]

#    group by year, state to get congress electors

# DC gets 3 electors
pres = pres.merge(pres_electors, on=["year", "state_po"], how="left")
pres.loc[pres["electors"].isnull(), "electors"] = 0
pres["electors"] = pres["electors"].astype(int)
pres["electors"] += 2
pres.loc[pres["state_po"] == "DC", "electors"] += 1
# every state gets 2 for their senators

# each elector will roll a die
#generate random numbers of size 
# their vote will be according to the state's weight
# TODO : try doing this with the weight from each congressional district?
size = np.random.rand(pres.shape[0]*pres["electors"].max()).reshape(pres.shape[0],pres["electors"].max())
threshold = pres["candidatevotes"]/pres["totalvotes"]
n_iter = 200
fnames = np.arange(55)
tv = pres.copy()
tv[fnames.astype(str)] = 0
tv[fnames.astype(str)]

for i in range(n_iter):
  fname = F"dice_win-{i}"
  #fnames.append(fname)
  house[fname] = 0
  print(F"running iteration {i+1}/{n_iter}")
  tv = house.groupby(["year", "state_district"])["totalvotes"].max().reset_index()
  tv[F"rand-{i}"] = np.random.rand(len(tv))
  tv[F"die-{i}"] = tv["totalvotes"]*tv[F"rand-{i}"]
  tv = tv.rename({'totalvotes': F'tv-{i}'}, axis=1)

  house = house.merge(tv, on=["year", "state_district"])
  house["win_tot"] = house["cumsum"]-house[F"die-{i}"]
  house.loc[house["win_tot"] < 0, "win_tot"] = HUGE
