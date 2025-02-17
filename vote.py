import pandas as pd
import numpy as np
import torch 
import matplotlib.pyplot as plt

HUGE = 10000000000

# TODO measure political spectrum: correspondence of other parties with strength of democratic or republican majority. More democratic majority? we're saying the associated parties are likely to be more left, or at least something...let's just look at that correlation, so that's one dimension we see if it separates anything.
# TODO second metric based on text or explicit ideology
# just make random political spectra and then adjust until people's first choice preferences match the house results.
# the ranked choice voting...:
# do this in a piece by piece aggregation
# each vote needs to be self-contained and have a single result communcated forward

# left house_er, right house

fi = "csv/1976-2020-house.csv"
house = extract_elections(fi)
fi = "election-results/election_results_house.csv"
house_fte = extract_elections(fi)
#house = pd.read_csv(fi)# stringsAsFactors=False)

# filter to just the new ones more recent than the old ones and combine
house_fte = house_fte[house_fte["year"] > house["year"].max()]
house_combined = pd.concat([house, house_fte])

# cleanup function
house_combined = house_cleanup(house_combined)

# take subset
# perform election (format) : single, dice, approval, rcv, etc...
# election results: Just add each win to "wins" column : tot_count, win_count
# format : needs
# check for .csv of election (format, subset, iter), read and if not, run the election
# single : winner (1, 0)
# dice : Roll*total_votes (in [0,total_votes]), winner (1,0)
# approval : approval count (in [0,total_votes]), winner (1, 0)
# quota : winner (1,0)
# rcv : choice count (1,2,3,...), average rank, winner | entire dataset of each voter and their ranks...(this is STUPID) 
# second df just stores sum of counts over many runs, so gives back a vector of winners for each iteration

# return frame with results of election. 
# part to rerun
n_iter = 200
fnames = []
for i in range(n_iter):
  house_combined["cumsum"] = house_combined.groupby(["year", "state_district"])["candidatevotes"].cumsum()

  fname = F"dice_win-{i}"
  fnames.append(fname)
  house_combined[fname] = 0
  print(F"running iteration {i+1}/{n_iter}")
  tv = house_combined.groupby(["year", "state_district"])["totalvotes"].max().reset_index()
  tv[F"rand-{i}"] = np.random.rand(len(tv))
  tv[F"die-{i}"] = tv["totalvotes"]*tv[F"rand-{i}"]
  tv = tv.rename({'totalvotes': F'tv-{i}'}, axis=1)

  house_iter = house_combined.merge(tv, on=["year", "state_district"])
  house_iter["win_tot"] = house_iter["cumsum"]-house_iter[F"die-{i}"]
  house_iter.loc[house_iter["win_tot"] < 0, "win_tot"] = HUGE
  house_iter["iter"] = i
  # TODO write this out. 

  house_combined = house_combined.merge(tv, on=["year", "state_district"])
  house_combined["win_tot"] = house_combined["cumsum"]-house_combined[F"die-{i}"]
  house_combined.loc[house_combined["win_tot"] < 0, "win_tot"] = HUGE
  #house_combined["win_pos"] = house_combined["win_tot"] > 0
  # note this has an issue if a candidate received zero votes, this will find both numbers n and n+0 and mark both as winners. Can't have that, so remove any candidates with 0 votes from house_combined

  # designates winner
  wdx = house_combined.groupby(["year", "state_district"], sort=False)["win_tot"].transform("min") == house_combined["win_tot"]
  house_combined.loc[wdx, fname] = 1
  


# # part out by year
# # year = 2020
# for year in house_combined["year"].unique():
#   y_house_combined = house_combined.loc[house_combined["year"] == year]
#   # by state and district
#   test = torch.rand(100000).numpy()
#   print(F"Running stochastic election for {int(year)}")
#   # get subset that is unique by state and district
#   for sd in y_house_combined["state_district"].unique():
#     y_sd_house_combined = y_house_combined.loc[y_house_combined["state_district"] == sd]
#     # gotta deal with runoff
#     # take total, pick random, multiply by total, if within first range, candidate 1, second range ...
#     total = y_sd_house_combined["candidatevotes"].sum()
#     test = y_sd_house_combined["candidatevotes"].cumsum()
#     die = np.random.rand()*total
#     win = ((test-die) > 0).idxmax()
#     house_combined.loc["dice_win",win] = 1

  
fo = "csv/1976-2020-house_combined-stoch_test.csv"
house_combined.to_csv(fo)

# aggregate by party and year

dnames = fnames
fnames.append("count_win")
house_combined["party_major"] = house_combined["party"]
house_combined.loc[(house_combined["party"] != "DEMOCRAT") & (house_combined["party"] != "REPUBLICAN"), "party_major"] = "OTHER"

control = house_combined.groupby(list(["year", "party_major"]))[fnames].sum().reset_index()
#control = control.loc[(control["dice_win"] > 0) | (control["count_win"] > 0)]

control_c = control.groupby(["year"])[fnames].sum().reset_index()
fo = "csv/1976-2020-house_combined-stoch_test-control.csv"
control.to_csv(fo)
# how to keep each histogram??!? don't want columns, do rows instead, sort, for intersection, compute histogram and plot. 
# TODO modify like senate so carries forward previoius winner unless this race changes. 
for year in control["year"].unique():
  # TODO make subplot showing popular vote by party like my stuff for euclid
  fig = plt.figure()
  control_y = control.loc[control["year"] == year]
  da = np.array(control_y.loc[control_y["party_major"] == "DEMOCRAT", dnames])[0]
  ra = np.array(control_y.loc[control_y["party_major"] == "REPUBLICAN", dnames])[0]
  oa = np.array(control_y.loc[control_y["party_major"] == "OTHER", dnames])[0]
  bins = np.histogram(np.hstack((da, ra, oa)), bins=60)[1]
  dh = plt.hist(da, bins, color="blue", alpha=0.3, label="Democrat")
  rh = plt.hist(ra, bins, color="red", alpha=0.3, label="Republican")  
  oh = plt.hist(oa, bins, color="green", alpha=0.3, label = "Other")
  ymax = np.array([dh[0], rh[0], oh[0]]).max()
  dc = np.array(control_y.loc[control_y["party_major"] == "DEMOCRAT", "count_win"])[0]
  rc = np.array(control_y.loc[control_y["party_major"] == "REPUBLICAN", "count_win"])[0]
  oc = np.array(control_y.loc[control_y["party_major"] == "OTHER", "count_win"])[0]
  plt.vlines(dc, 0, ymax, color="blue", label=F"By Count: {dc}")
  plt.vlines(rc, 0, ymax, color="red", label=F"By Count: {rc}")
  plt.vlines(oc, 0, ymax, color="green", label=F"By Count: {oc}")

  plt.vlines(da.mean(), 0, ymax, color="blue", label=F"Dice Mean: {da.mean()}", linestyles="dashed")
  plt.vlines(ra.mean(), 0, ymax, color="red", label=F"Dice Mean: {ra.mean()}", linestyles="dashed")
  plt.vlines(oa.mean(), 0, ymax, color="green", label=F"Dice Mean: {oa.mean()}", linestyles="dashed")
  plt.legend()
  plt.title(F"Histogram of Stochastic voting outcomes for House Election in {year}")
  plt.xlabel("Number of House Seats Awarded to each Party")
  plt.ylabel("Count of Scenarios for Each Balance of Power")
  fo = F"img/1976-2020-house-hist-{year}.png"
  fig.set_size_inches(14, 7)
  fig.savefig(fo)

# group other parties in control all together

# check unique state-districts in 2016 versus 2018
d20 = house_combined.loc[house_combined["year"] == 2020, "state_district"].unique()
d18 = house_combined.loc[house_combined["year"] == 2018, "state_district"].unique()
d16 = house_combined.loc[house_combined["year"] == 2016, "state_district"].unique()
house_combined_1976 = house_combined.loc[(house_combined["year"] == 2018) & (house_combined["state_district"] == "NEW YORK-2")]
house_combined_dice = house_combined_1976.loc[house_combined_1976["dice_win-0"] > 0]
hdg = house_combined_dice.groupby(["state_district"])["dice_win-0"].sum().reset_index()
# # questions: who wins more nailbiters?
# # how badly gerrymandered are different states?
# # how does state popular vote fraction relate with number of representatives apportioned by this system
# # how stable is this -- if some noise is added to each state/district (1 standard deviation) how stable is the voting preference
# # how stable is this under bias -- increase republican turnout by 1%, how much does the outcome change under each system?

# # notes on voting strategy
# #	gerymandering and districting due to lack of representation becomes irelevant
# # 	vote for your longshot candidate because it actually increases their chance of getting elected
# #       veto voting -- allow people to cast their vote negatively, ONLY decrease the chances of one candidate
# # makes it much harder for career politicians

# #       
