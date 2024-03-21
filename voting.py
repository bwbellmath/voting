import pandas as pd
import numpy as np
import torch 
import matplotlib.pyplot as plt

from utils.utils import *

HUGE = 10000000000
TEENY_WEENY = 0.0000000001

fo_md = "README.md"
fo_tex = "voting.tex"

report = Report(fo_md, fo_tex)

# TODO measure political spectrum: correspondence of other parties with strength of democratic or republican majority. More democratic majority? we're saying the associated parties are likely to be more left, or at least something...let's just look at that correlation, so that's one dimension we see if it separates anything.
# TODO second metric based on text or explicit ideology
# just make random political spectra and then adjust until people's first choice preferences match the house results.
# the ranked choice voting...:
# do this in a piece by piece aggregation
# each vote needs to be self-contained and have a single result communcated forward
# TODO pull and merge in the fivethirtyeight data
# TODO Use Report to put the writeup in README.md"
ft_house  = pd.read_csv("election-results/election_results_house.csv")
ft_pres   = pd.read_csv("election-results/election_results_presidential.csv")
ft_senate = pd.read_csv("election-results/election_results_senate.csv")

# House
fi = "csv/1976-2020-house.csv"
house = pd.read_csv(fi)# stringsAsFactors=False)
house = house.loc[house["state_po"] != "DC"]
# change ft_house names to match house names
ft_house.columns = id                           30920  year                      1976  
                   race_id                       9224  state                  ALABAMA
                   state_abbrev                    MO->state_po                    AL
                   state                     Missouri  state_fips                   1
                   office_id                      296  state_cen                   63
                   office_name             U.S. House  state_ic                    41
                   office_seat_name        District 4  office                US HOUSE
                   cycle                         2022  district                     1
                   stage                      general  stage                      GEN
                   special                      False  runoff                   False
                   party                          NaN  special                  False
                   politician_id                18975  candidate         JACK EDWARDS
                   candidate_id                 30914  party               REPUBLICAN
                   candidate_name      David A. Haave  writein                  False
                   ballot_party                     W  mode                     TOTAL
                   votes                            1  candidatevotes           98257
                   percent                0.000392035  totalvotes              157170
                   unopposed                    False  unofficial               False
                   winner                       False  version               20220331
                   alt_result_text                NaN  fusion_ticket            False
                   source          

# join ft_house_named into this
# do all the stuff we used to do
house["count"] = 1
# everyone who is running unopposed needs to get at least one vote
unopposed = house.groupby(["year", "state", "district"])["count", "candidatevotes"].sum().reset_index()
unopposed["unopposed"] = 0
unopposed.loc[(unopposed["count"] == 1) & (unopposed["candidatevotes"] < 1), "unopposed"] = 1
#unopposed["unopposed"] = 1
unopposed = unopposed[["year", "state", "district", "unopposed"]]

house = house.merge(unopposed, on=["year", "state", "district"], how="outer")
house.loc[house["unopposed"] > 0, "candidatevotes"] = 1
house = house.loc[house["candidatevotes"] != 0]
house = house.fillna("Blank")

# add state district field
house["state_district"] = house.agg('{0[state]}-{0[district]}'.format, axis=1)

house["count_win"] = 0
# simulate instant runoff?

house["cumsum"] = house.groupby(["year", "state_district"])["candidatevotes"].cumsum()
idx = house.groupby(["year", "state_district"], sort=False)["candidatevotes"].transform(max) == house["candidatevotes"]
house.loc[idx, "count_win"] = 1

# part to rerun
n_iter = 200
fnames = []
for i in range(n_iter):
  fname = F"dice_win-{i}"
  fnames.append(fname)
  house[fname] = 0
  print(F"running iteration {i+1}/{n_iter}")
  tv = house.groupby(["year", "state_district"])["totalvotes"].max().reset_index()
  tv[F"rand-{i}"] = np.random.rand(len(tv))
  tv[F"die-{i}"] = tv["totalvotes"]*tv[F"rand-{i}"]
  tv = tv.rename({'totalvotes': F'tv-{i}'}, axis=1)

  house = house.merge(tv, on=["year", "state_district"])
  house["win_tot"] = house["cumsum"]-house[F"die-{i}"]
  house.loc[house["win_tot"] < 0, "win_tot"] = HUGE
  #house["win_pos"] = house["win_tot"] > 0
  # note this has an issue if a candidate received zero votes, this will find both numbers n and n+0 and mark both as winners. Can't have that, so remove any candidates with 0 votes from house

  wdx = house.groupby(["year", "state_district"], sort=False)["win_tot"].transform(min) == house["win_tot"]
  house.loc[wdx, fname] = 1
  


# # part out by year
# # year = 2020
# for year in house["year"].unique():
#   y_house = house.loc[house["year"] == year]
#   # by state and district
#   test = torch.rand(100000).numpy()
#   print(F"Running stochastic election for {int(year)}")
#   # get subset that is unique by state and district
#   for sd in y_house["state_district"].unique():
#     y_sd_house = y_house.loc[y_house["state_district"] == sd]
#     # gotta deal with runoff
#     # take total, pick random, multiply by total, if within first range, candidate 1, second range ...
#     total = y_sd_house["candidatevotes"].sum()
#     test = y_sd_house["candidatevotes"].cumsum()
#     die = np.random.rand()*total
#     win = ((test-die) > 0).idxmax()
#     house.loc["dice_win",win] = 1

  
fo = "csv/1976-2020-house-stoch_test.csv"
house.to_csv(fo)

# aggregate by party and year

dnames = fnames
fnames.append("count_win")
house["party_major"] = house["party"]
house.loc[(house["party"] != "DEMOCRAT") & (house["party"] != "REPUBLICAN"), "party_major"] = "OTHER"

control = house.groupby(list(["year", "party_major"]))[fnames].sum().reset_index()
#control = control.loc[(control["dice_win"] > 0) | (control["count_win"] > 0)]

control_c = control.groupby(["year"])[fnames].sum().reset_index()
fo = "csv/1976-2020-house-stoch_test-control.csv"
control.to_csv(fo)
for year in control["year"].unique():
  fig = plt.figure()
  control_y = control.loc[control["year"] == year]
  da = np.array(control_y.loc[control_y["party_major"] == "DEMOCRAT", dnames])[0]
  ra = np.array(control_y.loc[control_y["party_major"] == "REPUBLICAN", dnames])[0]
  oa = np.array(control_y.loc[control_y["party_major"] == "OTHER", dnames])[0]
  bins = np.histogram(np.hstack((da, ra, oa)), bins=60)[1]
  dh = plt.hist(da, bins, color="blue", alpha=0.3, label="Democrat")
  rh = plt.hist(ra, bins, color="red", alpha=0.3, label="Republican")  
  oh = plt.hist(oa, bins, color="green", alpha=0.3, Label = "Other")
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
d20 = house.loc[house["year"] == 2020, "state_district"].unique()
d18 = house.loc[house["year"] == 2018, "state_district"].unique()
d16 = house.loc[house["year"] == 2016, "state_district"].unique()
house_1976 = house.loc[(house["year"] == 2018) & (house["state_district"] == "NEW YORK-2")]
house_dice = house_1976.loc[house_1976["dice_win-0"] > 0]
hdg = house_dice.groupby(["state_district"])["dice_win-0"].sum().reset_index()

# President
# Senate

