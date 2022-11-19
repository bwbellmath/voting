import pandas as pd
import numpy as np
import torch 
import matplotlib.pyplot as plt

HUGE = 10000000000
fi = "csv/1976-2020-senate.csv"
senate = pd.read_csv(fi)# stringsAsFactors=False)
senate = senate.loc[senate["state_po"] != "DC"]
senate["count"] = 1
# everyone who is running unopposed needs to get at least one vote
unopposed = senate.groupby(["year", "state", "district", "class"])["count", "candidatevotes"].sum().reset_index()
unopposed["unopposed"] = 0
unopposed.loc[(unopposed["count"] == 1) & (unopposed["candidatevotes"] < 1), "unopposed"] = 1
#unopposed["unopposed"] = 1
unopposed = unopposed[["year", "class", "state", "district", "unopposed"]]

senate = senate.merge(unopposed, on=["year", "state", "district", "class"], how="outer")
senate.loc[senate["unopposed"] > 0, "candidatevotes"] = 1
senate = senate.loc[senate["candidatevotes"] != 0]
senate = senate.fillna("Blank")
senate["class"] = senate["class"].astype("int")

# add state district field
senate["state_district"] = senate.agg('{0[state]}-{0[district]}'.format, axis=1)

senate["count_win"] = 0
# compute class
senate["year_class"] = ((((senate["year"]-1976)/2) % 3) + 1).astype(int)
senate["actually_special"] = senate["special"]
senate.loc[senate["class"] == senate["year_class"], "actually_special"] = False
# simulate instant runoff?

senate["cumsum"] = senate.groupby(["year", "state_district", "class"])["candidatevotes"].cumsum()
idx = senate.groupby(["year", "state_district", "class"], sort=False)["candidatevotes"].transform(max) == senate["candidatevotes"]
senate.loc[idx, "count_win"] = 1

# part to rerun
n_iter = 200
fnames = []
for i in range(n_iter):
  fname = F"dice_win-{i}"
  fnames.append(fname)
  senate[fname] = 0
  print(F"running iteration {i+1}/{n_iter}")
  tv = senate.groupby(["year", "state_district", "class"])["totalvotes"].max().reset_index()
  tv[F"rand-{i}"] = np.random.rand(len(tv))
  tv[F"die-{i}"] = tv["totalvotes"]*tv[F"rand-{i}"]
  tv = tv.rename({'totalvotes': F'tv-{i}'}, axis=1)

  senate = senate.merge(tv, on=["year", "state_district", "class"])
  senate["win_tot"] = senate["cumsum"]-senate[F"die-{i}"]
  senate.loc[senate["win_tot"] < 0, "win_tot"] = HUGE
  #senate["win_pos"] = senate["win_tot"] > 0
  # note this has an issue if a candidate received zero votes, this will find both numbers n and n+0 and mark both as winners. Can't have that, so remove any candidates with 0 votes from senate

  wdx = senate.groupby(["year", "state_district", "class"], sort=False)["win_tot"].transform(min) == senate["win_tot"]
  senate.loc[wdx, fname] = 1
  
# senate_2 = senate.copy()
# senate_2["year"] += 2
# senate_carry = senate.merge(senate_2, on=["year", "state", "district"

# # part out by year
# # year = 2020
# for year in senate["year"].unique():
#   y_senate = senate.loc[senate["year"] == year]
#   # by state and district
#   test = torch.rand(100000).numpy()
#   print(F"Running stochastic election for {int(year)}")
#   # get subset that is unique by state and district
#   for sd in y_senate["state_district"].unique():
#     y_sd_senate = y_senate.loc[y_senate["state_district"] == sd]
#     # gotta deal with runoff
#     # take total, pick random, multiply by total, if within first range, candidate 1, second range ...
#     total = y_sd_senate["candidatevotes"].sum()
#     test = y_sd_senate["candidatevotes"].cumsum()
#     die = np.random.rand()*total
#     win = ((test-die) > 0).idxmax()
#     senate.loc["dice_win",win] = 1

  
fo = "csv/1976-2020-senate-stoch_test.csv"
senate.to_csv(fo)

# aggregate by party and year

dnames = fnames
fnames.append("count_win")
#senate["party_major"] = senate["party_detailed"]
#senate.loc[(senate["party_detailed"] != "DEMOCRAT") & (senate["party_detailed"] != "REPUBLICAN"), "party_major"] = "OTHER"

senate_main = senate.loc[senate["actually_special"] == False].copy()
senate_main["election_cur"] = 0
senate_main["election_year"] = senate_main["year"].copy()
senate_main_2 = senate_main.copy()
senate_main_4 = senate_main.copy()
senate_main_2["year"] += 2
senate_main_4["year"] += 4
senate_main_2["election_cur"] = 2
senate_main_4["election_cur"] = 4
senate_main_cur = pd.concat([senate_main, senate_main_2, senate_main_4]).reset_index()

senate_special = senate.loc[senate["actually_special"] == True].copy()

# find nearest class year to each special election

senate_special["class_diff"] = ((senate_special["class"]-1) - (senate_special["year_class"] - 1)) % 3
senate_special["class_diff_backward"] = ((senate_special["year_class"]-1) - (senate_special["class"] - 1)) % 3


senate_special["original_year"] = (senate_special["year"]-senate_special["year"]%2)	-2*senate_special["class_diff_backward"] 

# senate["year_class"] = ((((senate["year"]-1976)/2) % 3) + 1).astype(int)
# senate_special.loc[:, "class_year"] = (((senate_special["year"]-1976)/2) % 3) + 1
# for each senate special election (year, state, class)

senate_special_cur = senate_special
for ii in senate_special["class_diff"].unique():
  subset = senate_special.loc[senate_special["class_diff"] == ii].copy()
  for jj in range(ii-1):
    subset["year"] += (jj+1)*2
    senate_special_cur = pd.concat([senate_special_cur, subset])
# find and kill overlapping original election rows
senate_special_cur = senate_special_cur.reset_index()
senate_special_cur["original_year"] = (senate_special_cur["year"]-senate_special_cur["year"]%2)	-2*senate_special_cur["class_diff_backward"] 



senate_special_list = senate_special[["year", "original_year", "class", "year_class" , "actually_special", "state"]].drop_duplicates()
count = 0
for row in senate_special_list.iterrows():
  # kill regular election that this overwrites
  # match on : year, original_year, state, class, and must be "special" == False
  idx = senate_main_cur.loc[(senate_main_cur["year"] >= row[1]["year"]) &
                            (senate_main_cur["election_year"] == row[1]["original_year"]) &
                            (senate_main_cur["state"] == row[1]["state"]) &
                            (senate_main_cur["class"] == row[1]["class"])
                            ].index# &
                              
  count += len(idx)
  senate_main_cur.drop(idx, inplace=True)
print(F"Deleted {count} carried forward regular election results superceded by special elections.")
senate_cur =  pd.concat([senate_special_cur, senate_main_cur])

# get class from adjusted years
# mark most recent election (before merging in extras)
# special class versus this year's class: 0, 1, 2.
# 	0 : take the most recent election
#	1 : take the special election
# 	2 : take the special election? 
# only want to carry forward 0 or 1 cycle -- if the special class - this year's class 
# control2 = control.copy()
# control2["year"] += 2
# control4 = control.copy()
# control4["year"] += 4
# control_cur = pd.concat([control, control2, control4])
# do this for the regular elections
# specially elected senators sever the rest of the usual term. 
# Now for each special election (unique year and class among the special elections) look up the old elections under this year and replace it

# now take unique by state, year, and class, but if there are duplicates, keep the one with "special"

# every year we have a cohort of classes 1, 2, and 3
# if this year is class 1, then we need all the winners this year plus all the class 2s and 3s from last year
# If there were any special elections for class 2 or 3 then we want those elections to replace these elections 

#control_class = control_cur.groupby("year", 
senate_cur["eff_year"] = senate_cur["year"]+(senate_cur["year"]%2)

fo = "csv/1976-2020-senate_cur.csv"
senate_cur.to_csv(fo)

control = senate_cur.groupby(list(["eff_year", "party_simplified"]))[fnames].sum().reset_index()
control_tot = senate_cur.groupby(["eff_year"])[fnames].sum().reset_index()
control_state = senate_cur.groupby(["eff_year", "state"])[fnames].sum().reset_index()	
#control = control.loc[(control["dice_win"] > 0) | (control["count_win"] > 0)]

#control_c = control_tot.groupby(["year"])[fnames].sum().reset_index()
fo = "csv/1976-2020-senate-stoch_test-control.csv"
control.to_csv(fo)
# keep count total for each party
ada = np.zeros(len(control["eff_year"].unique()))
ara = np.zeros(len(control["eff_year"].unique()))
aoa = np.zeros(len(control["eff_year"].unique()))
ala = np.zeros(len(control["eff_year"].unique()))
# keep dice mean for each party
adm = np.zeros(len(control["eff_year"].unique()))
arm = np.zeros(len(control["eff_year"].unique()))
aom = np.zeros(len(control["eff_year"].unique()))
alm = np.zeros(len(control["eff_year"].unique()))
count = 0
for year in control["eff_year"].unique():
  fig = plt.figure()
  control_y = control.loc[control["eff_year"] == year]
  #control_yy = control_cur.loc[control_cur["year"] == year]
  da = np.array(control_y.loc[control_y["party_simplified"] == "DEMOCRAT", dnames])[0]
  ra = np.array(control_y.loc[control_y["party_simplified"] == "REPUBLICAN", dnames])[0]
  oinds = control_y.loc[control_y["party_simplified"] == "OTHER", dnames].index
  if (len(oinds) > 0):
    oa = np.array(control_y.loc[oinds, dnames])[0]
    oc = np.array(control_y.loc[oinds, "count_win"])[0]
  else:
    oa = np.zeros(len(da))
    oc = np.array([0])
  linds = control_y.loc[control_y["party_simplified"] == "LIBERTARIAN", dnames].index
  if (len(linds) > 0):
    la = np.array(control_y.loc[linds, dnames])[0]
    lc = np.array(control_y.loc[linds, "count_win"])[0]
  else:
    la = np.zeros(len(da))
    lc = np.array([0])


  #oa = np.array()[0]
  #la = np.array(control_y.loc[control_y["party_simplified"] == "LIBERTARIAN", dnames])[0]
  bins = np.histogram(np.hstack((da, ra, oa, la)), bins=60)[1]
  dh = plt.hist(da, bins, color="blue", alpha=0.3, label="Democrat")
  rh = plt.hist(ra, bins, color="red", alpha=0.3, label="Republican")  
  oh = plt.hist(oa, bins, color="green", alpha=0.3, label = "Other")
  lh = plt.hist(la, bins, color="gray", alpha=0.3, label = "Libertarian")
  ymax = np.array([dh[0], rh[0], oh[0]]).max()
  dc = np.array(control_y.loc[control_y["party_simplified"] == "DEMOCRAT", "count_win"])[0]
  rc = np.array(control_y.loc[control_y["party_simplified"] == "REPUBLICAN", "count_win"])[0]
  ada[count] = dc
  ara[count] = rc
  aoa[count] = oc
  ala[count] = lc
  # keep dice mean for each party
  adm[count] = da.mean()
  arm[count] = ra.mean()
  aom[count] = oa.mean()
  alm[count] = la.mean()

  #lc = np.array(control_y.loc[control_y["party_simplified"] == "LIBERTARIAN", "count_win"])[0]
  plt.vlines(dc, 0, ymax, color="blue", label=F"By Count: {dc}")
  plt.vlines(rc, 0, ymax, color="red", label=F"By Count: {rc}")
  plt.vlines(oc, 0, ymax, color="green", label=F"By Count: {oc}")
  plt.vlines(lc, 0, ymax, color="gray", label=F"By Count: {lc}")

  plt.vlines(da.mean(), 0, ymax, color="blue", label=F"Dice Mean: {da.mean()}", linestyles="dashed")
  plt.vlines(ra.mean(), 0, ymax, color="red", label=F"Dice Mean: {ra.mean()}", linestyles="dashed")
  plt.vlines(oa.mean(), 0, ymax, color="green", label=F"Dice Mean: {oa.mean()}", linestyles="dashed")
  plt.vlines(la.mean(), 0, ymax, color="gray", label=F"Dice Mean: {la.mean()}", linestyles="dashed")


  plt.legend()
  plt.title(F"Histogram of Stochastic voting outcomes for Senate Election in {year}")
  plt.xlabel("Number of Senate Seats Awarded to each Party")
  plt.ylabel("Count of Scenarios for Each Balance of Power")
  fo = F"img/senate-hist-{year}.png"
  fig.set_size_inches(14,7)
  fig.savefig(fo)
  count += 1

# big aggregate plot
agg = plt.figure()

years = control["eff_year"].unique().copy()
years.sort()
years = years[0:-3]
ada = ada[0:-3]
ara = ara[0:-3]
aoa = aoa[0:-3]
ala = ala[0:-3]
adm = adm[0:-3]
arm = arm[0:-3]
aom = aom[0:-3]
alm = alm[0:-3]

plt.fill_between(years, 0, ada, color="blue", label="Democrat", alpha=0.3)
plt.fill_between(years, ada, ada+aoa, color="green", label="Other", alpha=0.3)
plt.fill_between(years, ada+aoa, ada+aoa+ala, color="gray", label="Libertarial", alpha=0.3)
plt.fill_between(years, 100, 100-ara, color="red", label="Republican", alpha=0.3)

plt.plot(years, adm, color="blue", label="Democrat Dice Avg")
plt.plot(years, arm, color="red", label="Republican Dice Avg")
plt.plot(years, aom, color="green", label="Other Dice Avg")
plt.plot(years, alm, color="gray", label="Libertarian Dice Avg")

plt.hlines(50, years.min(), years.max(), color="black", label="majority")

# group other parties in control all together


# check unique state-districts in 2016 versus 2018
# d20 = senate.loc[senate["eff_year"] == 2020, "state_district"].unique()
# d18 = senate.loc[senate["eff_year"] == 2018, "state_district"].unique()
# d16 = senate.loc[senate["eff_year"] == 2016, "state_district"].unique()
# senate_1976 = senate.loc[(senate["eff_year"] == 2018) & (senate["state_district"] == "NEW YORK-2")]
# senate_dice = senate_1976.loc[senate_1976["dice_win-0"] > 0]
# hdg = senate_dice.groupby(["state_district"])["dice_win-0"].sum().reset_index()
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
