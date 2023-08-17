# load csv files mdm_hiearchy3, mdm_hierarchy4, mdm_hiearchy5.csv
# and create a correlation matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load csv files
df3 = pd.read_csv('mdm_hierarchy3.csv')
df4 = pd.read_csv('mdm_hierarchy4.csv')
df5 = pd.read_csv('mdm_hierarchy5.csv')

# load reward csv file
df_reward = pd.read_csv('mdm_reward.csv')

# get data for each data
hu3 = [df3['MDM_1 - hierarchy_use/hierarchy_selected_3'], df3['MDM_1 - hierarchy_use/hierarchy_selected_3'], df3['MDM_1 - hierarchy_use/hierarchy_selected_3']]
hu4 = [df4['MDM_1 - hierarchy_use/hierarchy_selected_4'], df4['MDM_1 - hierarchy_use/hierarchy_selected_4'], df4['MDM_1 - hierarchy_use/hierarchy_selected_4']]
hu5 = [df5['MDM_1 - hierarchy_use/hierarchy_selected_5'], df5['MDM_1 - hierarchy_use/hierarchy_selected_5'], df5['MDM_1 - hierarchy_use/hierarchy_selected_5']]

# get reward data
rw = [df_reward['MDM_1 - training/episode/reward'], df_reward['MDM_2 - training/episode/reward'], df_reward['MDM_3 - training/episode/reward']]

# get index that df3['Step'] is closest to 50M
idx = (np.abs(df3['Step'] - 50000000)).idxmin()

hu3_c = np.array([hu3[0][:idx], hu3[1][:idx], hu3[2][:idx]]).flatten()
hu4_c = np.array([hu4[0][:idx], hu4[1][:idx], hu4[2][:idx]]).flatten()
hu5_c = np.array([hu5[0][:idx], hu5[1][:idx], hu5[2][:idx]]).flatten()
rw_c = np.array([rw[0][:idx], rw[1][:idx], rw[2][:idx]]).flatten()

# cut everthing in hu3 hu4 hu5 rw before idx
hu3_ = np.stack([hu3[0][:idx], hu3[1][:idx], hu3[2][:idx]], axis=1)
hu4_ = np.stack([hu4[0][:idx], hu4[1][:idx], hu4[2][:idx]], axis=1)
hu5_ = np.stack([hu5[0][:idx], hu5[1][:idx], hu5[2][:idx]], axis=1)
rw_ = np.stack([rw[0][:idx], rw[1][:idx], rw[2][:idx]], axis=1)

# correlation between hu3_c and rw_c, hu4_c and rw_c, hu5_c and rw_c
# use scipy to find correlation
from scipy.stats import pearsonr, spearmanr
# first find index that both hu3_c and rw_c are not nan
idx1 = np.argwhere(~np.isnan(hu3_c)).squeeze()
idx2 = np.argwhere(~np.isnan(rw_c)).squeeze()
idx = np.intersect1d(idx1, idx2)
corr1 = pearsonr(hu3_c[idx], rw_c[idx])
corr3_s = spearmanr(hu3_c[idx], rw_c[idx])

# find index that both hu4_c and rw_c are not nan
idx1 = np.argwhere(~np.isnan(hu4_c)).squeeze()
idx2 = np.argwhere(~np.isnan(rw_c)).squeeze()
idx = np.intersect1d(idx1, idx2)
corr2 = pearsonr(hu4_c[idx], rw_c[idx])
corr4_s = spearmanr(hu4_c[idx], rw_c[idx])

# find index that both hu5_c and rw_c are not nan
idx1 = np.argwhere(~np.isnan(hu5_c)).squeeze()
idx2 = np.argwhere(~np.isnan(rw_c)).squeeze()
idx = np.intersect1d(idx1, idx2)
corr3 = pearsonr(hu5_c[idx], rw_c[idx])
corr5_s = spearmanr(hu5_c[idx], rw_c[idx])


# three hexbin plots for hu3_c and rw_c, hu4_c and rw_c, hu5_c and rw_c
# first make figure subplots
fig, ax = plt.subplots(1, 3, figsize=(30, 10))

# set font size to 20
plt.rcParams.update({'font.size': 20})

# use seaborn to plot
# first find index that both hu3_c and rw_c are not nan
idx1 = np.argwhere(~np.isnan(hu3_c)).squeeze()
idx2 = np.argwhere(~np.isnan(rw_c)).squeeze()
idx = np.intersect1d(idx1, idx2)
# plot
sns.regplot(x=hu3_c[idx], y=rw_c[idx], ax=ax[0], scatter_kws={'alpha':0.1}, color='b')
ax[0].set_xlabel('Proportion of using Hierarchy 3', fontsize=20)
ax[0].set_ylabel('Reward')

# find index that both hu4_c and rw_c are not nan
idx1 = np.argwhere(~np.isnan(hu4_c)).squeeze()
idx2 = np.argwhere(~np.isnan(rw_c)).squeeze()
idx = np.intersect1d(idx1, idx2)
# plot
sns.regplot(x=hu4_c[idx], y=rw_c[idx], ax=ax[1], scatter_kws={'alpha':0.1}, color='g')
ax[1].set_xlabel('Proportion of using Hierarchy 4')
ax[1].set_ylabel('Reward')

# find index that both hu5_c and rw_c are not nan
idx1 = np.argwhere(~np.isnan(hu5_c)).squeeze()
idx2 = np.argwhere(~np.isnan(rw_c)).squeeze()
idx = np.intersect1d(idx1, idx2)
# plot
sns.regplot(x=hu5_c[idx], y=rw_c[idx], ax=ax[2], scatter_kws={'alpha':0.1}, color='r')
ax[2].set_xlabel('Proportion of using Hierarchy 5')
ax[2].set_ylabel('Reward')

# set ylim from 0 to 250
ax[0].set_ylim([0, 250])
ax[1].set_ylim([0, 250])
ax[2].set_ylim([0, 250])

# save figure
plt.savefig('hierarchy_use_reward.png', dpi=300, bbox_inches='tight')
