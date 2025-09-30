import ast
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
experiment = input("experiment: ")
figurename = input("figurename: ")

kappas = np.linspace(0.1,2.1,21)
signals = np.int64(np.linspace(0,d-1,d//2))
rho = 0.01
Ctr = np.diag(([i for i in range(1,d+1)])[::-1]); Ctr = (d/np.trace(Ctr))*Ctr

filepath_m = f'runs/{experiment}/simulation_m.txt'
with open(filepath_m, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
data = ast.literal_eval(contents)
data = sorted(data, key=lambda x: x[0])

filepath_s = f'runs/{experiment}/simulation_s.txt'
with open(filepath_s, 'r') as f:
    contents = f.read().strip()
if contents.endswith(','):
    contents = contents[:-1]
contents = f'[{contents}]'
stds = ast.literal_eval(contents)
stds = sorted(stds, key=lambda x: x[0])

sns.set(style="white",font_scale=2.7,palette="mako")
plt.rcParams['lines.linewidth'] = 5.5
fig, axes = plt.subplots(1, 2, figsize=(24,10), sharey=True, constrained_layout=True)

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plot_inds = [0, 39, 49, 54, 59] #np.int64(np.linspace(0,len(signals)-1,6))

kappas_theory = np.linspace(0.05,2.15,43)
pretrain_theory = [ICL_pretraining(Ctr, tau, alpha, kappa, rho, numavg=10) for kappa in kappas_theory]
pretrain_experiment = [ICL_pretraining(Ctr, tau, alpha, kappa, rho, numavg=10) for kappa in kappas]

matched = [ICL_error(Ctr, Ctr, tau, alpha, kappa, rho, 100) for kappa in kappas_theory]
axes[0].plot(kappas_theory, matched, color='red')
axes[0].scatter(kappas, [item[1][-1] for item in data],  color='red', edgecolors='red',s=200,linewidth=2.5,zorder=1,label = fr"$C_{{\mathrm{{test}}}} = C_{{\mathrm{{train}}}}$")
axes[0].scatter(kappas, [item[1][-1] for item in data],  color='#FAFAFA', edgecolors='red',s=200,linewidth=2.5,zorder=10)
axes[0].fill_between(kappas, np.array([item[1][-1] for item in data]) - np.array([item[1][-1] for item in stds]), np.array([item[1][-1] for item in data]) + np.array([item[1][-1] for item in stds]), color='red', alpha = 0.2)

computed = []
for i, plot_ind in enumerate(plot_inds):
    myerrorlist = [ICL_error(Ctr, np.diag(spikevalue(d,0,signals[plot_ind])), tau, alpha, kappa, rho, 100) for kappa in kappas_theory]
    computed.append(myerrorlist)
    axes[0].plot(kappas_theory, myerrorlist, color=color_cycle[i+1])
    axes[0].scatter(kappas, [item[1][plot_ind] for item in data], color=color_cycle[i+1], edgecolors=color_cycle[i+1],s=200,linewidth=2.5,zorder=1,label = f'idx {signals[plot_ind]+1}/{d}')
    axes[0].scatter(kappas, [item[1][plot_ind] for item in data], color='#FAFAFA', edgecolors=color_cycle[i+1],s=200,linewidth=2.5,zorder=10)
    axes[0].fill_between(kappas, np.array([item[1][plot_ind] for item in data]) - np.array([item[1][plot_ind] for item in stds]), np.array([item[1][plot_ind] for item in data]) + np.array([item[1][plot_ind] for item in stds]), color=color_cycle[i+1], alpha = 0.2)

axes[1].plot(kappas_theory, np.array(matched) - np.array(pretrain_theory), color='red')
axes[1].scatter(kappas, np.array([item[1][-1] for item in data]) - np.array(pretrain_experiment),  color='red', edgecolors='red',s=200,linewidth=2.5,zorder=1,label = fr"$C_{{\mathrm{{test}}}} = C_{{\mathrm{{train}}}}$")
axes[1].scatter(kappas, np.array([item[1][-1] for item in data]) - np.array(pretrain_experiment),  color='#FAFAFA', edgecolors='red',s=200,linewidth=2.5,zorder=10)
axes[1].fill_between(kappas, np.array([item[1][-1] for item in data]) - np.array(pretrain_experiment) - np.array([item[1][-1] for item in stds]), np.array([item[1][-1] for item in data]) - np.array(pretrain_experiment)+ np.array([item[1][-1] for item in stds]), color='red', alpha = 0.2)

for i, plot_ind in enumerate(plot_inds):
    axes[1].plot(kappas_theory, np.array(computed[i]) - np.array(pretrain_theory), color=color_cycle[i+1])
    axes[1].scatter(kappas, np.array([item[1][plot_ind] for item in data]) - np.array(pretrain_experiment), color=color_cycle[i+1], edgecolors=color_cycle[i+1],s=200,linewidth=2.5,zorder=1,label =fr"idx {signals[plot_ind]+1}/{d}")
    axes[1].scatter(kappas, np.array([item[1][plot_ind] for item in data]) - np.array(pretrain_experiment), color='#FAFAFA', edgecolors=color_cycle[i+1],s=200,linewidth=2.5,zorder=10)
    axes[1].fill_between(kappas, np.array([item[1][plot_ind] for item in data]) - np.array(pretrain_experiment) - np.array([item[1][plot_ind] for item in stds]), np.array([item[1][plot_ind] for item in data]) - np.array(pretrain_experiment) + np.array([item[1][plot_ind] for item in stds]), color=color_cycle[i+1], alpha = 0.2)

# Get handles and labels from the *current axes*
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=6,frameon=False)

# leg.get_frame().set_alpha(0)
axes[0].spines['top'].set_color('lightgray')
axes[0].spines['right'].set_color('lightgray')
axes[0].spines['bottom'].set_color('lightgray')
axes[0].spines['left'].set_color('lightgray')
axes[0].set_xlabel(r'$\kappa$ = k/d')
axes[0].set_ylabel(fr"$e_{{\mathrm{{ICL}}}}(C_{{\mathrm{{train}}}},C_{{\mathrm{{test}}}})$")
axes[0].tick_params(axis='both', which='major',labelsize=22)

axes[1].spines['top'].set_color('lightgray')
axes[1].spines['right'].set_color('lightgray')
axes[1].spines['bottom'].set_color('lightgray')
axes[1].spines['left'].set_color('lightgray')
axes[1].set_xlabel(r'$\kappa$ = k/d')
axes[1].set_ylabel(fr"$e_{{\mathrm{{misalign}}}}(C_{{\mathrm{{train}}}},C_{{\mathrm{{test}}}})$")
axes[1].tick_params(axis='both', which='major',labelsize=22)

plt.savefig(f'figs/{figurename}.pdf', bbox_inches='tight')
plt.clf()
