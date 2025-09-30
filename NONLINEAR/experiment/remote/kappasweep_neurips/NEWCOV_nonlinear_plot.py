import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
numavg = int(input("numavg: "))
experiment = input("experiment: ")
figurename = input("figurename: ")

kappas = [0.2, 0.5, 1, 2, 10]
rho = 0.01
train_power = 0.9
Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d

reads = []
for i in range(numavg):
    filepath_m = f'NEWCOVARIANCERUNS/{experiment}/test_equals_train_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    data = ast.literal_eval(contents)
    data = sorted(data, key=lambda x: x[0])
    reads.append([item[1] for item in data])
test_equals_train_m = np.mean(np.array(reads),axis=0)
print('shape train', test_equals_train_m.shape)
test_equals_train_s = np.std(np.array(reads),axis=0)
# This will just be a length kappa 1-d array

reads = []
for i in range(numavg):
    filepath_m = f'NEWCOVARIANCERUNS/{experiment}/test_powers_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    data = ast.literal_eval(contents)
    data = sorted(data, key=lambda x: x[0])
    reads.append([item[1] for item in data])
test_on_powers_m = np.mean(np.array(reads),axis=0)
test_on_powers_s = np.std(np.array(reads),axis=0)
print('shape powers', test_on_powers_m.shape)
# This will be 2d with shape num(kappas) x num(powers)

reads = []
for i in range(numavg):
    filepath_m = f'NEWCOVARIANCERUNS/{experiment}/test_exps_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    data = ast.literal_eval(contents)
    data = sorted(data, key=lambda x: x[0])
    reads.append([item[1] for item in data])
test_on_exps_m = np.mean(np.array(reads),axis=0)
test_on_exps_s = np.std(np.array(reads),axis=0)
print('shape exps', test_on_exps_m.shape)
# This will be 2d with shape num(kappas) x num(spikes)

reads = []
for i in range(numavg):
    filepath_m = f'NEWCOVARIANCERUNS/{experiment}/test_ranks_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    data = ast.literal_eval(contents)
    data = sorted(data, key=lambda x: x[0])
    reads.append([item[1] for item in data])
test_on_ranks_m = np.mean(np.array(reads),axis=0)
test_on_ranks_s = np.std(np.array(reads),axis=0)
print('shape ranks', test_on_ranks_m.shape)
# This will be 2d with shape num(kappas) x num(spikes)

test_powers = np.linspace(train_power - 0.5, train_power + 0.5, 11)
expowers = np.linspace(train_power - 0.5, train_power + 0.5, 11)
rankfs = [0.2,0.4,0.6,0.8,1]

experiment_d = d

sns.set(style="white",font_scale=4,palette="mako")
plt.rcParams['lines.linewidth'] = 7
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, axes = plt.subplots(1, 4, figsize=(46,10), sharey=True, constrained_layout=True)
same_color = 'red' #"#07C573"

keys = ['mary', 'F', 'trace', 'cka']
for plotting_index, key in enumerate(keys):
    kappa_average_spearman = []
    for i, kappa in enumerate(kappas):
        if key == 'mary':
            alignment_match = ICL_alignment(Ctr, Ctr, tau, alpha, kappa, rho, numavg=100)
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append(ICL_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
            alignment_exps = []
            for expower in expowers:
                Ctest = np.diag(np.array([np.exp(-expower*(j+1)) for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_exps.append(ICL_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
            alignment_ranks = []
            for f in rankfs:
                Ctest = np.diag(complexity_class_covariance(d, int(d*f), True))
                alignment_ranks.append(ICL_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))

        elif key == 'trace':
            alignment_match = (1/d)*np.trace(np.linalg.inv(Ctr)@Ctr)
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append((1/d)*np.trace(np.linalg.inv(Ctr)@Ctest))
            alignment_exps = []
            for expower in expowers:
                Ctest = np.diag(np.array([np.exp(-expower*(j+1)) for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_exps.append((1/d)*np.trace(np.linalg.inv(Ctr)@Ctest))
            alignment_ranks = []
            for f in rankfs:
                Ctest = np.diag(complexity_class_covariance(d, int(d*f), True))
                alignment_ranks.append((1/d)*np.trace(np.linalg.inv(Ctr)@Ctest))

        elif key == 'F':
            alignment_match = resolvent_alignment(Ctr, Ctr, tau, alpha, kappa, rho, numavg=100)
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append(resolvent_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
            alignment_exps = []
            for expower in expowers:
                Ctest = np.diag(np.array([np.exp(-expower*(j+1)) for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_exps.append(resolvent_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))
            alignment_ranks = []
            for f in rankfs:
                Ctest = np.diag(complexity_class_covariance(d, int(d*f), True))
                alignment_ranks.append(resolvent_alignment(Ctr, Ctest, tau, alpha, kappa, rho, numavg=100))

        elif key == 'cka':
            alignment_match = (cka(d, Ctr, Ctr)/np.sqrt(cka(d, Ctr, Ctr)*cka(d, Ctr, Ctr)))**(-1)
            alignment_powers = []
            for test_power in test_powers:
                Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_powers.append((cka(d,Ctr,Ctest)/np.sqrt(cka(d,Ctr,Ctr)*cka(d,Ctest,Ctest)))**(-1))
            alignment_exps = []
            for expower in expowers:
                Ctest = np.diag(np.array([np.exp(-expower*(j+1)) for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
                alignment_exps.append((cka(d,Ctr,Ctest)/np.sqrt(cka(d,Ctr,Ctr)*cka(d,Ctest,Ctest)))**(-1))
            alignment_ranks = []
            for f in rankfs:
                Ctest = np.diag(complexity_class_covariance(d, int(d*f), True))
                alignment_ranks.append((cka(d,Ctr,Ctest)/np.sqrt(cka(d,Ctr,Ctr)*cka(d,Ctest,Ctest)))**(-1))

        if i == 0:
            axes[plotting_index].scatter(alignment_powers, test_on_powers_m[i,:], marker='o', s=280, color='grey', label = 'Test on powerlaw')
            axes[plotting_index].scatter(alignment_ranks, test_on_ranks_m[i,:], marker='^', s=300, color='grey', label = 'Test on low ranks')
            # axes[plotting_index].scatter(alignment_exps, test_on_exps_m[i,:], marker='P', s=150, color='grey', label = 'Test on exponential decay')
            axes[plotting_index].scatter(alignment_match, test_equals_train_m[i], marker='*', s=600, color=same_color, label = 'Test on pretrain')

        concatenated_x = alignment_ranks + alignment_powers + [alignment_match]
        concatenated_y = list(test_on_ranks_m[i,:]) +  list(test_on_powers_m[i,:]) + [test_equals_train_m[i]]
        zipped = list(zip(concatenated_x, concatenated_y))
        sorted_pairs = sorted(zipped, key=lambda pair: pair[0])
        sorted_X, sorted_Y = zip(*sorted_pairs)
        #axes[plotting_index].plot(sorted_X,sorted_Y,color = color_cycle[i+1], alpha = 0.8, label =fr"$\kappa = $ {kappa}",zorder=1)
        kappa_average_spearman.append(scipy.stats.spearmanr(sorted_X, sorted_Y).statistic) 

        axes[plotting_index].plot(alignment_ranks, test_on_ranks_m[i,:], ':', color=color_cycle[i+1], alpha = 0.8, zorder=1)
        axes[plotting_index].plot(alignment_powers, test_on_powers_m[i,:], '-', color=color_cycle[i+1], alpha = 0.8, label =fr"$\kappa = $ {kappa}", zorder=1)
        axes[plotting_index].scatter(alignment_ranks, test_on_ranks_m[i,:], marker='^', s=300, color=color_cycle[i+1], zorder = i+2)
        axes[plotting_index].scatter(alignment_powers, test_on_powers_m[i,:], marker='o', s=280, color=color_cycle[i+1], zorder = i+2)
        axes[plotting_index].scatter(alignment_match, test_equals_train_m[i], marker='*', s=600, color=same_color, zorder = i+10)

    kappa_average_spearman = np.array(kappa_average_spearman)
    axes[plotting_index].spines['top'].set_color('lightgray')
    axes[plotting_index].spines['right'].set_color('lightgray')
    axes[plotting_index].spines['bottom'].set_color('lightgray')
    axes[plotting_index].spines['left'].set_color('lightgray')
    if key == 'mary':
        axes[plotting_index].set_xlabel(fr"$e_{{\mathrm{{misalign}}}} = \langle C_{{\mathrm{{test}}}} \mathcal{{K}} \rangle$")
    if key == 'trace':
        axes[plotting_index].set_xlabel(fr"$\langle C_{{\mathrm{{test}}}} C_{{\mathrm{{train}}}}^{{-1}} \rangle$")
    if key == 'F':
        axes[plotting_index].set_xlabel(fr"$\langle C_{{\mathrm{{test}}}} F_\kappa(\sigma) \rangle$")
    if key == 'cka':
        axes[plotting_index].set_xlabel(fr"$\mathrm{{CKA}}(C_{{\mathrm{{tr}}}}, C_{{\mathrm{{test}}}})^{{-1}}$")
    if plotting_index==0:
        axes[plotting_index].set_ylabel('Transformer ICL error')
    axes[plotting_index].tick_params(axis='both', which='major', labelsize=30)
    print('key is ', key, 'and spearman is ', np.mean(kappa_average_spearman))
    
# # Get handles and labels from first subplot (or any subplot)
handles, labels = axes[0].get_legend_handles_labels()

# # Put legend to the right, vertically centered
# fig.legend(handles, labels,
#            loc='center left',
#            bbox_to_anchor=(1, 0.5),  # push slightly further right
#            frameon=False)

# Add a single legend above all subplots
fig.legend(handles, labels,
           loc='upper center',        # place legend at top center
           bbox_to_anchor=(0.5, 1.12),# adjust vertical position
           ncol=len(labels),          # put entries in a single row
           frameon=False)             # optional: no box

# Adjust layout and save
plt.savefig(f'final_alignment_plots/{figurename}.pdf', bbox_inches='tight')
plt.clf()