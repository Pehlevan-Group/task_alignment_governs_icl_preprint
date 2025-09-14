import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
kappa = float(input("kappa: "))
numavg = int(input("numavg: "))
experiment = input("experiment: ")
figurename = input("figurename: ")

rho = 0.01
train_power = 0.9
Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d

reads = []
for i in range(1,numavg+1):
    filepath_m = f'runs/{experiment}/test_pretrain_m_{i}.txt'
    with open(filepath_m, 'r') as f:
        contents = f.read().strip()
    if contents.endswith(','):
        contents = contents[:-1]
    contents = f'[{contents}]'
    reads.append(ast.literal_eval(contents))
test_equals_train_m = np.mean(np.array(reads),axis=0)
print('shape train', test_equals_train_m.shape)
test_equals_train_s = np.std(np.array(reads),axis=0)

# reads = []
# for i in range(1,numavg+1):
#     filepath_m = f'runs/{experiment}/test_largerpower_m_{i}.txt'
#     with open(filepath_m, 'r') as f:
#         contents = f.read().strip()
#     if contents.endswith(','):
#         contents = contents[:-1]
#     contents = f'[{contents}]'
#     reads.append(ast.literal_eval(contents))
# test_on_larger_power_m = np.mean(np.array(reads),axis=0)
# test_on_larger_power_s = np.std(np.array(reads),axis=0)

# reads = []
# for i in range(1,numavg+1):
#     filepath_m = f'runs/{experiment}/test_smallerpower_m_{i}.txt'
#     with open(filepath_m, 'r') as f:
#         contents = f.read().strip()
#     if contents.endswith(','):
#         contents = contents[:-1]
#     contents = f'[{contents}]'
#     reads.append(ast.literal_eval(contents))
# test_on_smaller_power_m = np.mean(np.array(reads),axis=0)
# test_on_smaller_power_s = np.std(np.array(reads),axis=0)

sns.set(style="white",font_scale=2,palette="rocket")
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams["figure.figsize"] = (13,10)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# fig, axes = plt.subplots(1, 4, constrained_layout=True) 
same_color = "#07C573"

alphas_TEST = np.logspace(np.log10(0.5),np.log10(100),20)
plt.scatter(alphas_TEST, test_equals_train_m, label = 'test on train')
# plt.scatter(alphas_TEST, test_on_smaller_power_m, label = 'test on smaller power')
# plt.scatter(alphas_TEST, test_on_larger_power_m, label = 'test on larger power')
plt.legend()
plt.xscale('log')
plt.xlabel('alpha test')
plt.ylabel('icl error')

plt.savefig(f'figs/{figurename}.png')
plt.clf()