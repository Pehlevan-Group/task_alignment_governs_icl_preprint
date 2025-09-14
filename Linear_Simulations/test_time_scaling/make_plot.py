import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from theory import *

d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
experiment = input("experiment: ")
figurename = input("figurename: ")
    
kappa = 1
alpha_TESTS = np.logspace(np.log10(0.1), np.log10(1000), 21)
rho = 0.01

fulldata_m = []
fulldata_s = []
for option in range(5):
    print(f"option {option}")
    filepath_m = f'runs/{experiment}/simulation_{option}.txt'
    with open(filepath_m, 'r') as file:
        content = file.read().rstrip(',')  # Remove trailing comma
        data = ast.literal_eval(f'[{content}]')  # Safely parse to list of [val, vector]
    vectors = np.array([entry[1] for entry in data])
    fulldata_m.append(list(np.mean(vectors, axis=0)))
    fulldata_s.append(list(np.std(vectors, axis=0)))


Ctr0 = np.eye(d)
# filename = f'runs/{experiment}/matrix.txt'
# with open(filename, 'r') as file:
#     content = file.read().rstrip(',')  # remove whitespace and trailing comma
# Ctr1 = np.array(ast.literal_eval(content))
Ctr2 = np.diag(np.array([(j + 1) ** (-float(0.5)) for j in range(d)])); Ctr2 = (Ctr2/np.trace(Ctr2))*d
Ctr3 = np.diag(np.array([(j + 1) ** (-float(1.5)) for j in range(d)])); Ctr3 = (Ctr3/np.trace(Ctr3))*d
Ctr4 = np.diag(np.array([(j + 1) ** (-float(1.0)) for j in range(d)])); Ctr4 = (Ctr4/np.trace(Ctr4))*d

sns.set(style="white",font_scale=2.5,palette="mako")
plt.rcParams['lines.linewidth'] = 4
plt.rcParams["figure.figsize"] = (14,10)

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.axvline(alpha,linestyle=':',linewidth=3.5,color='grey',label=r'Training $\alpha$')

start = 0
plt.plot(alpha_TESTS[start:], np.array([ICL_error(Ctr0, Ctr0, tau, alpha, kappa, rho, 100, alpha_test) for alpha_test in alpha_TESTS[start:]]), color=color_cycle[2], label = r'$C_\text{train}$ isotropic')
# plt.plot(alpha_TESTS[start:], np.array([ICL_error(Ctr1, Ctr1, tau, alpha, kappa, rho, 100, alpha_test) for alpha_test in alpha_TESTS[start:]]), color=color_cycle[3], label = r'$C_\text{train}$ random covariance')
# plt.plot(alpha_TESTS[start:], np.array([ICL_error(Ctr2, Ctr2, tau, alpha, kappa, rho, 100, alpha_test) for alpha_test in alpha_TESTS[start:]]), color=color_cycle[4], label = r'$C_\text{train}$ powerlaw, power = 0.5')
plt.plot(alpha_TESTS[start:], np.array([ICL_error(Ctr4, Ctr4, tau, alpha, kappa, rho, 100, alpha_test) for alpha_test in alpha_TESTS[start:]]), color=color_cycle[3], label = r'$C_\text{train}$ powerlaw, power = 1')
plt.plot(alpha_TESTS[start:], np.array([ICL_error(Ctr3, Ctr3, tau, alpha, kappa, rho, 100, alpha_test) for alpha_test in alpha_TESTS[start:]]), color=color_cycle[4], label = r'$C_\text{train}$ powerlaw, power = 1.5')

plt.scatter(alpha_TESTS[start:], fulldata_m[0], color='#FAFAFA', edgecolors=color_cycle[2],s=200,linewidth=3,zorder=10)
plt.fill_between(alpha_TESTS[start:], np.array(fulldata_m[0]) - np.array(fulldata_s[0]), np.array(fulldata_m[0]) + np.array(fulldata_s[0]), color=color_cycle[2], alpha=0.2)

# plt.scatter(alpha_TESTS[start:], fulldata_m[1], color='#FAFAFA', edgecolors=color_cycle[3],s=150,linewidth=2.5,zorder=10)
# plt.fill_between(alpha_TESTS[start:], np.array(fulldata_m[1]) - np.array(fulldata_s[1]), np.array(fulldata_m[1]) + np.array(fulldata_s[1]), color=color_cycle[3], alpha=0.2)

# plt.scatter(alpha_TESTS[start:], fulldata_m[2], color='#FAFAFA', edgecolors=color_cycle[4],s=150,linewidth=2.5,zorder=10)
# plt.fill_between(alpha_TESTS[start:], np.array(fulldata_m[2]) - np.array(fulldata_s[2]), np.array(fulldata_m[2]) + np.array(fulldata_s[2]), color=color_cycle[4], alpha=0.2)

plt.scatter(alpha_TESTS[start:], fulldata_m[4], color='#FAFAFA', edgecolors=color_cycle[3],s=200,linewidth=3,zorder=10)
plt.fill_between(alpha_TESTS[start:], np.array(fulldata_m[4]) - np.array(fulldata_s[4]), np.array(fulldata_m[4]) + np.array(fulldata_s[4]), color=color_cycle[3], alpha=0.2)

plt.scatter(alpha_TESTS[start:], fulldata_m[3], color='#FAFAFA', edgecolors=color_cycle[4],s=200,linewidth=3,zorder=10)
plt.fill_between(alpha_TESTS[start:], np.array(fulldata_m[3]) - np.array(fulldata_s[3]), np.array(fulldata_m[3]) + np.array(fulldata_s[3]), color=color_cycle[4], alpha=0.2)

leg = plt.legend()
leg.get_frame().set_alpha(0)
plt.gca().spines['top'].set_color('lightgray')
plt.gca().spines['right'].set_color('lightgray')
plt.gca().spines['bottom'].set_color('lightgray')
plt.gca().spines['left'].set_color('lightgray')
plt.xscale('log')
plt.xlabel(r'Test-time $\alpha$ = $\ell$/d')
plt.xticks(fontsize=20)
plt.yscale('log')
plt.ylabel('ICL error')
plt.yticks(fontsize=20)
plt.gca().tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig(f'figs/{figurename}.pdf')



