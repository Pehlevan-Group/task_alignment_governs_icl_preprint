import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))  # Adds the parent directory to the module search path
from common import *

directory = sys.argv[1]
d = int(sys.argv[2])
alpha = float(sys.argv[3])
tau = float(sys.argv[4])
avgind = int(sys.argv[5])
option = int(sys.argv[6])
kappa = 1
print('kappa is ', kappa)

signals = np.int64(np.linspace(0,d-1,d//2))

if option == 0:
    Ctr = np.eye(d)
if option == 1:
    G = np.random.randn(d,d)
    Ctr = G@G.T; Ctr = (d/np.trace(Ctr))*Ctr
    filename = f'{directory}/matrix.txt'
    with open(filename, 'a') as file:
        file.write(f'{Ctr.tolist()},')
if option == 2:
    Ctr = np.diag(np.array([(j + 1) ** (-float(0.5)) for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d
if option == 3:
    Ctr = np.diag(np.array([(j + 1) ** (-float(1.5)) for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d
if option == 4:
    Ctr = np.diag(np.array([(j + 1) ** (-float(1.0)) for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d


rho = 0.01
test_errors = []

alpha_TESTS = np.logspace(np.log10(0.1), np.log10(1000), 21)

runs = []
Gamma = final_gamma(d, tau, alpha, kappa, rho, Ctr, lam=0.0001)
for alpha_TEST in alpha_TESTS:
    Ctest = Ctr
    runs.append(trace_formula_gamma(d, rho, int(alpha_TEST*d), np.zeros(d), Ctest, Gamma))

filename = f'{directory}/simulation_{option}.txt'
with open(filename, 'a') as file:
    file.write(f'[{avgind}, {runs}],')

