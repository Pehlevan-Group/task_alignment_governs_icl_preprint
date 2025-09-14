import numpy as np
import optax
from theory import *
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from trainmini import train
from model.transformer import TransformerConfig
from task.regression_structured import fulltasksampler, finitetasksampler

rho = 0.01
d = int(sys.argv[1]);
alpha = 2; l = int(alpha*d)
tau = 4; n = int(tau*(d**2));
kappa = 1; k = int(kappa*d);

h = d+1;

myname = sys.argv[2] # grab value of $mydir to add results
avgind = int(sys.argv[3]) # average index specified by array

# train_power = 0.9
# Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d
Ctr = np.eye(d)

trainobject = finitetasksampler(d, l, n, k, rho, Ctr)
testobject_1 = fulltasksampler(d, l, n, rho, Ctr)
testobject_2 = fulltasksampler(d, l, n, rho, Ctr)

config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=1, n_mlp_layers=0, pure_linear_self_att=True)
state, hist = train(config, data_iter=iter(trainobject), test_1_iter=iter(testobject_1), test_2_iter=iter(testobject_2), batch_size=16, loss='mse', test_every=100, train_iters=1000, optim=optax.adamw,lr=1e-4)

print('TRAINING DONE',flush=True)
# file_path = f'./{myname}/pickles/train-{avgind}.pkl'
# with open(file_path, 'wb') as fp:
#     pickle.dump(hist, fp)

alphas_TEST = np.logspace(np.log10(0.5),np.log10(100),20)

sametraintest_test_m = []
sametraintest_test_s = []
for alphatest in alphas_TEST:
    loss_func = optax.squared_error
    numsamples = 500
    testobject = fulltasksampler(d, int(alphatest*d), n, rho, Ctr)
    tracker = []
    for _ in range(numsamples):
        xs, labels = next(testobject); # generates data
        logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
        tracker.append(loss_func(logits, labels).mean())
    tracker = np.array(tracker)
    sametraintest_test_m.append(np.mean(tracker))
    sametraintest_test_s.append(np.std(tracker))
    print(f'{alphatest} isnt too big')
file_path = f'./{myname}/test_pretrain_m_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{sametraintest_test_m}')
file_path = f'./{myname}/test_pretrain_s_{avgind}.txt'
with open(file_path, 'a') as file: 
    file.write(f'{sametraintest_test_s}')
print('DONE: TESTING ON PRETRAIN')

# smaller_test = 0.5
# smaller_Ctest = np.diag(np.array([(j + 1) ** -smaller_test for j in range(d)])); smaller_Ctest = (smaller_Ctest/np.trace(smaller_Ctest))*d
# smallerpower_test_m = []
# smallerpower_test_s = []
# for alphatest in alphas_TEST:
#     loss_func = optax.squared_error
#     numsamples = 500
#     testobject = fulltasksampler(d, int(alphatest*d), n, rho, smaller_Ctest)
#     tracker = []
#     for _ in range(numsamples):
#         xs, labels = next(testobject); # generates data
#         logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
#         tracker.append(loss_func(logits, labels).mean())
#     tracker = np.array(tracker)
#     smallerpower_test_m.append(np.mean(tracker))
#     smallerpower_test_s.append(np.std(tracker))
# file_path = f'./{myname}/test_smallerpower_m_{avgind}.txt'
# with open(file_path, 'a') as file:
#     file.write(f'{smallerpower_test_m}')
# file_path = f'./{myname}/test_smallerpower_s_{avgind}.txt'
# with open(file_path, 'a') as file: 
#     file.write(f'{smallerpower_test_s}')
# print('DONE: TESTING ON SMALLER POWER')

# larger_test = 1.5
# larger_Ctest = np.diag(np.array([(j + 1) ** -larger_test for j in range(d)])); larger_Ctest = (larger_Ctest/np.trace(larger_Ctest))*d
# largerpower_test_m = []
# largerpower_test_s = []
# for alphatest in alphas_TEST:
#     loss_func = optax.squared_error
#     numsamples = 500
#     testobject = fulltasksampler(d, int(alphatest*d), n, rho, larger_Ctest)
#     tracker = []
#     for _ in range(numsamples):
#         xs, labels = next(testobject); # generates data
#         logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
#         tracker.append(loss_func(logits, labels).mean())
#     tracker = np.array(tracker)
#     largerpower_test_m.append(np.mean(tracker))
#     largerpower_test_s.append(np.std(tracker))
# file_path = f'./{myname}/test_largerpower_m_{avgind}.txt'
# with open(file_path, 'a') as file:
#     file.write(f'{largerpower_test_m}')
# file_path = f'./{myname}/test_largerpower_s_{avgind}.txt'
# with open(file_path, 'a') as file: 
#     file.write(f'{largerpower_test_s}')
# print('DONE: TESTING ON LARGER POWER')

