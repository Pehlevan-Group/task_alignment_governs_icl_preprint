import numpy as np
from scipy import optimize

# ----------------------------------- THEORY --------------------------------
# -------------------------------------------------------------------------------

def M_kappa_nu_empirical(kappa, nu, Ctr):
  d = len(Ctr)
  estimate = 0
  numavg = 500
  for _ in range(numavg):
    samples = np.random.multivariate_normal(np.zeros(d), Ctr, int(d*kappa)).T;
    samplecov = (samples@samples.T)/int(d*kappa)
    estimate = estimate + (1/d)*np.trace(np.linalg.inv(samplecov + nu*np.eye(d)))
  estimate = estimate/numavg
  return estimate

def M2_kappa_nu_empirical(kappa, nu, Ctr):
  d = len(Ctr)
  estimate = 0
  numavg = 500
  for _ in range(numavg):
    samples = np.random.multivariate_normal(np.zeros(d), Ctr, int(d*kappa)).T;
    samplecov = (samples@samples.T)/int(d*kappa)
    estimate = estimate + (1/d)*np.trace(np.linalg.inv((samplecov + nu*np.eye(d))@(samplecov + nu*np.eye(d))))
  estimate = estimate/numavg
  return estimate

def objectivefunc(xi, tau, kappa, Ctr, rhotr_alpha):
    return xi*M_kappa_nu_empirical(kappa, xi + rhotr_alpha, Ctr) + tau - 1

def xi_tau_less_1(tau, kappa, Ctr, rhotr_alpha):
    leftbound = 0
    rightbound = 2*rhotr_alpha*tau
    while objectivefunc(rightbound, tau, kappa, Ctr, rhotr_alpha) < 0:
        rightbound = rightbound+1
    root = optimize.brentq(objectivefunc, leftbound, rightbound, args=(tau, kappa, Ctr, rhotr_alpha))
    return root

def ICL_error(Ctr, Ctesthat, tau, alpha, kappa, rho):
    d = len(Ctr)
    rhotr = (1/d)*np.trace(Ctr) + rho
    rhotest = (1/d)*np.trace(Ctesthat) + rho;

    if tau == 1:
        return None
    if tau > 1:
        xi = 0
    if tau < 1:
        xi = xi_tau_less_1(tau, kappa, Ctr, rhotr/alpha)

    nu = rhotr/alpha + xi
    M =  M_kappa_nu_empirical(kappa, nu, Ctr)
    Mprime = -M2_kappa_nu_empirical(kappa, nu, Ctr);
    FR = np.linalg.inv((1 - 1/kappa + (nu/kappa)*M)*Ctr + nu*np.eye(d))
    FR2 = FR @ (((1/kappa)*M + (nu/kappa)*Mprime)*Ctr + np.eye(d)) @ FR

    idg = (rho + nu - (nu**2)*M - xi*(1 - 2*nu*M - (nu**2)*Mprime))/(tau - (1 - 2*xi*M - (xi**2)*Mprime))
    pretraining_term = rho + (rhotest/alpha)*(1 + (idg-2*nu)*M + (xi*idg - nu**2)*Mprime)
    interaction_term = idg*(1/d)*np.trace(Ctesthat@FR) - (idg*xi - nu**2)*(1/d)*np.trace(Ctesthat@FR2)
    return pretraining_term, interaction_term

def ealign(Ctr, Ctesthat, tau, alpha, kappa, rho, numavg=10):
    d = len(Ctr)
    rhotr = (1/d)*np.trace(Ctr) + rho
    rhotest = (1/d)*np.trace(Ctesthat) + rho;

    if tau == 1:
        return None
    if tau > 1:
        xi = 0
    if tau < 1:
        xi = xi_tau_less_1(tau, kappa, Ctr, rhotr/alpha)

    nu = rhotr/alpha + xi
    M =  M_kappa_nu_empirical(kappa, nu, Ctr)
    Mprime = -M2_kappa_nu_empirical(kappa, nu, Ctr);
    FR = np.linalg.inv((1 - 1/kappa + (nu/kappa)*M)*Ctr + nu*np.eye(d))
    FR2 = FR @ (((1/kappa)*M + (nu/kappa)*Mprime)*Ctr + np.eye(d)) @ FR

    idg = (rho + nu - (nu**2)*M - xi*(1 - 2*nu*M - (nu**2)*Mprime))/(tau - (1 - 2*xi*M - (xi**2)*Mprime))
    interaction_term = idg*(1/d)*np.trace(Ctesthat@FR) - (idg*xi - nu**2)*(1/d)*np.trace(Ctesthat@FR2)
    return interaction_term

def complexity_class_covariance(d, comp, traceadjust):
    diags = np.array(list(np.ones(comp)) + list(np.zeros(d-comp)))
    if traceadjust:
        return diags*d/comp
    return diags