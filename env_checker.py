import numpy
import scipy
import matplotlib
import seaborn
import tqdm
import jax
import flax
import optax
from reduced_model_codebase.reduced import spikevalue

print("task-align-icl is ready")
print("JAX devices:", jax.devices())
print("Example spike covariance diagonal:", spikevalue(4, 0, 1))