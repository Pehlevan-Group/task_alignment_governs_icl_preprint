# INSTRUCTIONS for Figure 6

`run.py` includes the theory code necessary for constructing the data, which are stored in `csv` files once ran. 

To use this, please run `script.sh` to batch a job array over the $\kappa$ values. This isn't strictly necessary, technically you could run this code in a loop, but it's far less efficient. 

The `run.py` code takes the following values in order.
1. `directory`, this is where all data computed during the run will be saved 
2. `d` desired token dimension
3. `alpha` desired $\alpha = \ell/d$ value
4. `tau` desired $\tau = n/d^2$ value 
5. `numavg` number of times to average over various random quantities
6. `kappaind` **index** of the array of $\kappa$ values. We have written it this way for ease of job array batching. 

So in other words, the code used in `script.sh` 
```
python run.py $newdir 80 80 80 30 $SLURM_ARRAY_TASK_ID
```
runs this with a new directory at `$newdir`, with $d=80$, $\alpha = 80$, $\tau = 80$, and whatever $\kappa$ value is stored at `kappas[$SLURM_ARRAY_TASK_ID]`.

To use the plotting code, first run `script.sh` to make sure the data is populated, then you will be prompted for the values 
```
d = int(input("d: "))
alpha = float(input("alpha: "))
tau = float(input("tau: "))
experiment = input("experiment: ")
figurename = input("figurename: ")
```
Use the same directory name you used for `run` or `script` as the `experiment` field, as this is where it will look for the data stored. 