## Mamba Usage
0. Create a python environment in terminal, copy code from dgp_mamba.py into a python script. Then simply either type "python dgp_mamba.py", or submit a long job for all scripts. Repeat this process for the following scripts in order:
1. train_mamba.py - The training loop for the Mamba Architecture Fitting to the DGP 500 samples. Takes a couple of hours
2. train_arma.py - The training loop for the ARMA Fitting to the DGP 500 samples. Takes a couple of hours
3. arma_test2.py - The rolling forecast loop for the ARMA model on the DGP. Takes days.
4. mamba_test2.py - The rolling forecast loop for the Mamba model on the DGP. Takes Days
6. seedsearch.py - Not neccesary unless you want to change seeds
7. realts_arma.py - No need for long job, just run "python realts_arma.py". It is the ARMA training plus rolling forecast script for the Nile River Data
8. realts_mamba.py - No need for long job, just run "python realts_mamba.py". It is the Mamba training plus rolling forecast script for the Nile River Data
