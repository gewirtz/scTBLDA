# scTBLDA

# Usage

## Requirements

scTBLDA is implemented in Python 3.8. It requires the following packages (used versions are in parentheses):

1. pandas 1.0.5
2. numpy 1.18.5
3. torch 1.10.0
4. pyro 1.4.0
5. h5py 2.10.0

## Running TBLDA

**Required Input:**

1. `expr_f`: Path to the file containing expression count data with samples as rows and genes as columns
2. `geno_f`: Path to the file containing genotype data in dosage format [0,1,2] with SNPs as rows and individuals as columns
3. `beta_f`: Path to the file containing estimated ancestry topics with SNPs as rows and topics as columns
4. `tau_f`: Path to the file containing ancestry proportions with SNPs as rows and individuals as columns
5. `samp_map_f`: Path to the file containing an L-length vector of individual IDs, where L is the total number of samples and individuals are given an ID in [0,N] where N is the total number of individuals
6. `K`: The number of shared latent factors in the model. We recommend starting with K in between 5 and 100. Defaults to 50.

**Optional Input:**

1. Seed (`--seed`): Value to seed the random number generator. Defaults to 21.
2. File delimiter (`--file_delim`): Character that separates columns across all files. Defaults to tab.
3. Learning rate (`--lr`): Learning rate for inference. Defaults to 0.05.
4. Number of epochs (`--n_epochs`). Defaults to 200.
5. Write iterations (`--write_its`): Specifies how often intermediate results are saved (write output every [X] iterations). Defaults to 10.
6. Xi (`--xi`): Per-factor gene hyperparameter. Defaults to 0.02.
7. Zeta (`--zeta`): Per-factor genotype hyperparameter. Defaults to 0.02.
8. Gamma (`--gamma`): Per-factor genotype hyperparameter. Defaults to 0.02.
9. Sigma (`--sigma`): Per-cell factor proportion hyperparameter. Defaults to 0.02.
10. Mu (`--mu`): Maximum weight given to the genotype-specific subspace. Defaults to 0.7.
11. Delta (`--delta`): Minimum weight given to the genotype-specific subspace. Defaults to 0.05.
12 Number of minibatches to use for inference (`--n_minibatches`). Defaults to 5.

  **Output**: X refers to the number of epochs run
  
  1. results_X_epochs.save: This contains the pyro parameter store with the estimates of xi, sigma, zeta, and gamma.
  2. results_X_epochs_loss.data: This contains the loss estimates at every epoch
