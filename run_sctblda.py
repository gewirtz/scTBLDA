""" Reads in data and runs inference for the scTBLDA model """

from def_params import *
from scTBLDA import *
from helper_functions import *
import argparse

parser = argparse.ArgumentParser(description='Run scTBLDA')
parser.add_argument("expr_f", \
                    help="Expression count file with samples as rows")
parser.add_argument("geno_f", \
                    help="Genotype file in dosage format with SNPs as rows")
parser.add_argument("beta_f", \
                    help="File of estimated ancestry topics with SNPs as rows")
parser.add_argument("tau_f", \
                    help="File with ancestry proportions per individual with donors as rows")
parser.add_argument("samp_map_f", \
                    help="File of individual IDs [0,...,N] for each sample")
parser.add_argument("--seed", type=int, default=21,\
                    help="Random seed")
parser.add_argument("K", type=int, default=50, \
                    help="Number of latent shared topics")
parser.add_argument("--file_delim", default='tab', choices=['tab','space','comma'],\
                    help="Delimiter for all files")
parser.add_argument("--lr", default=0.002, \
                    help="Learning rate")
parser.add_argument("--n_epochs", default=200, \
                    help='Number of epochs to run')
parser.add_argument("--write_its", default=10, \
                    help='Write intermediate output every <X> iterations')
parser.add_argument("--xi", default=0.02, \ 
                    help='per-factor gene hyperparameter')
parser.add_argument("--sigma", default=0.02,\
                    help='per-cell factor proportion hyperparameter')
parser.add_argument("--zeta", default=0.02, \
                    help='per-factor genotype hyperparameter')
parser.add_argument("--gamma", default=0.02, \
                    help='per-factor genotype hyperparameter')
parser.add_argument("--mu", default=0.7, \
                    help='Maximum weight given to the genotype-specific subspace')args = parser.parse_args()
parser.add_argument("--delta", default=0.05, \
                    help='Minimum weight given to the genotype-specific subspace')args = parser.parse_args()

if args.file_delim == 'tab':
    f_delim = '\t'
elif args.file_delim == 'space':
    f_delim = ' '
else:
    f_delim = ','

# check argument validity
if args.K < 2:
    raise argparse.ArgumentTypeError('Value must be at least 2 (minimum recommended 5)')
if args.lr <= 0 or args.lr >= 1:
    raise argparse.ArgumentTypeError('Learning rate must be between 0 and 1')
if args.delta > args.mu:
    raise argparse.ArgumentTypeError('delta must be larger than mu')
if args.delta < 0.01 or args.delta > 0.99:
    raise argparse.ArgumentTypeError('You must choose a valid value for delta')
if args.mu < 0.01 or args.mu > 0.99:
    raise argparse.ArgumentTypeError('You must choose a valid value for mu')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read in data
x, y, anc_portion, cell_ind_matrix = import_data(args.expr_f, args.geno_f, args.beta_f, \
                                                    args.tau_f, args.samp_map_f, f_delim, device)
# create scTBLDA object
sctblda = scTBLDA(n_inds=y.shape[1], n_genes=x.shape[1], n_snps=y.shape[0], k_b=k_b, \
                    n_cells=x.shape[0], anc_portion=anc_portion, cell_ind_matrix=cell_ind_matrix,\
                    xi=args.xi, sigma=args.sigma,x_norm=x_norm, device=device, zeta=args.zeta, \
                    gamma=args.gamma, delta=args.delta, mu=args.mu)

sctblda = sctblda.to(device)


run_vi(sctblda, x, y, lr=args.lr, n_epochs=args.n_epochs, seed=args.seed, verbose=True, write_its=args.write_its)

