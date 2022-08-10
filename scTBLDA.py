""" Single Cell Telescoping Bimodal Latent Dirichlet Allocation Model """

import pickle
import numpy as np
import time
import math
import pandas as pd
import scipy as sp
import scanpy as sc
import scanpy.external as sce
import anndata as ad
import h5py as h5
import copy
import matplotlib.pyplot as plt
import os.path
from os import path
import sys

import torch
from torch import nn
import torch.distributions as tdist
from torch.utils import data
from torch.autograd import grad
import torch.optim as optim
from torch.distributions import constraints

import pyro
import pyro.optim as poptim
import pyro.distributions as dist
from pyro.ops.indexing import Vindex
from pyro.infer import SVI, TraceMeanField_ELBO, TraceGraph_ELBO, Trace_ELBO, config_enumerate, TraceEnum_ELBO
from pyro.optim import ClippedAdam, MultiStepLR
from torch.optim import Adam

sc.settings.verbosity = 2
sc.logging.print_versions()
pyro.enable_validation(True)
pyro.set_rng_seed(1)


class Encoder(nn.Module):

    def __init__(self, n_inds, n_genes, n_snps, k_b, n_cells):
        super().__init__()
        self.fc1 = nn.Linear(n_genes, 300)
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150, k_b)
        self.bn1 = nn.BatchNorm1d(300)
        self.bn2 = nn.BatchNorm1d(150)
        self.bn3 = nn.BatchNorm1d(k_b, affine=False)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.sigmoid(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        out = self.bn2(out)
        out self.fc3(out)
        out = self.bn3(out)
        phi_est = self.sm(out)
        return(phi_est)

class scTBLDA(nn.Module):
    
    def __init__(self, n_inds, n_genes, n_snps, k_b, n_cells, anc_portion, cell_ind_matrix,\
                 x_norm, device, delta=0.05, mu=0.5, zeta=1.0, gamma=1.0, xi=1.0, sigma=1.0):
        super().__init__()
        self.n_inds = n_inds
        self.n_genes = n_genes
        self.n_snps = n_snps
        self.k_b = k_b
        self.n_cells = n_cells
        self.anc_portion = anc_portion.to(device)
        self.x_norm = x_norm.to(device)
        self.delta = delta
        self.mu = mu
        self.cell_ind_matrix = cell_ind_matrix.to(device)
        self.one_tensor = torch.tensor([1.0], device=device)
        self.zeta = torch.tensor([zeta], device=device)
        self.sigma = torch.ones([k_b], device=device) * sigma
        self.xi = torch.ones([n_genes], device=device) * xi
        self.gamma = torch.tensor([gamma], device=device)
        self.device = device
        self.encoder = Encoder(n_inds, n_genes, n_snps, k_b, n_cells)
    


    @pyro.poutine.scale(scale=1.0 / (self.n_cells * self.n_genes))
    def model(self, x, y, mb_cells, x_use=None, cell_ind_matrix=None):
        # declare plates
        snp_plt = pyro.plate('snps', self.n_snps, dim=-2)
        ind_plt = pyro.plate('inds', self.n_inds)
        k_b_plt = pyro.plate('k_b', self.k_b)
        cell_plt = pyro.plate('cells', x.shape[0])

        # global
        phi_prior = pyro.sample('phi_prior', dist.Dirichlet(self.sigma))
        with k_b_plt:
            lambda_g = pyro.sample("lambda_g", dist.Dirichlet(self.xi)) # [k_b, n_genes]

        # local - cell level. 
        with cell_plt as ind:
            phi = pyro.sample("phi", dist.Dirichlet(phi_prior))
            pi_g = torch.mm(phi, lambda_g) # [tot_cells, n_genes]
            pyro.sample('x', dist.Multinomial(probs=pi_g, validate_args=False), obs=x[ind])

        with snp_plt, k_b_plt:
            lambda_s = pyro.sample("lambda_s", dist.Beta(self.zeta, self.gamma)) # [n_snps, k_b]

        alpha = pyro.sample('alpha', dist.Uniform(torch.tensor([self.delta],device=self.device), \
                                                             torch.tensor([self.mu], \
                                                            device=self.device)))
        if cell_ind_matrix is None:
            cell_ind_matrix = self.cell_ind_matrix
        phi_ind = torch.mm(phi.t(), cell_ind_matrix[mb_cells]) #[k_b, n_inds]
        phi_ind = phi_ind / torch.sum(phi_ind, dim=0).view([1, self.n_inds])
        pi_s = (alpha * self.anc_portion)
        pi_s += ((1 - alpha) * torch.mm(lambda_s, phi_ind)) # [n_snps, n_inds]

        with snp_plt:#, ind_plt:
            pyro.sample('y', dist.Binomial(2, pi_s), obs=y) # [n_snps, n_inds]
            
    @pyro.poutine.scale(scale=1.0 / (self.n_cells * self.n_genes))
    def guide(self, x, y, mb_cells, x_use=None, cell_ind_matrix=None):
        
        # declare plates
        snp_plt = pyro.plate('snps', self.n_snps, dim=-2)
        ind_plt = pyro.plate('inds', self.n_inds)
        k_b_plt = pyro.plate('k_b', self.k_b)
        cell_plt = pyro.plate('cells', x.shape[0], subsample=mb_cells)

        # global

        xi_g = pyro.param("xi_g", \
                          lambda: dist.Uniform(0.5, 1.5).sample([self.k_b, self.n_genes]).to(self.device), \
                          constraint=constraints.positive)

        with k_b_plt:
            lambda_g = pyro.sample("lambda_g", dist.Dirichlet(xi_g)) # [k_phi, n_genes]

        phi_prior_post = pyro.param('phi_prior_post', lambda: dist.Dirichlet(torch.ones([self.k_b])).sample().to(self.device), constraint=constraints.simplex)
        phi_prior = pyro.sample("phi_prior", dist.Delta(phi_prior_post).to_event())
        
        # local - cell level. 
        pyro.module("encoder", self.encoder)
        if x_use is None:
            x_use = self.x_norm
        with cell_plt as ind:
            x_use = x_use[ind]
            phi = self.encoder(x_use)
            pyro.sample("phi", dist.Delta(phi, event_dim=1)) # [tot_cells, k_phi]

        alpha_p = pyro.param("alpha_p", torch.tensor([0.1]).to(self.device), \
                             constraint=constraints.interval(torch.tensor([self.delta],device=self.device), \
                                                             torch.tensor([self.mu],device=self.device)))
        alpha = pyro.sample("alpha", dist.Delta(alpha_p))
        

        zeta = pyro.param("zeta", lambda: dist.Uniform(0.5,1.5).sample([self.n_snps, self.k_b]).to(self.device), constraint=constraints.positive)
        gamma = pyro.param("gamma", lambda: dist.Uniform(0.5,1.5).sample([self.n_snps, self.k_b]).to(self.device), constraint=constraints.positive)
        with snp_plt, k_b_plt:
            lambda_s = pyro.sample("lambda_s", dist.Beta(zeta, gamma)) # [n_snps, k_b]
            
    

