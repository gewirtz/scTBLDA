""" Auxilliary functions that help run the model """

from scTBLDA import *


def import_data(expr_f, geno_f, beta_f, tau_f, samp_map_f, f_delim, device):
    
    """
    Loads data matrices

    Args:
        expr_f: h5ad file containing expression count data in X where X is cells x genes. This file should
                also have unique individual labels for each cell in the obs.ind_cov slot as integers

        geno_f: File containing minor allele counts [0,1,2], SNPs are rows, individuals are cols
        
        beta_f: File containing estimated beta matrix (ancestry topics)
        
        tau_f: File containing estimated tau matrix (individual ancestry proportions)
        
        f_delim: Delimiter character for reading in files

        device: GPU or CPU depending on availability
 
    Returns:
        x: [samples x genes] pytorch tensor of expression counts
        
        x_norm: [samples x genes] pytorch tensor of normalized expression counts
        
        y: [snps x individuals] pytorch tensor of minor allele counts [0,1,2]
        
        anc_portion: Estimated ancestral structure (genotype-specific space; product of zeta and gamma)
        
        cell_ind_matrix: [cells x individuals] pytorch indicator tensor where each row has a single
                         1 coded at the position of the donor individual
    """

    x_file = ad.read_h5ad(expr_f, 'r') # x_file.X is cells x genes
    x = torch.from_numpy(x_file.X[:].todense())
    x = x.to(device)
    y = torch.from_numpy(np.genfromtxt(geno_f, delim=f_delim).astype('int8'))
    y = y.to(device)
    x_norm = torch.log(1 + x)
    x_norm = x_norm.to(device)

    anc_loadings = torch.from_numpy(np.genfromtxt(beta_f, delimiter=f_delim))
    anc_facs = torch.from_numpy(np.genfromtxt(tau_f, delimiter=f_delim)).t()
    anc_portion = torch.mm(anc_loadings, anc_facs)
    del(anc_loadings)
    del(anc_facs)

    cell_inds = torch.tensor(x_file.obs.ind_cov.astype('int8').tolist())

    n_inds = y.shape[1]
    n_cells = x.shape[0]
    print('X Matrix dimensions: ' + x.shape)
    print('Y Matrix dimensions: ' + y.shape)

    cell_ind_matrix = torch.zeros([n_cells, n_inds])
    for cell in range(n_cells):
        cell_ind_matrix[cell, cell_inds[cell].item()] = 1

    return(x, x_norm, y, anc_portion, cell_ind_matrix)



def run_vi(sctblda, x, y, lr=0.002, n_epochs=250, seed=29, verbose=True, write_its=50):
        """
        Run variational inference through Pyro to fit the TBLDA model
    
    Args:
        sctblda: scTBLDA object
        
        x: [samples x genes] pytorch tensor of expression counts
        
        y: [snps x individuals] pytorch tensor of minor allele counts [0,1,2]
        
        lr: Learning rate

        n_epochs: Number of epochs to run inference for
        
        seed: Value to seed the random number generator with
        
        write_its: How often to write intermediate output

        verbose: Whether to print out information such as epoch progression
        """
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()

        n_cells = x.shape[0]
        n_inds = y.shape[1]

        cells_per_mb = int(np.ceil(n_cells / n_minibatches))
        if verbose:
            print('There are ' + str(n_minibatches) + ' minibatches with ' + str(cells_per_mb) + ' cells.', flush=True)
    
        # calculate cells/individual to include in each minibatch
        cells_pp = np.floor(np.bincount(cell_inds) / n_minibatches)
    
        opt1 = poptim.ClippedAdam({"lr": lr})
        svi = SVI(sctblda.model, sctblda.guide, opt1, loss=Trace_ELBO())

        mb_inds = torch.ones([n_cells], dtype=torch.int16) * -2
        # go through each person and assign their cells equally among minibatches
        for person in range(n_inds):
            p_cell_inds = (cell_inds==person).nonzero().squeeze()
            rand_indx = torch.randperm(p_cell_inds.nelement())
            # take the cells_pp and assign them to a minibatch
            for mb in range(n_minibatches):
                minibatch_start = (mb * int(cells_pp[person]))
                minibatch_end = min((minibatch_start + int(cells_pp[person])), (p_cell_inds.nelement()-1))
                mb_inds[p_cell_inds[rand_indx[minibatch_start:minibatch_end]]] = mb
        # now assign all leftover cells
        cells_in_mb = np.bincount(mb_inds[mb_inds>=0])
        for mb in range(n_minibatches):
            cur_mb_size = cells_in_mb[mb]
            if(cur_mb_size < cells_per_mb):
                n_cells_to_add = cells_per_mb - cur_mb_size
                avail_cells = (mb_inds<0).nonzero().squeeze()
                if avail_cells.nelement()==0:
                    next
                if avail_cells.nelement()==1:
                    mb_inds[avail_cells] = mb
                else:
                    rand_indx = torch.randperm(avail_cells.nelement())
                    mb_inds[avail_cells[rand_indx[0:n_cells_to_add]]] = mb
        
        losses = [] 
        for epoch in range(n_epochs):
            if verbose:
                print('EPOCH ' + str(epoch), flush=True)
            elbo = 0.0
            for mb in range(n_minibatches):
                x_inds = (mb_inds==mb).nonzero().squeeze().numpy()
                elbo += svi.step(x, y, x_inds) 
            losses.append(elbo / float(n_minibatches))
            if ( (epoch % write_its) == 0) and epoch > 0 :
                pyro.get_param_store().save(('results_' + str(epoch) + '_epochs.save'))
                with open('results_' + str(epoch) + '_epochs_loss.data'), 'wb') as filehandle:
                    pickle.dump(losses, filehandle)
                # remove old files
                if epoch > write_its:
                    os.remove('results_' + str(epoch - write_its) + '_epochs.save')
                    os.remove('results_' + str(epoch - write_its) + '_epochs_loss.data')

        pyro.get_param_store().save(('results_' + str(epoch) + '_epochs.save'))
        with open('results_' + str(epoch) + '_epochs_loss.data'), 'wb') as filehandle:
            pickle.dump(losses, filehandle)


