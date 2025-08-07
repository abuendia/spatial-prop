import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
import pickle
import os
import copy
import random
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, one_hot, to_networkx


def generate_from_anndata ():
    '''
    Generates all predictions for AnnData object:
        loader - DataLoader object
        model - torch_geometric model object
        perturbation [str] - key in adata.obsm containing perturbed expression
        inject [bool] - whether model requires center cell type
    
    Output is adata with "predicted" and "predicted_perturbed" in adata.obsm
        If perturbation is "batch_steer", will also add "target"
    '''
    # Create a dataloader that creates batches of graphs from the adata object
    
    return ()


def propagate_perturbation (adata, method='distribution', temper=0.05):
    '''
    Given outputs from generate_from_anndata computes the perturbed gene expression for each cell:
        adata [AnnData] - output from generate_from_dataloader() or generate_from_anndata()
        method [str] - method for tempering predicting perturbation ("prediction_delta", "distribution")
        temper [float] - parameter for tempering
                       - for 'distribution', temper and 1-temper are quantile cutoffs for constraining perturbation
                       
    Output is adata with perturbed expression in adata.obsm["perturbed"]
    '''
    # extract results from adata
    true_expn = adata.X
    pred_expn = adata.obsm["predicted"]
    pred_perturb_expn = adata.obsm["predicted_perturbed"]
    
    # temper and compute perturbed expression
    true_perturb_expn = temper (true_expn, pred_expn, pred_perturb_expn, method=method, temper=temper)
    
    # add to adata
    adata.obsm["perturbed"] = true_perturb_expn
    
    return (adata)
	
	
def temper (true_expn, pred_expn, pred_perturb_expn, method="distribution", temper=0.05):
    '''
    Compute the perturbed gene expression given:
        true_expn - array (cell x gene) of true unperturbed expression
        pred_expn - array (cell x gene) of predicted unperturbed expression
        pred_perturb_expn - array (cell x gene) of predicted perturbed expression
        temper - float parameter (for distribution, this indicates the temper and 1-temper cutoffs for constraining perturb)
    Returns:
        true_perturb_expn - array (cell x gene) of perturbed expression
    '''
    def renorm_expression (s, p_hat):
        row_norm = torch.sum(s, dim=1) / torch.sum(p_hat, dim=1)
        p = p_hat * row_norm.unsqueeze(1)
        return (p)
    
    if method == "prediction_delta":
        
        # computes effect of perturbation in prediction space
        diff = pred_perturb_expn - pred_expn
        diff[torch.abs(diff) < torch.quantile(torch.abs(diff), 0.9)] = 0 # clip to top 10%
        true_perturb_expn = true_expn + diff
    
    elif method == "distribution":
        
        # computes prediction error distribution to calibrate
        errors = true_expn - pred_expn
        
        # compute quantiles of error distribution
        percentile_cutoff_top = torch.quantile(errors, 1-temper, axis=0)
        percentile_cutoff_bottom = torch.quantile(errors, temper, axis=0)
        
        # get perturbation mask based on cutoffs (same gene)
        pred_perturb_delta = true_expn - pred_perturb_expn
        perturb_mask = torch.logical_or((percentile_cutoff_top < pred_perturb_delta), (percentile_cutoff_bottom > pred_perturb_delta))
        
        # further mask so that perturbation must be greater magnitude than prediction error (same gene, same cell)
        perturb_mask = torch.logical_and(perturb_mask, torch.abs(pred_perturb_delta) > torch.abs(errors))
        
        # further mask to remove negative values (i.e. left piecewise converts all negatives to zero)
        perturb_mask = torch.logical_and(perturb_mask, pred_perturb_expn >= 0)
        
        # mask perturbations
        true_perturb_expn = true_expn.clone()
        true_perturb_expn[perturb_mask] = pred_perturb_expn[perturb_mask]
    
    
    elif method == "distribution_dampen":
        
        # computes prediction error distribution to calibrate
        errors = true_expn - pred_expn
        
        # compute quantiles of error distribution
        percentile_cutoff_top = torch.quantile(errors, 1-temper, axis=0)
        percentile_cutoff_bottom = torch.quantile(errors, temper, axis=0)
        
        # get perturbation mask based on cutoffs (same gene)
        pred_perturb_delta = true_expn - pred_perturb_expn
        perturb_mask = torch.logical_or((percentile_cutoff_top < pred_perturb_delta), (percentile_cutoff_bottom > pred_perturb_delta))
        
        # further mask so that perturbation must be greater magnitude than prediction error (same gene, same cell)
        perturb_mask = torch.logical_and(perturb_mask, torch.abs(pred_perturb_delta) > torch.abs(errors))
        
        # compute calibrated update to perturbation (i.e. right piecewise performs calibrated rescaling)
        def get_p (s, p_hat):
            L = get_L(s, p_hat)
            p = (1-L)*s+L*p_hat
            return(p)
        def get_L (s, p_hat):
            L = 1/(1+torch.log(torch.abs(s-p_hat)+1)) # logarithm decay of L from 1 to 0
            return (L)
        #calibrated_perturb_expn = true_expn + ((pred_perturb_expn - pred_expn) * ((1+true_expn)/(1+pred_expn)))
        calibrated_perturb_expn = get_p(true_expn, pred_perturb_expn)
        
        # further mask to remove negative values (i.e. left piecewise converts all negatives to zero)
        perturb_mask = torch.logical_and(perturb_mask, calibrated_perturb_expn >= 0)
        
        # mask perturbations
        true_perturb_expn = true_expn.clone()
        true_perturb_expn[perturb_mask] = calibrated_perturb_expn[perturb_mask]
    
    
    elif method == "distribution_renormalize":
        
        # computes prediction error distribution to calibrate
        errors = true_expn - pred_expn
        
        # compute quantiles of error distribution
        percentile_cutoff_top = torch.quantile(errors, 1-temper, axis=0)
        percentile_cutoff_bottom = torch.quantile(errors, temper, axis=0)
        
        # get perturbation mask based on cutoffs (same gene)
        pred_perturb_delta = true_expn - pred_perturb_expn
        perturb_mask = torch.logical_or((percentile_cutoff_top < pred_perturb_delta), (percentile_cutoff_bottom > pred_perturb_delta))
        
        # further mask so that perturbation must be greater magnitude than prediction error (same gene, same cell)
        perturb_mask = torch.logical_and(perturb_mask, torch.abs(pred_perturb_delta) > torch.abs(errors))
        
        # further mask to remove negative values
        perturb_mask = torch.logical_and(perturb_mask, pred_perturb_expn >= 0)
        
        # mask perturbations
        true_perturb_expn = true_expn.clone()
        true_perturb_expn[perturb_mask] = pred_perturb_expn[perturb_mask]
        
        # compute renormalization
        true_perturb_expn = renorm_expression(true_expn, true_perturb_expn)
        
        
    elif method == "renormalize":
        
        # further mask to remove negative values
        perturb_mask = pred_perturb_expn >= 0
        
        # mask perturbations
        true_perturb_expn = true_expn.clone()
        true_perturb_expn[perturb_mask] = pred_perturb_expn[perturb_mask]
        
        # compute renormalization
        true_perturb_expn = renorm_expression(true_expn, true_perturb_expn)
        
    else:
        raise Exception ("temper method not recognized")
        
    return (true_perturb_expn)
	

def perturb_data (data, perturbation, mask=None, method="add"):
    '''
    Makes perturbation given a graph data object, perturbation, and cell mask
        data - torch_geometric input data object corresponding to subgraph for a cell
        perturbation - array of gene perturbations (of length data.x.shape[1])
        mask - boolean array for which cells to perturb (of length data.x.shape[0])
        method - str specifying how to perturb ("add", "multiply"); default is "add"
    '''
    data_perturbed = data
    if mask is None:
        if method == "add":
            data_perturbed.x = data.x + perturbation
        elif method == "multiply":
            data_perturbed.x = data.x * perturbation
    else:
        if method == "add":
            data_perturbed.x[mask,:] = data.x[mask,:] + perturbation
        elif method == "multiply":
            data_perturbed.x[mask,:] = data.x[mask,:] * perturbation
    return (data_perturbed)
	
    
def predict (model, data, inject=False):
    '''Get predicted expression
    '''
    if inject is False:
        out = model(data.x, data.edge_index, data.batch, None)
    else:
        out = model(data.x, data.edge_index, data.batch, data.inject)
    return (out)


def get_center_celltypes (data):
    '''Get center cell types in data
    '''
    celltypes = []
    for i in range(len(data.celltypes)):
        celltypes.append(data.celltypes[i][data.center_node[i]])
    celltypes = np.array(celltypes)
    return (celltypes)
	
    
### STEERING MODULES
def batch_steering_mean (data, actual, out, center_celltypes, target=None, prop=1.0):
    '''
    Make perturbations by steering all graphs (of same cell type) to the mean
    cell type-specific expression of the first graph in the batch
    
    target [None or PyG data object] - if None, then uses first graph in data batch, otherwise should be a PyG data object
    prop [float 0 to 1] - proportion of steering to pursue
    '''
    data = data.clone()
    
    # extract first graph as target
    if target is None:
        target_celltype = center_celltypes[0]
        target_x = data.x[data.batch==0,:]
        target_y = actual[0,:]
        target_out = out[0,:]

    # get mean cell type expression for target
    target_celltypes = data.celltypes[0]
    target_mean_ct = []
    for ct in np.unique(target_celltypes):
        target_mean_ct.append(torch.mean(target_x[target_celltypes==ct,:], axis=0))

    # perturb all graphs with same center cell type as first graph
    for bi in np.unique(data.batch):
        if (center_celltypes[bi] == target_celltype) and (bi>0):
            origin_x = data.x[data.batch==bi,:]
            net_celltypes = data.celltypes[bi]
            # perturb only cell types also in target
            perturb_vecs = []
            for i, ct in enumerate(np.unique(target_celltypes)):
                if ct in net_celltypes:
                    origin_mean_exp = torch.mean(origin_x[net_celltypes==ct,:], axis=0)
                    perturb_vec = target_mean_ct[i] - origin_mean_exp
                    # perturb data
                    submask = np.arange(data.x.shape[0])[data.batch==bi][net_celltypes==ct]
                    data = perturb_data (data, prop*perturb_vec, mask=submask)

    return (data, target_celltype, target_y, target_out)


def batch_steering_cell (data, actual, out, center_celltypes, target=None, prop=1.0):
    '''
    Make perturbations by steering all graphs (of same cell type) by
    replacing cells with random draws from target graph
    
    target [None or PyG data object] - if None, then uses first graph in data batch, otherwise should be a PyG data object
    prop [float 0 to 1] - proportion of steering to pursue    
    '''
    data = data.clone()
    
    # extract first graph as target
    if target is None:
        target_celltype = center_celltypes[0]
        target_x = data.x[data.batch==0,:]
        target_y = actual[0,:]
        target_out = out[0,:]
    else: # no target_out for OOD predictions since gene panels don't match
        target_celltype = target.celltypes[target.center_node]
        target_x = target.x
        target_y = target.y
        target_out = torch.full_like(target_y, float('nan'))

    # perturb all graphs with same center cell type as first graph
    for bi in np.unique(data.batch):
        if (center_celltypes[bi] == target_celltype) and (bi>0):
            # make full random draws from target
            origin_x = data.x[data.batch==bi,:]
            torch.manual_seed(444)
            rand_idxs = torch.randint(target_x.shape[0], size=(origin_x.shape[0],))
            full_perturbed_x = target_x[rand_idxs,:]
            
            # prop to determine number to replace first
            origin_x_perturbed = origin_x.clone()
            
            # update perturbation (mask out missing genes and only update both on measured genes)
            missing_mask = np.median(target_x, axis=0) != -1
            origin_x_perturbed[:round(prop*origin_x.shape[0]),missing_mask] = full_perturbed_x[:round(prop*origin_x.shape[0]),missing_mask]
            #origin_x_perturbed[:round(prop*origin_x.shape[0]),:] = full_perturbed_x[:round(prop*origin_x.shape[0]),:]
            
            # update data with perturbation
            data.x[data.batch==bi,:] = origin_x_perturbed

    return (data, target_celltype, target_y, target_out)
    

### GO INTERACTION MODULES
def perturb_by_multiplier (data, gene_indices, perturb_celltype=None, prop=1.0):
    '''
    Multiplies gene expression by a scalar factor for a set of indices
    '''
    data = data.clone()

    # perturb all graphs with same center cell type as first graph
    for bi in np.unique(data.batch):
        origin_x = data.x[data.batch==bi,:]

        net_celltypes = data.celltypes[bi]
        
        # create multiplication mask (1 or prop for modified genes)
        perturb_vec = torch.ones(data.x.shape[1])
        perturb_vec[gene_indices] = prop
    
        # get cell mask
        submask = np.arange(data.x.shape[0])[data.batch==bi]
        if perturb_celltype is not None:
            submask = submask[data.celltypes[bi] == perturb_celltype]
        
        # make perturbation
        data = perturb_data (data, perturb_vec, mask=submask, method="multiply")

    return (data)