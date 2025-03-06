import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.stats import pearsonr, spearmanr
import pickle
import os
from decimal import Decimal
import copy
import random

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

os.chdir("/labs/abrunet1/Eric/GNNPerturbation")

from aging_gnn_model import *
from perturbation import *

import argparse

# set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("steering_approach", help="steering method to use", type=str)
parser.add_argument("num_props", help="number of intervals from 0 to 1", type=int)
parser.add_argument("train_or_test", help="train or test dataset", type=str)
parser.add_argument("ood_subfoldername", help="subfolder name for OOD dataset", type=str)
args = parser.parse_args()

# load parameters from arguments
steering_approach = args.steering_approach
num_props = args.num_props
props = np.linspace(0,1,num_props+1) # proportions to scale
train_or_test = args.train_or_test
ood_subfoldername = args.ood_subfoldername


###### load in dataset ########################################################


# get params
k_hop = 2
augment_hop = 2
center_celltypes = ["T cell","NSC","Pericyte"]
node_feature = "expression"
loss = "weightedl1"
learning_rate = 0.0001
use_model = "model"
#use_model = "best_model"
inject_feature = "center_celltype"
inject=False

#--------------------------------------------------------------
# train
train_ids = [
    ['1','101','14','19','30','38','42','46','53','61','7','70','75','80','86','97'], # aging coronal
    ['2','34','39','62','68'], # aging sagittal
    None, # exercise
    None, # reprogramming
#     ['MsBrainAgingSpatialDonor_10_0', 'MsBrainAgingSpatialDonor_10_1', 'MsBrainAgingSpatialDonor_10_2', 'MsBrainAgingSpatialDonor_11_0', 'MsBrainAgingSpatialDonor_11_1', 'MsBrainAgingSpatialDonor_11_2', 'MsBrainAgingSpatialDonor_12_0', 'MsBrainAgingSpatialDonor_12_1', 'MsBrainAgingSpatialDonor_13_1', 'MsBrainAgingSpatialDonor_13_2', 'MsBrainAgingSpatialDonor_14_1', 'MsBrainAgingSpatialDonor_15_0', 'MsBrainAgingSpatialDonor_15_1', 'MsBrainAgingSpatialDonor_16_0', 'MsBrainAgingSpatialDonor_16_1', 'MsBrainAgingSpatialDonor_17_0', 'MsBrainAgingSpatialDonor_17_1', 'MsBrainAgingSpatialDonor_18_0', 'MsBrainAgingSpatialDonor_18_1', 'MsBrainAgingSpatialDonor_19_0', 'MsBrainAgingSpatialDonor_19_1', 'MsBrainAgingSpatialDonor_19_2', 'MsBrainAgingSpatialDonor_2_0', 'MsBrainAgingSpatialDonor_2_1', 'MsBrainAgingSpatialDonor_3_0', 'MsBrainAgingSpatialDonor_3_1', 'MsBrainAgingSpatialDonor_4_0', 'MsBrainAgingSpatialDonor_4_1', 'MsBrainAgingSpatialDonor_4_2', 'MsBrainAgingSpatialDonor_5_0', 'MsBrainAgingSpatialDonor_5_1', 'MsBrainAgingSpatialDonor_5_2', 'MsBrainAgingSpatialDonor_6_0', 'MsBrainAgingSpatialDonor_6_1', 'MsBrainAgingSpatialDonor_6_2', 'MsBrainAgingSpatialDonor_7_0', 'MsBrainAgingSpatialDonor_7_1', 'MsBrainAgingSpatialDonor_7_2', 'MsBrainAgingSpatialDonor_8_0', 'MsBrainAgingSpatialDonor_8_1', 'MsBrainAgingSpatialDonor_8_2', 'MsBrainAgingSpatialDonor_9_1', 'MsBrainAgingSpatialDonor_9_2'], # allen
#     None, # androvic
#     ['CNTRL_PEAK_B_R2', 'CNTRL_PEAK_B_R3', 'CNTRL_PEAK_B_R4', 'EAE_PEAK_B_R2', 'EAE_PEAK_B_R3', 'EAE_PEAK_B_R4'], # kukanja
#     ['Middle1', 'Old1', 'Old2', 'Young1', 'Young2'], # pilot
]

# test
test_ids = [
    ["11","33","57","93"], # aging coronal
    ['81'], # aging sagittal
    [], # exercise
    [], # reprogramming
#     ["MsBrainAgingSpatialDonor_13_0","MsBrainAgingSpatialDonor_9_0","MsBrainAgingSpatialDonor_14_0","MsBrainAgingSpatialDonor_1_0"], # allen
#     [], # androvic
#     ['CNTRL_PEAK_B_R1', 'EAE_PEAK_B_R1'], # kukanja
#     ["Middle2"], # pilot
]


if inject is False:
    inject_feature = None

# init test data
test_dataset = SpatialAgingCellDataset(subfolder_name="test",
                                       target="expression",
                                       k_hop=k_hop,
                                       augment_hop=augment_hop,
                                       node_feature=node_feature,
                                       inject_feature=inject_feature,
                                       num_cells_per_ct_id=100,
                                       center_celltypes=center_celltypes,
                                  use_ids=test_ids)

# init train data
train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                        target="expression",
                                        k_hop=k_hop,
                                        augment_hop=augment_hop,
                                        node_feature=node_feature,
                                        inject_feature=inject_feature,
                                        num_cells_per_ct_id=100,
                                        center_celltypes=center_celltypes,
                                use_ids=train_ids)

# concatenate datasets
all_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

# define data loaders
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
all_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# LOAD MODELS
model_dirname = loss+f"_{learning_rate:.0e}".replace("-","n")
save_dir = os.path.join("results/gnn",train_dataset.processed_dir.split("/")[-2],model_dirname)

# init model
if inject is True:
    model = GNN(hidden_channels=64,
                input_dim=int(train_dataset.get(0).x.shape[1]),
                output_dim=len(train_dataset.get(0).y), # added for multivariate targets
                inject_dim=int(train_dataset.get(0).inject.shape[1]), # added for injecting features into last layer (after pooling)
                method="GIN", pool="add", num_layers=k_hop)
else:
    model = GNN(hidden_channels=64,
            input_dim=int(train_dataset.get(0).x.shape[1]),
            output_dim=len(train_dataset.get(0).y), # added for multivariate targets
            method="GIN", pool="add", num_layers=k_hop)

# load model weights
model.load_state_dict(torch.load(os.path.join(save_dir, f"{use_model}.pth"),
                                map_location=torch.device('cpu')))

####################################################################################

# pick loader
if train_or_test == "train":
    loader = train_loader
elif train_or_test == "test":
    loader = test_loader
elif train_or_test == "all":
    loader = all_loader
else:
    raise Exception ("train_or_test not recognized!")

# save name
if ood_subfoldername.lower() == "none":
    savename = steering_approach+"_"+str(num_props)+"steps_"+train_or_test
else:
    savename = ood_subfoldername+"OOD_"+steering_approach+"_"+str(num_props)+"steps_"+train_or_test


### Load in OOD dataset

# this is really just to get the path to the data folder (will need to copy as subfolder into same directory as main dataset)
if ood_subfoldername.lower() != "none": # OOD steering
    target_dataset = SpatialAgingCellDataset(subfolder_name=ood_subfoldername,
                                            target="expression",
                                            k_hop=k_hop,
                                            augment_hop=augment_hop,
                                            node_feature=node_feature,
                                            inject_feature=inject_feature,
                                            num_cells_per_ct_id=100,
                                            center_celltypes=center_celltypes,
                                    use_ids=None)
#target_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)



# STEERING
model.eval()

for prop in props:

    perturb_props = []
    target_expressions = []
    target_predictions = []
    target_celltypes = []
    start_expressions_list = []
    perturb_expressions_list = []
    start_celltypes_list = []

    for data in loader:
    
        # random draw of graph from target graphs
        if ood_subfoldername.lower() != "none": # OOD steering
            random_target_idx = np.random.choice(np.arange(len(target_dataset)))

        # make predictions
        out = predict (model, data, inject)

        # get actual expression
        if data.y.shape != out.shape:
            actual = torch.reshape(data.y.float(), out.shape)
        else:
            actual = data.y.float()

        # get center cell types
        celltypes = get_center_celltypes(data)

        ### STEERING PERTURBATION (and appending target expressions)
        subset_same_celltype = False

        if steering_approach == "batch_steer_mean":
            #### shift cell type expression closer to mean of target
            if ood_subfoldername.lower() != "none":
                # use an external target (random draw from other dataset or list of data objects)
                pdata, target_celltype, target_expression, target_out = batch_steering_mean(data, actual, out, celltypes, target=target_dataset.get(random_target_idx), prop=prop)
                subset_same_celltype = True
            else:
                pdata, target_celltype, target_expression, target_out = batch_steering_mean(data, actual, out, celltypes, prop=prop)
                subset_same_celltype = True

        elif steering_approach == "batch_steer_cell":
            #### randomly draw cell from first graph and replace in other graphs
            if ood_subfoldername.lower() != "none":
                pdata, target_celltype, target_expression, target_out = batch_steering_cell(data, actual, out, celltypes, target=target_dataset.get(random_target_idx), prop=prop)
                subset_same_celltype = True
            else:
                pdata, target_celltype, target_expression, target_out = batch_steering_cell(data, actual, out, celltypes, prop=prop)
                subset_same_celltype = True

        else:
            raise Exception("steering_approach not recognized")

        # append target expression and prop
        target_expressions.append(target_expression)
        target_predictions.append(target_out)
        target_celltypes.append(target_celltype)
        start_celltypes_list.append(celltypes)
        perturb_props.append(round(prop,3))

        # get perturbed predicted
        pout = predict (model, pdata, inject)

        # temper perturbed expression
        perturbed = temper(actual, out, pout, method="distribution")

        start_expressions = []
        perturb_expressions = []
        for bi in np.unique(pdata.batch):
            # subset to only those that have same center cell type as first graph
            if subset_same_celltype is True:
                if (celltypes[bi] == target_celltype) and (bi>0):
                    start_expressions.append(actual[bi,:])
                    perturb_expressions.append(perturbed[bi,:])
            else:
                start_expressions.append(actual[bi,:])
                perturb_expressions.append(perturbed[bi,:])

        # append start expressions and perturb expressions
        start_expressions_list.append(start_expressions)
        perturb_expressions_list.append(perturb_expressions)
    
    print(f"Finished {round(prop,3)} proportion", flush=True)
    
    # save lists
    save_dict = {
        "perturb_props": perturb_props,
        "target_expressions": target_expressions,
        "target_predictions": target_predictions,
        "target_celltypes": target_celltypes,
        "start_expressions_list": start_expressions_list,
        "perturb_expressions_list": perturb_expressions_list,
        "start_celltypes_list": start_celltypes_list,
        }
    with open(os.path.join(save_dir, f"{savename}_{round(prop,3)*1000}.pkl"), 'wb') as f:
        pickle.dump(save_dict, f)
    
    
# Compute stats comparing to target (actual)
r_list_start = []
s_list_start = []
mae_list_start = []
r_list_perturb = []
s_list_perturb = []
mae_list_perturb = []
prop_list = []

for prop in props:

    # load in each saved file
    with open(os.path.join(save_dir, f"{savename}_{round(prop,3)*1000}.pkl"), 'rb') as f:
        save_dict = pickle.load(f)
    perturb_props = save_dict["perturb_props"]
    target_expressions = save_dict["target_expressions"]
    start_expressions_list = save_dict["start_expressions_list"]
    perturb_expressions_list = save_dict["perturb_expressions_list"]
    target_celltypes = save_dict["target_celltypes"]
    start_celltypes_list = save_dict["start_celltypes_list"]
    
    # compute stats
    for i in range(len(target_expressions)):
        target = target_expressions[i].detach().numpy()
        
        # mask out missing values to compute stats
        missing_mask = target != -1
        
        for start in start_expressions_list[i]:
            # compute stats for start
            start = start.detach().numpy()
            r_list_start.append(pearsonr(start[missing_mask], target[missing_mask])[0])
            s_list_start.append(spearmanr(start[missing_mask], target[missing_mask])[0])
            mae_list_start.append(np.mean(np.abs(start[missing_mask]-target[missing_mask])))
        
        for perturb in perturb_expressions_list[i]:
            # compute stats for perturb
            perturb = perturb.detach().numpy()
            r_list_perturb.append(pearsonr(perturb[missing_mask], target[missing_mask])[0])
            s_list_perturb.append(spearmanr(perturb[missing_mask], target[missing_mask])[0])
            mae_list_perturb.append(np.mean(np.abs(perturb[missing_mask]-target[missing_mask])))
            
            prop_list.append(perturb_props[i])

stats_df = pd.DataFrame(np.vstack((r_list_start+r_list_perturb,
                                   s_list_start+s_list_perturb,
                                   mae_list_start+mae_list_perturb,
                                   prop_list+prop_list,
                                   ["Start"]*len(r_list_start)+["Perturbed"]*len(r_list_perturb))).T,
                        columns=["Pearson", "Spearman", "MAE", "Prop", "Type"])
for col in ["Pearson", "Spearman", "MAE"]:
    stats_df[col] = stats_df[col].astype(float)

# save stats
stats_df.to_csv(os.path.join(save_dir, f"{savename}_actualtarget.csv"))




# Compute stats comparing to target (predicted) -- predictions only made if not OOD so only do if not OOD
if ood_subfoldername.lower() == "none":

    r_list_start = []
    s_list_start = []
    mae_list_start = []
    r_list_perturb = []
    s_list_perturb = []
    mae_list_perturb = []
    prop_list = []

    for prop in props:

        # load in each saved file
        with open(os.path.join(save_dir, f"{savename}_{round(prop,3)*1000}.pkl"), 'rb') as f:
            save_dict = pickle.load(f)
        perturb_props = save_dict["perturb_props"]
        target_predictions = save_dict["target_predictions"]
        start_expressions_list = save_dict["start_expressions_list"]
        perturb_expressions_list = save_dict["perturb_expressions_list"]
        target_celltypes = save_dict["target_celltypes"]
        start_celltypes_list = save_dict["start_celltypes_list"]
        
        # compute stats
        for i in range(len(target_predictions)):
            try:
                target = target_predictions[i].detach().numpy()
            except:
                target = target_predictions[i]
            
            # mask out missing values to compute stats
            missing_mask = target != -1
            
            for start in start_expressions_list[i]:
                # compute stats for start
                start = start.detach().numpy()
                r_list_start.append(pearsonr(start[missing_mask], target[missing_mask])[0])
                s_list_start.append(spearmanr(start[missing_mask], target[missing_mask])[0])
                mae_list_start.append(np.mean(np.abs(start[missing_mask]-target[missing_mask])))
            
            for perturb in perturb_expressions_list[i]:
                # compute stats for perturb
                perturb = perturb.detach().numpy()
                r_list_perturb.append(pearsonr(perturb[missing_mask], target[missing_mask])[0])
                s_list_perturb.append(spearmanr(perturb[missing_mask], target[missing_mask])[0])
                mae_list_perturb.append(np.mean(np.abs(perturb[missing_mask]-target[missing_mask])))
                
                prop_list.append(perturb_props[i])

    stats_df = pd.DataFrame(np.vstack((r_list_start+r_list_perturb,
                                       s_list_start+s_list_perturb,
                                       mae_list_start+mae_list_perturb,
                                       prop_list+prop_list,
                                       ["Start"]*len(r_list_start)+["Perturbed"]*len(r_list_perturb))).T,
                            columns=["Pearson", "Spearman", "MAE", "Prop", "Type"])
    for col in ["Pearson", "Spearman", "MAE"]:
        stats_df[col] = stats_df[col].astype(float)

    # save stats
    stats_df.to_csv(os.path.join(save_dir, f"{savename}_predictedtarget.csv"))