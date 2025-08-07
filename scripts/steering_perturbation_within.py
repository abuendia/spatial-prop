import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.stats import pearsonr, spearmanr, ttest_ind
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set_style("ticks")
from sklearn.neighbors import BallTree
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from decimal import Decimal
import copy
import json 
import random
import networkx as nx
import argparse
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score

from aging_gnn_model import *
from perturbation import *

### Load in dataset configs

def load_dataset_config():
    """Load dataset configurations from JSON file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'datasets.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in dataset configuration file at {config_path}")

def main():
    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to use (aging_coronal, aging_sagittal, exercise, reprogramming, allen, kukanja, pilot)", type=str, required=True)
    parser.add_argument("--base_path", help="Base path to the data directory", type=str, required=True)
    parser.add_argument("--k_hop", help="k-hop neighborhood size", type=int, required=True)
    parser.add_argument("--augment_hop", help="number of hops to take for graph augmentation", type=int, required=True)
    parser.add_argument("--center_celltypes", help="cell type labels to center graphs on, separated by comma. Use 'all' for all cell types or 'none' for no cell type filtering", type=str, required=True)
    parser.add_argument("--node_feature", help="node features key, e.g. 'celltype_age_region'", type=str, required=True)
    parser.add_argument("--inject_feature", help="inject features key, e.g. 'center_celltype'", type=str, required=True)
    parser.add_argument("--learning_rate", help="learning rate", type=float, required=True)
    parser.add_argument("--loss", help="loss: balanced_mse, npcc, mse, l1", type=str, required=True)
    parser.add_argument("--epochs", help="number of epochs", type=int, required=True)
    parser.add_argument("--gene_list", help="Path to file containing list of genes to use (optional)", type=str, default=None)
    
    # steering-specific arguments
    parser.add_argument("--steering_approach", help="steering method to use", type=str)
    parser.add_argument("--num_props", help="number of intervals from 0 to 1", type=int)
    parser.add_argument("--train_or_test", help="train or test dataset", type=str)
    
    args = parser.parse_args()
    
    steering_approach = args.steering_approach
    num_props = args.num_props
    props = np.linspace(0,1,num_props+1) # proportions to scale
    train_or_test = args.train_or_test

    # Load dataset configurations
    DATASET_CONFIGS = load_dataset_config()
    
    # set which model to use
    use_model = "model" # "best_model"

    # Validate dataset choice
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset must be one of: {', '.join(DATASET_CONFIGS.keys())}")
    print(f"\n {args.dataset}", flush=True)
    
    # load parameters from arguments
    dataset_config = DATASET_CONFIGS[args.dataset]
    train_ids = dataset_config['train_ids']
    test_ids = dataset_config['test_ids']
    file_path = os.path.join(args.base_path, dataset_config['file_name'])
    k_hop = args.k_hop
    augment_hop = args.augment_hop
    
    # Handle center_celltypes
    if args.center_celltypes.lower() == 'none':
        center_celltypes = None
    elif args.center_celltypes.lower() == 'all':
        center_celltypes = 'all'
    else:
        center_celltypes = args.center_celltypes.split(",")
    
    node_feature = args.node_feature
    inject_feature = args.inject_feature
    learning_rate = args.learning_rate
    loss = args.loss
    epochs = args.epochs

    if inject_feature.lower() == "none":
        inject_feature = None
        inject=False
    else:
        inject=True

    # determine gpu / cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    # train
    exp_name = f"{k_hop}hop_{augment_hop}augment_{node_feature}_{inject_feature}_{learning_rate:.0e}lr_{loss}_{epochs}epochs"

    # Load gene list if provided
    gene_list = None
    if args.gene_list is not None:
        try:
            with open(args.gene_list, 'r') as f:
                gene_list = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Gene list file not found at {args.gene_list}")

    # Build cell type index
    celltypes_to_index = {}
    for ci, cellt in enumerate(dataset_config["celltypes"]):
        celltypes_to_index[cellt] = ci
    
    
    # init dataset with settings
    train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                            dataset_prefix=args.dataset,
                                            target="expression",
                                            k_hop=k_hop,
                                            augment_hop=augment_hop,
                                            node_feature=node_feature,
                                            inject_feature=inject_feature,
                                            num_cells_per_ct_id=100,
                                            center_celltypes=center_celltypes,
                                            use_ids=train_ids,
                                            raw_filepaths=[file_path],
                                            gene_list=gene_list,
                                            celltypes_to_index=celltypes_to_index)

    test_dataset = SpatialAgingCellDataset(subfolder_name="test",
                                        dataset_prefix=args.dataset,
                                        target="expression",
                                        k_hop=k_hop,
                                        augment_hop=augment_hop,
                                        node_feature=node_feature,
                                        inject_feature=inject_feature,
                                        num_cells_per_ct_id=100,
                                        center_celltypes=center_celltypes,
                                        use_ids=test_ids,
                                        raw_filepaths=[file_path],
                                        gene_list=gene_list,
                                        celltypes_to_index=celltypes_to_index)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)

    print(len(train_dataset), flush=True)
    print(len(test_dataset), flush=True)


    # init GNN model
    if inject is True:
        model = GNN(hidden_channels=64,
                    input_dim=int(train_dataset.get(0).x.shape[1]),
                    output_dim=len(train_dataset.get(0).y), # added for multivariate targets
                    inject_dim=int(train_dataset.get(0).inject.shape[1]), # added for injecting features into last layer (after pooling),
                    method="GIN", pool="add", num_layers=k_hop)
    else:
        model = GNN(hidden_channels=64,
                    input_dim=int(train_dataset.get(0).x.shape[1]),
                    output_dim=len(train_dataset.get(0).y), # added for multivariate targets
                    method="GIN", pool="add", num_layers=k_hop)
    model.to(device)
    print(device, flush=True)

    # create directory to save results
    model_dirname = loss+f"_{learning_rate:.0e}".replace("-","n")
    save_dir = os.path.join("results/gnn",train_dataset.processed_dir.split("/")[-2],model_dirname,"steer_within")
    os.makedirs(save_dir, exist_ok=True)
    
    ### STEERING EXPERIMENTS ###
    
    # pick loader
    if train_or_test == "train":
        loader = train_loader
    elif train_or_test == "test":
        loader = test_loader
    elif train_or_test == "all":
        loader = all_loader
    else:
        raise Exception ("train_or_test not recognized!")
        
    # get savename
    savename = steering_approach+"_"+str(num_props)+"steps_"+train_or_test
    
    # STEERING runs
    
    ood_subfoldername = "none" # set to this for the within runs
    
    model.eval()
    
    print("Running steering...", flush=True)
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
            perturbed = temper(actual, out, pout, method="distribution_renormalize")

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
    
    # make plots
    stats_df = pd.read_csv(os.path.join(save_dir, f"{savename}_actualtarget.csv"))

    for value in ["Pearson", "Spearman", "MAE"]:
        
        # plot densities
        fig, ax = plt.subplots(figsize=(6,4))
        sns.kdeplot(stats_df[stats_df["Type"]=="Perturbed"], x=value, hue="Prop", ax=ax)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.7))
        #plt.title(save_dir.split("/")[-3], fontsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        for ax in plt.gcf().axes:
            l = ax.get_xlabel()
            ax.set_xlabel(l, fontsize=16)
            l = ax.get_ylabel()
            ax.set_ylabel(l, fontsize=16)
        ax.get_legend().set_title("Steering")
        plt.setp(ax.get_legend().get_texts(), fontsize='14')
        plt.setp(ax.get_legend().get_title(), fontsize='16')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{savename}_actualtarget_{value}_density.pdf"), bbox_inches='tight')
        plt.close()
        
        # plot line with confidence interval
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(stats_df[stats_df["Type"]=="Perturbed"], x="Prop", y=value,
                     ci=95, color='k', ax=ax)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for ax in plt.gcf().axes:
            l = ax.get_xlabel()
            #ax.set_xlabel(l, fontsize=16)
            ax.set_xlabel("Steering", fontsize=16)
            l = ax.get_ylabel()
            ax.set_ylabel(l, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{savename}_actualtarget_{value}.pdf"), bbox_inches='tight')
        plt.close()
    
    print("Finished actualtarget steering...", flush=True)




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
        print("Finished predictedtarget steering...", flush=True)
        
        # make plots
        stats_df = pd.read_csv(os.path.join(save_dir, f"{savename}_predictedtarget.csv"))

        for value in ["Pearson", "Spearman", "MAE"]:
            
            # plot densities
            fig, ax = plt.subplots(figsize=(6,4))
            sns.kdeplot(stats_df[stats_df["Type"]=="Perturbed"], x=value, hue="Prop", ax=ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.7))
            #plt.title(save_dir.split("/")[-3], fontsize=14)
            plt.xticks(rotation=30, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            for ax in plt.gcf().axes:
                l = ax.get_xlabel()
                ax.set_xlabel(l, fontsize=16)
                l = ax.get_ylabel()
                ax.set_ylabel(l, fontsize=16)
            ax.get_legend().set_title("Steering")
            plt.setp(ax.get_legend().get_texts(), fontsize='14')
            plt.setp(ax.get_legend().get_title(), fontsize='16')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{savename}_predictedtarget_{value}_density.pdf"), bbox_inches='tight')
            plt.close()
            
            # plot line with confidence interval
            fig, ax = plt.subplots(figsize=(6,4))
            sns.lineplot(stats_df[stats_df["Type"]=="Perturbed"], x="Prop", y=value,
                         ci=95, color='k', ax=ax)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            for ax in plt.gcf().axes:
                l = ax.get_xlabel()
                #ax.set_xlabel(l, fontsize=16)
                ax.set_xlabel("Steering", fontsize=16)
                l = ax.get_ylabel()
                ax.set_ylabel(l, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{savename}_predictedtarget_{value}.pdf"), bbox_inches='tight')
            plt.close()
    
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()