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
    
    # paired terms-specific arguments
    parser.add_argument("--pairs_path", help="path to GO pairs directory", type=str)
    parser.add_argument("--perturb_approach", help="perturbation method to use", type=str)
    parser.add_argument("--num_props", help="number of intervals from 0 to 1", type=int)
    parser.add_argument("--train_or_test", help="train or test dataset", type=str)
    
    args = parser.parse_args()
    
    pairs_path = args.pairs_path
    perturb_approach = args.perturb_approach
    num_props = args.num_props
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
    save_dir = os.path.join("results/gnn",train_dataset.processed_dir.split("/")[-2],model_dirname,"GO_ITXN")
    os.makedirs(save_dir, exist_ok=True)
    
    ### GO INTERACTION EXPERIMENTS ###
    
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
    savename = perturb_approach+"_"+str(num_props)+"steps_"+train_or_test
    
    # get genes
    gene_names = np.char.lower(train_dataset.gene_names.astype(str))
    
    # Interaction Runs
        
    model.eval()
    
    print("Running GO interactions...", flush=True)
    
    # extract terms from saved interactions
    terms_list = np.unique([x.split(".")[0].split("_")[-1] for x in os.listdir(pairs_path)])
    
    for term in terms_list:
    
        # read in production and response gene lists
        production_genes = np.char.lower(pd.read_csv(os.path.join(pairs_path,f"production_{term}.csv"), header=None).values.flatten().astype(str))
        response_genes = np.char.lower(pd.read_csv(os.path.join(pairs_path,f"response_{term}.csv"), header=None).values.flatten().astype(str))
        
        # get overlapping genes
        production_genes_overlap = np.intersect1d(production_genes, gene_names)
        response_genes_overlap = np.intersect1d(response_genes, gene_names)
        
        # skip term if not enough genes
        if (len(production_genes_overlap)==0) or (len(response_genes_overlap)==0):
            continue
            
        # get gene indices
        production_indices = np.where(np.isin(gene_names, production_genes_overlap))[0]
        response_indices = np.where(np.isin(gene_names, response_genes_overlap))[0]
        
        # run perturbations
        oneside_props = np.linspace(0,1,round((num_props+1)/2))
        props = np.unique(np.concatenate((-oneside_props,oneside_props)))
        
        for prop in np.power(10,props): # powers of 10

            perturb_props = []
            start_forwards_list = [] # for forward perturbation (perturb production, track response)
            perturb_forwards_list = [] # for forward perturbation (perturb production, track response)
            start_reverses_list = [] # for reverse perturbation (perturb response, track production)
            perturb_reverses_list = [] # for reverse perturbation (perturb response, track production)
            center_celltypes_list = [] # cell type of the center cell (measurement)

            for data in loader:

                # make predictions
                out = predict (model, data, inject)

                # get actual expression
                if data.y.shape != out.shape:
                    actual = torch.reshape(data.y.float(), out.shape)
                else:
                    actual = data.y.float()

                # get center cell types
                celltypes = get_center_celltypes(data)

                ### PERTURBATION
                if perturb_approach == "multiplier":
                    # multiply expression by prop
                    fdata = perturb_by_multiplier(data, production_indices, prop=prop) # forward perturb production genes
                    rdata = perturb_by_multiplier(data, response_indices, prop=prop) # reverse perturb response genes
                else:
                    raise Exception("perturb_approach not recognized")

                # append target expression and prop
                center_celltypes_list.append(celltypes)
                perturb_props.append(round(prop,3))

                # get perturbed predicted
                fout = predict (model, fdata, inject)
                rout = predict (model, rdata, inject)

                # temper perturbed expression
                fperturbed = temper(actual, out, fout, method="distribution")
                rperturbed = temper(actual, out, rout, method="distribution")
                
                start_forwards = []
                perturb_forwards = []
                start_reverses = []
                perturb_reverses = []
                
                for bi in np.unique(fdata.batch):
                    # measure forward perturb w/ response genes
                    start_forwards.append(torch.sum(actual[bi,response_indices]).detach().numpy())
                    perturb_forwards.append(torch.sum(fperturbed[bi,response_indices]).detach().numpy())
                    # measure reverse perturb w/ production genes
                    start_reverses.append(torch.sum(actual[bi,production_indices]).detach().numpy())
                    perturb_reverses.append(torch.sum(rperturbed[bi,production_indices]).detach().numpy())

                # append results and metadata
                start_forwards_list.append(np.array(start_forwards).flatten())
                perturb_forwards_list.append(np.array(perturb_forwards).flatten())
                start_reverses_list.append(np.array(start_reverses).flatten())
                perturb_reverses_list.append(np.array(perturb_reverses).flatten())
            
            print(f"Finished {round(prop,3)} proportion", flush=True)
            
            # save lists
            save_dict = {
                "perturb_props": perturb_props,
                "start_forwards_list": start_forwards_list,
                "perturb_forwards_list": perturb_forwards_list,
                "start_reverses_list": start_reverses_list,
                "perturb_reverses_list": perturb_reverses_list,
                "center_celltypes_list": center_celltypes_list,
                }
            os.makedirs(os.path.join(save_dir,term), exist_ok=True)
            with open(os.path.join(save_dir,term,f"{savename}_{round(prop*1000)}.pkl"), 'wb') as f:
                pickle.dump(save_dict, f)
        
    

        # Compute and plot results
        norm_forward_col = []
        norm_reverse_col = []
        celltype_col = []
        prop_col = []

        for prop in np.power(10,props):

            # load in each saved file
            with open(os.path.join(save_dir,term,f"{savename}_{round(prop*1000)}.pkl"), 'rb') as f:
                save_dict = pickle.load(f)
            perturb_props = save_dict["perturb_props"]
            start_forwards_list = save_dict["start_forwards_list"]
            perturb_forwards_list = save_dict["perturb_forwards_list"]
            start_reverses_list = save_dict["start_reverses_list"]
            perturb_reverses_list = save_dict["perturb_reverses_list"]
            center_celltypes_list = save_dict["center_celltypes_list"]
            
            # normalized response = perturb/start
            norm_forwards = np.concatenate(perturb_forwards_list) / np.concatenate(start_forwards_list)
            norm_reverses = np.concatenate(perturb_reverses_list) / np.concatenate(start_reverses_list)
            
            # get results and format
            expanded_props = np.concatenate([np.array([perturb_props[di]]*len(start_forwards_list[di])) for di in range(len(perturb_props))])
            
            norm_forward_col = np.concatenate((norm_forward_col, norm_forwards))
            norm_reverse_col = np.concatenate((norm_reverse_col, norm_reverses))
            prop_col = np.concatenate((prop_col, expanded_props))
            celltype_col = np.concatenate((celltype_col, np.concatenate(center_celltypes_list)))

        stats_df = pd.DataFrame(np.vstack((prop_col,norm_forward_col,norm_reverse_col,celltype_col)).T,
                                columns=["Prop", "Normalized Forward", "Normalized Reverse", "Celltype"])
        for col in ["Prop", "Normalized Forward", "Normalized Reverse"]:
            stats_df[col] = stats_df[col].astype(float)

        # save stats
        stats_df.to_csv(os.path.join(save_dir,term,f"{savename}_results.csv"))
        

        # make plots
        stats_df = pd.read_csv(os.path.join(save_dir,term,f"{savename}_results.csv"))
        
        ### FORWARD
        # plot lines with confidence interval
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(stats_df, x="Prop", y="Normalized Forward", hue="Celltype",
                     ci=95, ax=ax)
        plt.title(term, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for ax in plt.gcf().axes:
            l = ax.get_xlabel()
            ax.set_xlabel("Production Signature Perturbation", fontsize=16)
            ax.set_ylabel("Normalized Forward Signature", fontsize=16)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,term,f"{savename}_forward_celltype.pdf"), bbox_inches='tight')
        plt.close()
        
        # combined
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(stats_df, x="Prop", y="Normalized Forward",
                     ci=95, color='k', ax=ax)
        plt.title(term, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for ax in plt.gcf().axes:
            l = ax.get_xlabel()
            ax.set_xlabel("Production Signature Perturbation", fontsize=16)
            ax.set_ylabel("Normalized Forward Signature", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,term,f"{savename}_forward_combined.pdf"), bbox_inches='tight')
        plt.close()
        
        
        ### REVERSE
        # plot lines with confidence interval
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(stats_df, x="Prop", y="Normalized Reverse", hue="Celltype",
                     ci=95, ax=ax)
        plt.title(term, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for ax in plt.gcf().axes:
            l = ax.get_xlabel()
            ax.set_xlabel("Response Signature Perturbation", fontsize=16)
            ax.set_ylabel("Normalized Reverse Signature", fontsize=16)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,term,f"{savename}_reverse_celltype.pdf"), bbox_inches='tight')
        plt.close()
        
        # combined
        fig, ax = plt.subplots(figsize=(6,4))
        sns.lineplot(stats_df, x="Prop", y="Normalized Reverse",
                     ci=95, color='k', ax=ax)
        plt.title(term, fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for ax in plt.gcf().axes:
            l = ax.get_xlabel()
            ax.set_xlabel("Response Signature Perturbation", fontsize=16)
            ax.set_ylabel("Normalized Reverse Signature", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,term,f"{savename}_reverse_combined.pdf"), bbox_inches='tight')
        plt.close()

    print("DONE.", flush=True)

if __name__ == "__main__":
    main()