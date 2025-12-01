import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import argparse
import torch

from spatial_gnn.models.gnn_model import GNN
from spatial_gnn.utils.perturbation_utils import predict, temper, get_center_celltypes
from spatial_gnn.utils.dataset_utils import load_dataset_config, create_dataloader_from_dataset, load_model_from_path
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.models.baselines import global_mean_baseline_batch, khop_mean_baseline_batch
from spatial_gnn.utils.perturbation_utils import perturb_by_multiplier


def main():
    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to use (aging_coronal, aging_sagittal, exercise, reprogramming, allen, kukanja, pilot)", type=str, required=True)
    parser.add_argument("--exp_name", help="experiment name", type=str, required=True)
    parser.add_argument("--base_path", help="Base path to the data directory", type=str, required=True)
    parser.add_argument("--k_hop", help="k-hop neighborhood size", type=int, required=True)
    parser.add_argument("--augment_hop", help="number of hops to take for graph augmentation", type=int, required=True)
    parser.add_argument("--center_celltypes", help="cell type labels to center graphs on, separated by comma. Use 'all' for all cell types or 'none' for no cell type filtering", type=str, required=True)
    parser.add_argument("--node_feature", help="node features key, e.g. 'celltype_age_region'", type=str, required=True)
    parser.add_argument("--inject_feature", help="inject features key, e.g. 'center_celltype'", type=str, required=True)
    parser.add_argument("--model_type", help="model type to use", type=str, required=True)
    parser.add_argument("--model_path", help="Path to model to use", type=str, required=True)
    parser.add_argument("--debug", help="debug mode", action="store_true")
    
    # paired terms-specific arguments
    parser.add_argument("--pairs_path", help="path to GO pairs directory", type=str)
    parser.add_argument("--perturb_approach", help="perturbation method to use", type=str)
    parser.add_argument("--num_props", help="number of intervals from 0 to 1", type=int)

    args = parser.parse_args()
    
    pairs_path = args.pairs_path
    perturb_approach = args.perturb_approach
    num_props = args.num_props
    model_type = args.model_type

    # Load dataset configurations
    DATASET_CONFIGS = load_dataset_config()
    
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
        
    node_feature = args.node_feature
    inject_feature = args.inject_feature

    if inject_feature.lower() == "none":
        inject_feature = None
        inject=False
    else:
        inject=True

    # determine gpu / cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    # Build cell type index
    celltypes_to_index = {}
    for ci, cellt in enumerate(dataset_config["celltypes"]):
        celltypes_to_index[cellt] = ci

    train_dataset = SpatialAgingCellDataset(
        subfolder_name="train",
        dataset_prefix=args.dataset,
        target="expression",
        k_hop=k_hop,
        augment_hop=augment_hop,
        node_feature=node_feature,
        inject_feature=inject_feature,
        num_cells_per_ct_id=100,
        center_celltypes="all",
        use_ids=train_ids,
        raw_filepaths=[file_path],
        celltypes_to_index=celltypes_to_index,
        normalize_total=True,
        debug=args.debug,
        overwrite=False,
        use_mp=False,
    )

    test_dataset = SpatialAgingCellDataset(
        subfolder_name="test",
        dataset_prefix=args.dataset,
        target="expression",
        k_hop=k_hop,
        augment_hop=augment_hop,
        node_feature=node_feature,
        inject_feature=inject_feature,
        num_cells_per_ct_id=100,
        center_celltypes="all",
        use_ids=test_ids,
        raw_filepaths=[file_path],
        celltypes_to_index=celltypes_to_index,
        normalize_total=True,
        debug=args.debug,
        overwrite=False,
        use_mp=False,
    )

    train_dataset.process()
    print("Finished processing train dataset", flush=True)
    test_dataset.process()
    print("Finished processing test dataset", flush=True)

    _, train_loader = create_dataloader_from_dataset(
        dataset=train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        debug=args.debug,
    )
    _, test_loader = create_dataloader_from_dataset(
        dataset=test_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        debug=args.debug,
    )
    save_dir = os.path.join("results", "go_causal_interaction_final", args.exp_name, model_type)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}", flush=True)

    model, config = load_model_from_path(args.model_path, device)
    gene_names = np.char.upper(train_dataset.gene_names.astype(str))
    go_causal_interaction(model, train_loader, test_loader, save_dir, model_type, perturb_approach, num_props, pairs_path, gene_names, debug=args.debug, device=device)

def go_causal_interaction(
    model,
    train_loader,
    test_loader,
    save_dir,
    model_type,
    perturb_approach,
    num_props,
    pairs_path,
    gene_names,
    debug=False,
    device="cuda",
):
    # get savename
    savename = f"{perturb_approach}_{num_props}steps_{model_type}"

    if model_type == "global_mean":
        global_mean = global_mean_baseline_batch(train_loader)
    
    print("Running GO interactions...", flush=True)    
    terms_list = np.unique([("_").join(x.split(".")[0].split("_")[1:]) for x in os.listdir(pairs_path)])
    model.eval()

    if debug:
        terms_list = terms_list[:2]
    
    for term in tqdm(terms_list, total=len(terms_list)):

        # read in production and response gene lists (convert to uppercase for consistency)
        production_genes = np.char.upper(pd.read_csv(os.path.join(pairs_path,f"production_{term}.csv"), header=None).values.flatten().astype(str))
        response_genes = np.char.upper(pd.read_csv(os.path.join(pairs_path,f"response_{term}.csv"), header=None).values.flatten().astype(str))
        
        # get overlapping genes
        production_genes_overlap = np.intersect1d(production_genes, gene_names)
        response_genes_overlap = np.intersect1d(response_genes, gene_names)
        
        # skip term if not enough genes
        if (len(production_genes_overlap)==0) or (len(response_genes_overlap)==0):
            continue
            
        print(term, flush=True)
        print(f"num prod genes: {len(production_genes_overlap)}", flush=True)
        print(f"num resp genes: {len(response_genes_overlap)}", flush=True)
            
        # get gene indices
        production_indices = np.where(np.isin(gene_names, production_genes_overlap))[0]
        response_indices = np.where(np.isin(gene_names, response_genes_overlap))[0]
        
        # run perturbations
        props = np.linspace(0,1,round(num_props))
        for prop in np.power(10,props): # powers of 10

            perturb_props = []
            start_forwards_list = [] # for forward perturbation (perturb production, track response)
            perturb_forwards_list = [] # for forward perturbation (perturb production, track response)
            start_reverses_list = [] # for reverse perturbation (perturb response, track production)
            perturb_reverses_list = [] # for reverse perturbation (perturb response, track production)
            center_celltypes_list = [] # cell type of the center cell (measurement)

            for data in test_loader:
                data = data.to(device)
                out = predict(model, data, inject=False)

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
                perturb_props.append(round(prop,3))

                if model_type == "model":
                    fout = predict(model, fdata.to(device), inject=False)
                    rout = predict(model, rdata.to(device), inject=False)
                    fperturbed = temper(actual, out, fout, method="distribution_renormalize") # distribution_renormalize
                    rperturbed = temper(actual, out, rout, method="distribution_renormalize") # distribution_renormalize
                elif model_type == "global_mean":
                    batch_size = len(fdata.batch)
                    fperturbed = global_mean.unsqueeze(0).repeat(batch_size, 1)
                    rperturbed = global_mean.unsqueeze(0).repeat(batch_size, 1)
                elif model_type == "khop_mean":
                    fperturbed = khop_mean_baseline_batch(fdata)
                    rperturbed = khop_mean_baseline_batch(rdata)

                start_forwards = []
                perturb_forwards = []
                start_reverses = []
                perturb_reverses = []
                celltypes_subbed = []
                
                # collect the center cell start and perturbed expression pairs
                for bi in np.unique(fdata.batch.cpu()):
                    # drop any graphs with starting zero expression for production/response
                    if (torch.sum(actual[bi,response_indices]) != 0) and (torch.sum(actual[bi,production_indices]) != 0):
                        # measure forward perturb w/ response genes
                        start_forwards.append(torch.sum(actual[bi,response_indices].cpu()).detach().numpy())
                        perturb_forwards.append(torch.sum(fperturbed[bi,response_indices].cpu()).detach().numpy())
                        # measure reverse perturb w/ production genes
                        start_reverses.append(torch.sum(actual[bi,production_indices].cpu()).detach().numpy())
                        perturb_reverses.append(torch.sum(rperturbed[bi,production_indices].cpu()).detach().numpy())
                        # also filter celltypes
                        celltypes_subbed.append(celltypes[bi])

                # append results and metadata
                start_forwards_list.append(np.array(start_forwards).flatten())
                perturb_forwards_list.append(np.array(perturb_forwards).flatten())
                start_reverses_list.append(np.array(start_reverses).flatten())
                perturb_reverses_list.append(np.array(perturb_reverses).flatten())
                center_celltypes_list.append(np.array(celltypes_subbed).flatten())
            
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
        
if __name__ == "__main__":
    main()
