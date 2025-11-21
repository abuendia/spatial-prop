import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import pickle
import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from spatial_gnn.models.gnn_model import GNN, CellTypeGNN
from spatial_gnn.utils.perturbation_utils import batch_steering_mean, batch_steering_cell
from spatial_gnn.utils.dataset_utils import load_dataset_config
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.utils.perturbation_utils import predict, temper, get_center_celltypes
from spatial_gnn.models.mean_baselines import (
    global_mean_baseline_batch,
    khop_mean_baseline_batch,
)


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
    parser.add_argument("--exp_name", help="experiment name", type=str, required=True)
    parser.add_argument("--debug", help="debug mode", action="store_true")
    
    # steering-specific arguments
    parser.add_argument("--steering_approach", help="steering method to use", type=str)
    parser.add_argument("--num_props", help="number of intervals from 0 to 1", type=int)
    parser.add_argument("--model_type", help="model type to use", type=str, default=None)
    
    args = parser.parse_args()
    steering_approach = args.steering_approach
    num_props = args.num_props
    props = np.linspace(0,1,num_props+1) # proportions to scale
    model_type = args.model_type

    DATASET_CONFIGS = load_dataset_config()
    
    # Validate dataset choice
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset must be one of: {', '.join(DATASET_CONFIGS.keys())}")
    print(f"\n {args.dataset}", flush=True)
    
    temper_methods = ["renormalize", "distribution_renormalize"]

    # load parameters from arguments
    dataset_config = DATASET_CONFIGS[args.dataset]
    train_ids = dataset_config['train_ids']
    test_ids = dataset_config['test_ids']
    file_path = os.path.join(args.base_path, dataset_config['file_name'])
    k_hop = args.k_hop
    augment_hop = args.augment_hop
    
    # Handle center_celltypes
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

    all_train_data, all_test_data = [], []
    for idx, f in tqdm(enumerate(train_dataset.processed_file_names), total=len(train_dataset.processed_file_names)):
        if args.debug and idx > 2:
            break
        batch_list = torch.load(os.path.join(train_dataset.processed_dir, f), weights_only=False)
        all_train_data.extend(batch_list)
    for idx, f in tqdm(enumerate(test_dataset.processed_file_names), total=len(test_dataset.processed_file_names)):
        if args.debug and idx > 2:
            break
        batch_list = torch.load(os.path.join(test_dataset.processed_dir, f), weights_only=False)
        all_test_data.extend(batch_list)
    
    train_loader = DataLoader(all_train_data, batch_size=512, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)
    test_loader = DataLoader(all_test_data, batch_size=512, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=None, persistent_workers=False)
    save_dir = os.path.join("steer_within", test_dataset.processed_dir.split("/")[-2], args.exp_name, model_type)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}", flush=True)

    model = GNN(
        hidden_channels=64,
        input_dim=int(train_dataset.get(0).x.shape[1]),
        output_dim=len(train_dataset.get(0).y),
        inject_dim=int(train_dataset.get(0).inject.shape[1]) if inject is True else 0,
        method="GIN", 
        pool="center", 
        num_layers=k_hop,
        celltypes_to_index=celltypes_to_index,
        predict_celltype=False,
        train_multitask=False,
        ablate_gene_expression=False,
        use_one_hot_ct=False,
    )
    print(f"Model initialized on {device}")

    model_save_dir = f"/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/expression_only_khop2_no_genept_softmax_ct_center_pool/{args.dataset}_expression_2hop_2augment_expression_none/weightedl1_1en04/model.pth"
    model.load_state_dict(torch.load(model_save_dir), strict=False)
    model.to(device)

    eval_model(model, train_loader, test_loader, save_dir, model_type, steering_approach, num_props, props, temper_methods, device)
    

def eval_model(
    model,
    train_loader,
    test_loader, 
    save_dir, 
    model_type,
    steering_approach,
    num_props,
    props,
    temper_methods,
    device="cuda",
):
    savename = f"{steering_approach}_{num_props}steps_{model_type}"

    if model_type == "global_mean":
        global_mean = global_mean_baseline_batch(train_loader)

    for temper_method in temper_methods:

        model.eval()
        results_actual = []     
        results_pred = []     

        print("Running steering...", flush=True)
        for prop in props:

            print(f"Prop={prop:.3f}", flush=True)

            for data in tqdm(test_loader):
                data = data.to(device)
                out = predict(model, data, inject=False)

                # actual expression
                if data.y.shape != out.shape:
                    actual = data.y.float().reshape_as(out)
                else:
                    actual = data.y.float()

                center_celltypes = get_center_celltypes(data)

                if steering_approach == "batch_steer_mean":
                    pdata, target_celltype, target_expression, target_out = batch_steering_mean(
                        data.cpu(), actual.cpu(), out.cpu(), center_celltypes, prop=prop
                    )
                elif steering_approach == "batch_steer_cell":
                    pdata, target_celltype, target_expression, target_out = batch_steering_cell(
                        data, actual, out, center_celltypes, prop=prop
                    )

                # make predictions for perturbed data
                if model_type == "global_mean":
                    batch_n = data.center_node.shape[0]
                    perturbed = global_mean.unsqueeze(0).repeat(batch_n, 1)
                elif model_type == "khop_mean":
                    perturbed = khop_mean_baseline_batch(pdata)
                elif model_type == "model":
                    pout = predict(model, pdata.to(device), inject=False)
                    perturbed = temper(actual, out, pout, method=temper_method)

                unique_batches = torch.unique(pdata.batch.cpu())

                for bi in unique_batches:
                    bi_int = int(bi)
                    # only evaluate graphs eligible for steering
                    if (center_celltypes[bi_int] == target_celltype) and (bi_int > 0):

                        start_vec = actual[bi_int].detach().cpu().numpy()
                        pert_vec = perturbed[bi_int].detach().cpu().numpy()

                        target_act = target_expression.detach().cpu().numpy()
                        missing_mask_act = (target_act != -1)

                        r = pearsonr(start_vec[missing_mask_act], target_act[missing_mask_act])[0]
                        s = spearmanr(start_vec[missing_mask_act], target_act[missing_mask_act])[0]
                        mae = np.mean(np.abs(start_vec[missing_mask_act] - target_act[missing_mask_act]))
                        results_actual.append((r, s, mae, prop, "Start"))

                        r = pearsonr(pert_vec[missing_mask_act], target_act[missing_mask_act])[0]
                        s = spearmanr(pert_vec[missing_mask_act], target_act[missing_mask_act])[0]
                        mae = np.mean(np.abs(pert_vec[missing_mask_act] - target_act[missing_mask_act]))
                        results_actual.append((r, s, mae, prop, "Perturbed"))

                        try:
                            target_pr = target_out.detach().cpu().numpy()
                        except:
                            target_pr = target_out   # already numpy
                        missing_mask_pr = (target_pr != -1)

                        # predicted: start
                        r = pearsonr(start_vec[missing_mask_pr], target_pr[missing_mask_pr])[0]
                        s = spearmanr(start_vec[missing_mask_pr], target_pr[missing_mask_pr])[0]
                        mae = np.mean(np.abs(start_vec[missing_mask_pr] - target_pr[missing_mask_pr]))
                        results_pred.append((r, s, mae, prop, "Start"))

                        # predicted: perturbed
                        r = pearsonr(pert_vec[missing_mask_pr], target_pr[missing_mask_pr])[0]
                        s = spearmanr(pert_vec[missing_mask_pr], target_pr[missing_mask_pr])[0]
                        mae = np.mean(np.abs(pert_vec[missing_mask_pr] - target_pr[missing_mask_pr]))
                        results_pred.append((r, s, mae, prop, "Perturbed"))

        df_actual = pd.DataFrame(results_actual,
            columns=["Pearson", "Spearman", "MAE", "Prop", "Type"]
        )
        df_actual.to_csv(os.path.join(save_dir, f"{savename}_{temper_method}_actualtarget.csv"), index=False)
        print("Saved actualtarget.csv")

        df_pred = pd.DataFrame(results_pred,
            columns=["Pearson", "Spearman", "MAE", "Prop", "Type"]
        )
        df_pred.to_csv(os.path.join(save_dir, f"{savename}_{temper_method}_predictedtarget.csv"), index=False)
        print("Saved predictedtarget.csv")
 
if __name__ == "__main__":
    main()
