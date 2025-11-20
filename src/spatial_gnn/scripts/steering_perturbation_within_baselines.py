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
from spatial_gnn.utils.perturbation_utils import batch_steering_cell, batch_steering_mean
from spatial_gnn.utils.dataset_utils import load_dataset_config, parse_center_celltypes
from spatial_gnn.datasets.spatial_dataset import SpatialAgingCellDataset
from spatial_gnn.utils.perturbation_utils import predict, temper
from spatial_gnn.models.mean_baselines import (
    global_mean_baseline_batch,
    center_celltype_global_mean_baseline_batch,
    khop_mean_baseline_batch,
    center_celltype_mean_baseline_batch,
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
    parser.add_argument("--learning_rate", help="learning rate", type=float, required=True)
    parser.add_argument("--loss", help="loss: balanced_mse, npcc, mse, l1", type=str, required=True)
    parser.add_argument("--epochs", help="number of epochs", type=int, required=True)
    parser.add_argument("--gene_list", help="Path to file containing list of genes to use (optional)", type=str, default=None)
    parser.add_argument("--exp_name", help="experiment name", type=str, required=True)
    parser.add_argument("--debug", help="debug mode", action="store_true")
    
    # steering-specific arguments
    parser.add_argument("--steering_approach", help="steering method to use", type=str)
    parser.add_argument("--num_props", help="number of intervals from 0 to 1", type=int)
    parser.add_argument("--baseline_type", help="baseline type to use", type=str)
    
    args = parser.parse_args()
    steering_approach = args.steering_approach
    num_props = args.num_props
    props = np.linspace(0,1,num_props+1) # proportions to scale
    baseline_type = args.baseline_type

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
    baseline_type = args.baseline_type
    
    # Handle center_celltypes
    center_celltypes = parse_center_celltypes(args.center_celltypes)
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

    train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                            dataset_prefix=args.exp_name,
                                            target="expression",
                                            k_hop=k_hop,
                                            augment_hop=augment_hop,
                                            node_feature=node_feature,
                                            inject_feature=inject_feature,
                                            num_cells_per_ct_id=100,
                                            center_celltypes=center_celltypes,
                                            use_ids=train_ids,
                                            raw_filepaths=[file_path],
                                            celltypes_to_index=celltypes_to_index)
    
    # init dataset with settings
    test_dataset = SpatialAgingCellDataset(subfolder_name="test",
                                        dataset_prefix=args.exp_name,
                                        target="expression",
                                        k_hop=k_hop,
                                        augment_hop=augment_hop,
                                        node_feature=node_feature,
                                        inject_feature=inject_feature,
                                        num_cells_per_ct_id=100,
                                        center_celltypes=center_celltypes,
                                        use_ids=test_ids,
                                        raw_filepaths=[file_path],
                                        celltypes_to_index=celltypes_to_index)

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
    save_dir = os.path.join("baselines", test_dataset.processed_dir.split("/")[-2], args.exp_name, baseline_type)
    os.makedirs(save_dir, exist_ok=True)

    cell_type_model = CellTypeGNN(
        hidden_channels=64,
        input_dim=int(train_dataset.get(0).x.shape[1]),
        num_layers=k_hop,
        method="GIN",
        pool="add",
        celltypes_to_index=celltypes_to_index,
    )

    cell_type_model_path = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/final_exps/expression_with_celltype_decoupled_no_genept_softmax_ct/zeng_expression_2hop_2augment_expression_none/weightedl1_1en04/celltype_model.pth"
    cell_type_model.load_state_dict(torch.load(cell_type_model_path))
    
    model = GNN(
        hidden_channels=64,
        input_dim=int(train_dataset.get(0).x.shape[1]),
        output_dim=len(train_dataset.get(0).y),
        inject_dim=int(train_dataset.get(0).inject.shape[1]) if inject is True else 0,
        method="GIN", 
        pool="add", 
        num_layers=k_hop,
        celltypes_to_index=celltypes_to_index,
        predict_celltype=True,
        train_multitask=False,
        celltype_model=cell_type_model,  # Pass pre-trained model for decoupled training
        ablate_gene_expression=False,
        use_one_hot_ct=False,
        attention_pool=None,
    )
    print(f"Model initialized on {device}")

    model_save_dir = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/output/final_exps/expression_with_celltype_decoupled_no_genept_softmax_ct/zeng_expression_2hop_2augment_expression_none/weightedl1_1en04/"
    use_model = "model"
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f"{use_model}.pth")), strict=False)
    model.to(device)

    eval_model(model, train_loader, test_loader, center_celltypes, save_dir, baseline_type, steering_approach, num_props, props)
    
def eval_model(
    model,
    train_loader,
    test_loader, 
    center_celltypes,
    save_dir, 
    baseline_type,
    steering_approach,
    num_props,
    props,
    device="cuda",
):
    # get savename
    savename = f"{steering_approach}_{num_props}steps_{baseline_type}"

    # precompute training set baselines
    if baseline_type == "global_mean":
        global_mean = global_mean_baseline_batch(train_loader)
    elif baseline_type == "center_celltype_global_mean":
        ct_to_mean = center_celltype_global_mean_baseline_batch(train_loader)
    
    print("Running steering...", flush=True)
    for prop in props:

        perturb_props = []
        target_expressions = []
        target_predictions = []
        target_celltypes = []
        start_expressions_list = []
        perturb_expressions_list = []
        start_celltypes_list = []

        for data in tqdm(test_loader):
            data = data.to(device)
            out = predict(model, data, inject=False)

            breakpoint()

            # get actual expression
            if data.y.shape != out.shape:
                actual = torch.reshape(data.y.float(), out.shape)
            else:
                actual = data.y.float()

            ### STEERING PERTURBATION (and appending target expressions)
            subset_same_celltype = False

            random_target_idx = np.random.choice(np.arange(len(data)))
            target = data[random_target_idx]

            if steering_approach == "batch_steer_mean":
                pdata, target_celltype, target_expression, target_out = batch_steering_mean(data, actual, out, center_celltypes, target=target, prop=prop)
                subset_same_celltype = True
            elif steering_approach == "batch_steer_cell":
                pdata, target_celltype, target_expression, target_out = batch_steering_cell(data, actual, out, center_celltypes, target=target, prop=prop)
                subset_same_celltype = True
            else:
                raise Exception("steering_approach not recognized")

            # append target expression and prop
            target_expressions.append(target_expression)
            target_predictions.append(target_out)
            target_celltypes.append(target_celltype)
            start_celltypes_list.append(center_celltypes)
            perturb_props.append(round(prop,3))

            # make predictions with baseline
            if baseline_type == "global_mean":
                batch_size = len(data.center_node)
                pout = global_mean.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, num_genes]
            elif baseline_type == "center_celltype_global_mean":
                pout = []
                for c in np.concatenate(data.center_celltype):
                    pout.append(ct_to_mean[c])
                pout = torch.stack(pout)
            elif baseline_type == "khop_mean":
                pout = khop_mean_baseline_batch(pdata) # [512, 300]
            elif baseline_type == "khop_celltype_mean":
                pout = center_celltype_mean_baseline_batch(pdata) # [512, 300]   
            else:
                raise ValueError(f"Baseline type {baseline_type} not recognized!")

            # temper perturbed expression
            perturbed = pout

            start_expressions = []
            perturb_expressions = []
            for bi in np.unique(pdata.batch):
                # subset to only those that have same center cell type as first graph
                if subset_same_celltype is True:
                    if (center_celltypes[bi] == target_celltype) and (bi>0):
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

    stats_df.to_csv(os.path.join(save_dir, f"{savename}_actualtarget.csv"))
    stats_df = pd.read_csv(os.path.join(save_dir, f"{savename}_actualtarget.csv"))

 
if __name__ == "__main__":
    main()
