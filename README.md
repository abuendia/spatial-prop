# <img src="./assets/spatial-prop-logo.png" width="25" /> SpatialProp: Tissue perturbation modeling with spatially resolved single-cell transcriptomics

**SpatialProp** (Spatial Propagation of Single-cell Perturbations) is a computational framework leveraging graph deep learning to predict the spatial effects of single-cell genetic perturbations using spatially resolved single-cell transcriptomics data.

![architecture](./assets/spatial-prop-schematic.png)

SpatialProp takes as input a spatially resolved single-cell transcriptomics dataset of intact tissue and a user-defined set of single-cell perturbations represented by their perturbed gene expression profiles. Then, using the core graph neural network module, SpatialProp predicts perturbed gene expression in a cell-by-cell manner and calibrates these predictions for model error to update gene expression profiles for every cell in the tissue. Finally, SpatialProp outputs a prediction of the perturbed gene expression profiles for all cells in the spatially resolved single-cell transcriptomics data, including for cells that did not receive a direct user-specified perturbation.

To deploy SpatialProp on a new dataset, the following steps need to be taken:
- Train SpatialProp GNN from scratch on a spatially resolved single-cell transcriptomics dataset (e.g. MERFISH, Xenium, Slide-seq, Stereo-seq, STARmap, etc.).
- Specify the single-cell perturbations to make in the tissue.
- SpatialProp makes these perturbations and then uses the GNN along with the calibration framework to predict the perturbed gene expression across the entire tissue section. Training and predicting on different subsets of the data is recommended.
- SpatialProp includes additional utilities to visualize the predicted perturbed gene expression.

Also included are scripts for running a set of evaluation frameworks for SpatialProp (or any spatial perturbation model). This includes iterative steering of niches to a target state, and the cell-cell interaction benchmark under CausalInteractionBench: https://github.com/sunericd/CausalInteractionBench

![applications](./assets/spatial-prop-apps.png)

## Installation

To install SpatialProp, run the following command from the current directory:

    pip install .

## Training the SpatialProp GNN

We provide an example command to train SpatialProp on the `aging_coronal.h5ad` dataset from https://zenodo.org/records/13883177.

    python scripts/train_gnn_model_expression.py \
        --dataset aging_coronal \
        --base_path path/to/anndata/dir \
        --k_hop 2 \
        --augment_hop 2 \
        --center_celltypes "all" \
        --node_feature "expression" \
        --inject_feature "none" \
        --learning_rate 0.0001 \
        --loss weightedl1 \
        --epochs 50 \
        --do_eval

An optional `--debug` flag can be added for quick testing. Note that the `--do_eval` flag computes Pearson correlation, Spearman correlation, and MAE metrics between predictions and ground truth as reported in the preprint.

## Deploy SpatialProp with trained GNN (inflammatory signaling example)

For this example, you can access the following dataset at https://zenodo.org/records/13883177. We show use of the [lightweight API](./src/spatial_gnn/api/perturbation_api.py) to deploy a trained SpatialProp model in predicting inflammatory response.

In this example, we perturb pro-inflammatory cytokines IL-6, TNF, and IFN-γ in T cells and microglia of a coronal tissue section of mouse brain. Here the user specifies a dictionary of desired perturbations and multipliers that scale the gene expressions in the desired set of cells. For example, the baseline expression of IL-6 will be multiplied by 10 in both T cells and microglia in this example:

    perturbation_dict = {
        'T cell': {'Il6': 10.0, 'Tnf': 10.0, 'Ifng': 10.0},    
        'Microglia': {'Il6': 10.0, 'Tnf': 10.0, 'Ifng': 10.0},          
    }

Assume our input anndata object is loaded as `test_adata`. Then we can apply these perturbations and compute SpatialProp-predicted effects with the following API calls:

    save_path = create_perturbation_input_matrix(
        test_adata,
        perturbation_dict,
        save_path=save_path,
        normalize_total=True
    )
    adata_perturbed = predict_perturbation_effects(
        adata_path=save_path,
        exp_name="aging_coronal_perturbed_debug",
        model_path=model_path,
        perturbation_dict=perturbation_dict,
        perturbation_mask_key="perturbed_input"
    )

This will return an updated anndata object with predicted propagated expression that can be accessed as `adata_perturbed.layers['predicted_perturbed']`. For more details, refer to the [notebook example](./notebooks/api_demo.ipynb) of the API.
