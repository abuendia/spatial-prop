# <img src="./assets/spatialprop-logo.png" width="40" /> SpatialProp: Tissue perturbation modeling with spatially resolved single-cell transcriptomics

**SpatialProp** (Spatial Propagation of Single-cell Perturbations) is a computational framework leveraging graph deep learning to predict the spatial effects of single-cell genetic perturbations using spatially resolved single-cell transcriptomics data.

![architecture](./assets/spatialprop-arch.png)

SpatialProp takes as input a spatially resolved single-cell transcriptomics dataset of intact tissue and a user-defined set of single-cell perturbations represented by their perturbed gene expression profiles. Then, using the core graph neural network module, SpatialProp predicts perturbed gene expression in a cell-by-cell manner and calibrates these predictions for model error to update gene expression profiles for every cell in the tissue. Finally, SpatialProp outputs a prediction of the perturbed gene expression profiles for all cells in the spatially resolved single-cell transcriptomics data, including for cells that did not receive a direct user-specified perturbation.

## Using SpatialProp

To train and deploy SpatialProp on a new dataset, the following steps need to be taken:
- Train SpatialProp GNN from scratch on a spatially resolved single-cell transcriptomics dataset (e.g. MERFISH, Xenium, Slide-seq, Stereo-seq, STARmap, etc.).
- Specify the single-cell perturbations to make in the tissue.
- SpatialProp makes these perturbations and then uses the GNN along with the calibration framework to predict the perturbed gene expression across the entire tissue section. Training and predicting on different subsets of the data is recommended.
- SpatialProp includes additional utilities to visualize the predicted perturbed gene expression.

Also included are scripts for running a set of evaluation frameworks for SpatialProp (or any spatial perturbation model). These include iterative steering of niches to a target state, and the cell-cell causal interaction benchmark under CausalInteractionBench: https://github.com/sunericd/CausalInteractionBench. Please see the preprint for more details.

![applications](./assets/spatialprop-apps.png)

## Installation

To install SpatialProp, run the following from the current directory:

    pip install .

## Training the SpatialProp GNN

Training the SpatialProp GNN can be done through the accompanying lightweight [API](./src/spatial_gnn/api/perturbation_api.py). Here we show an example of training the GNN on the `aging_coronal.h5ad` dataset from [Sun et al., 2025](https://www.nature.com/articles/s41586-024-08334-8) found at this Zenodo [link](https://zenodo.org/records/13883177):

    training_args = {
        "dataset": "aging_coronal",
        "file_path": "/path/to/anndata",
        "train_ids": ["train_mouse_1", "train_mouse_2"], 
        "test_ids": ["test_mouse_1"],
        "exp_name": "aging_coronal_train",
        "k_hop": 2,
        "augment_hop": 2,
        "center_celltypes": "all",
        "node_feature": "expression",
        "inject_feature": "none",
        "learning_rate": 0.0001,
        "loss": "weightedl1",
        "epochs": 30,
        "normalize_total": True,
        "num_cells_per_ct_id": 100,
        "predict_celltype": False,
        "pool": "center",
        "do_eval": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    test_loader, gene_names, (model, model_config, trained_model_path) = train_perturbation_model(
        **training_args,
    )

Setting `"debug": True` in the training config can be used for quick testing. Note that setting `"do_eval": True` computes Pearson correlation, Spearman correlation, and MAE metrics between predictions and ground truth as reported in the preprint. An equivalent bash script for launching training is found [here](./bash/run_train_gnn_batch.sh).

## Deploy SpatialProp with a trained GNN: Inflammatory signaling example

We show use of the SpatialProp [API](./src/spatial_gnn/api/perturbation_api.py) to deploy a trained SpatialProp model in predicting response to perturbations on pro-inflammatory genes.

For instance, we may want to perturb cytokines IL-6, TNF, and IFNG in T cells and microglia of a coronal tissue section of mouse brain. Here the user specifies a dictionary of desired perturbations and multipliers that scale the gene expressions in the desired cell types. We increase expression tenfold by specifying the following perturbation dict:

    perturbation_dict = {
        'T cell': {'Il6': 10.0, 'Tnf': 10.0, 'Ifng': 10.0},    
        'Microglia': {'Il6': 10.0, 'Tnf': 10.0, 'Ifng': 10.0},          
    }

Then we can apply these perturbations and compute SpatialProp-predicted effects with the following API calls. The first call saves the perturbed gene expression matrix into an anndata object at `save_path` in the `anndata.obsm['perturbed_input']` attribute.

    save_path = create_perturbation_input_matrix(
        adata,
        perturbation_dict,
        save_path=save_path
    )
    adata_result = predict_perturbation_effects(
        save_path,
        trained_model_path,
        exp_name,
        use_ids=test_ids
    )

The `predict_perturbation_effects` function will return an updated anndata object with predicted propagated expression that can be accessed through `adata_result.layers['predicted_perturbed']` (raw GNN predictions) and `adata_result.layers['predicted_tempered']` (full SpatialProp pipeline with calibration).

Check out the [notebook example](./notebooks/api_demo.ipynb) for the end-to-end training and inference workflow on an example dataset, as well as plotting utilities.
