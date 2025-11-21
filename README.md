# 🧫 SpatialProp: Tissue perturbation modeling with spatially resolved single-cell transcriptomics

SpatialProp (Spatial Propagation of Single-cell Perturbations) is a computational framework leveraging graph deep learning to predict the spatial effects of single-cell genetic perturbations using spatially resolved single-cell transcriptomics data.

SpatialProp takes as input: a spatially resolved single-cell transcriptomics dataset of intact tissue and a user-defined set of single-cell perturbations represented by their perturbed gene expression profiles. Then, using the core graph neural network module, SpatialProp predicts perturbed gene expression in a cell-by-cell manner and calibrates these predictions for model error to update gene expression profiles for every cell in the tissue. Finally, SpatialProp outputs a prediction of the perturbed gene expression profiles for all cells in the spatially resolved single-cell transcriptomics data, including for cells that did not receive a direct user-specified perturbation.

To deploy SpatialProp on a new dataset, the following steps need to be taken:
- Train SpatialProp GNN from scratch on the new spatially resolved single-cell transcriptomics dataset (e.g. MERFISH, Xenium, Slide-seq, Stereo-seq, STARmap, etc.)
- Specify the single-cell perturbations to make in the tissue
- SpatialProp makes these perturbations and then uses the GNN along with the calibration framework to predict the perturbed gene expression across the entire tissue section (Recommendation: train and predict on different subsets of data)
- SpatialProp includes additional utilities to visualize the predicted perturbed gene expression

Also included are scripts for running a set of evaluation frameworks for SpatialProp (or any spatial perturbation model). This includes the cell-cell interaction benchmark under CausalInteractionBench: https://github.com/sunericd/CausalInteractionBench


## Installation

    conda env create -f environment.yml
    conda activate spatial-gnn
    pip install -e .

## Training GNN

We provide an example training command: 

    python scripts/train_gnn_model_expression.py \
        --dataset aging_coronal \
        --base_path /path/to/anndata \
        --k_hop 2 \
        --augment_hop 2 \
        --center_celltypes "T cell,NSC,Pericyte" \
        --node_feature "expression" \
        --inject_feature "none" \
        --learning_rate 0.0001 \
        --loss "weightedl1" \
        --epochs 50

## Deploy SpatialProp with trained GNN (inflammatory signaling example)

For this tutorial, you can access the following datasets at https://zenodo.org/records/13883177.
