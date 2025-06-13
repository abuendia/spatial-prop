import argparse
import sys
import scanpy as sc 
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from metrics import compute_ari_ami
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    # Raw expression
    adata_path = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/sampled_adata/aging_coronal_balanced.h5ad"
    adata = sc.read(adata_path)
    gene_list = adata.var_names

    # create map from cell id to gene expression vector
    cell_to_expression = {}
    for i, cell in enumerate(adata.obs_names):
        cell_to_expression[cell] = adata.X[i]
    
    # create map from cell id to celltype label
    cell_to_label = {}
    for i, cell in enumerate(adata.obs_names):
        cell_to_label[cell] = adata.obs["celltype"][i]
    
    cell_list = list(cell_to_expression.keys())
    exp_ari, exp_ami = compute_ari_ami(cell_list, cell_to_expression, cell_to_label)
    print(f"ARI for expression: {exp_ari}")
    print(f"AMI for expression: {exp_ami}")
    print(f"Embedding dim: {cell_to_expression[cell_list[0]].shape[0]}")
    print("-----------------")
    
    # GenePT-s base
    genepts_base = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/genepts_base.json"
    genepts_base = json.load(open(genepts_base))
    genepts_base_ari, genepts_base_ami = compute_ari_ami(cell_list, genepts_base, cell_to_label)
    print(f"ARI for GenePT-s base: {genepts_base_ari}")
    print(f"AMI for GenePT-s base: {genepts_base_ami}")
    print(f"Embedding dim: {len(genepts_base[cell_list[0]])}")
    print("-----------------")

    # GenePT-s + Image description
    genepts_img = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/genepts_genelist_plus_description.json"
    genepts_img = json.load(open(genepts_img))
    genepts_img_ari, genepts_img_ami = compute_ari_ami(cell_list, genepts_img, cell_to_label)
    print(f"ARI for GenePT-s + Image: {genepts_img_ari}")
    print(f"AMI for GenePT-s + Image: {genepts_img_ami}")
    print(f"Embedding dim: {len(genepts_img[cell_list[0]])}")
    print("-----------------")

    # GenePT-s + Genelist + Values + Description
    genepts_all = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/genepts_genelist_umi_description.json"
    genepts_all = json.load(open(genepts_all))
    genepts_all_ari, genepts_all_ami = compute_ari_ami(cell_list, genepts_all, cell_to_label)
    print(f"ARI for GenePT-s + Genelist + Values + Description: {genepts_all_ari}")
    print(f"AMI for GenePT-s + Genelist + Values + Description: {genepts_all_ami}")
    print(f"Embedding dim: {len(genepts_all[cell_list[0]])}")
    print("-----------------")

    # GenePT-w l2-norm 
    genepts_w = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/geneptw-l2-norm.json"
    genepts_w = json.load(open(genepts_w))
    genepts_w_ari, genepts_w_ami = compute_ari_ami(cell_list, genepts_w, cell_to_label)
    print(f"ARI for GenePT-w l2-norm: {genepts_w_ari}")
    print(f"AMI for GenePT-w l2-norm: {genepts_w_ami}")
    print(f"Embedding dim: {len(genepts_w[cell_list[0]])}")
    print("-----------------")

    # Spatial description only
    spatial_desc = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/spatial_description_only.json"
    spatial_desc = json.load(open(spatial_desc))
    spatial_desc_ari, spatial_desc_ami = compute_ari_ami(cell_list, spatial_desc, cell_to_label)
    print(f"ARI for Spatial description only: {spatial_desc_ari}")
    print(f"AMI for Spatial description only: {spatial_desc_ami}")
    print(f"Embedding dim: {len(spatial_desc[cell_list[0]])}")
    print("-----------------")

    # concat geneptw-l2-norm and spatial description only
    concat_geneptw_spatial = {}
    for cell in cell_list:
        # concat the two vectors
        concat_geneptw_spatial[cell] = np.concatenate([genepts_w[cell], spatial_desc[cell]])
    concat_geneptw_spatial_ari, concat_geneptw_spatial_ami = compute_ari_ami(cell_list, concat_geneptw_spatial, cell_to_label)
    print(f"ARI for concat GenePT-w l2-norm and Spatial description: {concat_geneptw_spatial_ari}")
    print(f"AMI for concat GenePT-w l2-norm and Spatial description: {concat_geneptw_spatial_ami}")
    print(f"Embedding dim: {len(concat_geneptw_spatial[cell_list[0]])}")
    print("-----------------")



def linear_probe(set_a, set_b, labels):
    class LinearProbe(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.weight_A = nn.Parameter(torch.ones(input_dim))  # Vector of shape (1536,)
            self.weight_B = nn.Parameter(torch.ones(input_dim))  # Vector of shape (1536,)
            self.fc = nn.Linear(input_dim, num_classes)  # Linear classifier

        def forward(self, X_A, X_B):
            combined = self.weight_A * X_A + self.weight_B * X_B  # Element-wise multiplication
            return self.fc(combined)

    # train-test split of keys of labels
    cell_list = list(labels.keys())
    # train-test split and make sure that the train and test sets have the same distribution of labels
    X_train, X_test, y_train, y_test = train_test_split(cell_list, [labels[cell] for cell in cell_list], test_size=0.8, stratify=[labels[cell] for cell in cell_list])

    # convert labels from string to int
    label_to_int = {label: i for i, label in enumerate(np.unique(y_train))}
    y_train = [label_to_int[label] for label in y_train]
    y_test = [label_to_int[label] for label in y_test]

    # convert to torch tensors
    X_A_train = np.array([set_a[cell] for cell in X_train])
    X_B_train = np.array([set_b[cell] for cell in X_train])
    y_train = np.array(y_train)
    X_A_test = np.array([set_a[cell] for cell in X_test])
    X_B_test = np.array([set_b[cell] for cell in X_test])
    y_test = np.array(y_test)

    X_A_train_torch = torch.tensor(X_A_train, dtype=torch.float32)
    X_B_train_torch = torch.tensor(X_B_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    X_A_test_torch = torch.tensor(X_A_test, dtype=torch.float32)
    X_B_test_torch = torch.tensor(X_B_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.long)

    num_features = X_A_train_torch.shape[1]
    num_classes = len(np.unique(y_train))

    model = LinearProbe(num_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # early stopping
    patience = 5  # Number of epochs to wait for improvement
    best_test_loss = float("inf")
    epochs_no_improve = 0

    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_A_train_torch, X_B_train_torch)
        loss = criterion(outputs, y_train_torch)

        # compute loss on test set
        test_outputs = model(X_A_test_torch, X_B_test_torch)
        test_loss = criterion(test_outputs, y_test_torch)

        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            epochs_no_improve = 0  # Reset counter if improvement
        else:
            epochs_no_improve += 1  # Increment counter if no improvement

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best test loss: {best_test_loss:.4f}")
            break  # Stop training

    # Evaluate on test set
    with torch.no_grad():
        test_outputs = model(X_A_test_torch, X_B_test_torch)
        test_preds = torch.argmax(test_outputs, dim=1)
        test_accuracy = accuracy_score(y_test, test_preds.numpy())

    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    # use the learned weight_A and weight_B to compute the combined embedding
    combined_embeddings = {}
    test_labels = {}
    for cell in X_test:
        combined_embeddings[cell] = model.weight_A.detach().numpy() * set_a[cell] + model.weight_B.detach().numpy() * set_b[cell]
        test_labels[cell] = labels[cell]

    combined_ari, combined_ami = compute_ari_ami(X_test, combined_embeddings, test_labels)
    print(f"ARI for combined embeddings: {combined_ari}")
    print(f"AMI for combined embeddings: {combined_ami}")
    print(f"Embedding dim: {len(combined_embeddings[X_test[0]])}")
    print("-----------------")


if __name__ == '__main__':
    genepts_w = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/geneptw-l2-norm.json"
    genepts_w = json.load(open(genepts_w))

    spatial_desc = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/spatial_description_only.json"
    spatial_desc = json.load(open(spatial_desc))

    adata_path = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/sampled_adata/aging_coronal_balanced.h5ad"
    adata = sc.read(adata_path)
    gene_list = adata.var_names
    cell_to_label = {}
    for i, cell in enumerate(adata.obs_names):
        cell_to_label[cell] = adata.obs["celltype"][i]

    linear_probe(genepts_w, spatial_desc, cell_to_label)
    
