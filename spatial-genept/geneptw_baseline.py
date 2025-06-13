import argparse
import scanpy as sc 
import json 
import numpy as np
import sys


def main(args):
    args.gene_to_summary_embed = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/genept/GenePT_emebdding_v2/summary_embeddings_all.json"
    # load adata object
    adata = sc.read(args.adata_path)
    # load gene to summary embed
    with open(args.gene_to_summary_embed, "r") as f:
        gene_to_summary_embed = json.load(f)   

    print("Loaded gene to summary embeddings", flush=True)
    sys.stdout.flush() 

    # create matrix of gene to summary embeddings 
    gene_to_summary_embed_matrix = []
    for gene in adata.var_names:
        gene = gene.upper()
        gene_to_summary_embed_matrix.append(gene_to_summary_embed[gene])
    gene_to_summary_embed_matrix = np.array(gene_to_summary_embed_matrix)

    print("Created gene to summary embeddings matrix", flush=True)
    sys.stdout.flush()

    cell_gene_counts = adata.X
    geneptw_embeds = cell_gene_counts @ gene_to_summary_embed_matrix
    # normalize geneptw_embeds rows using l2 norm
    geneptw_embeds = geneptw_embeds / np.linalg.norm(geneptw_embeds, axis=1, keepdims=True)

    print("Computed geneptw embeddings", flush=True)
    sys.stdout.flush()

    cellid_to_embed = {}

    for i, cell_id in enumerate(adata.obs_names):
        cellid_to_embed[cell_id] = geneptw_embeds[i].tolist()
    
    with open(args.output_path, "w") as f:
        json.dump(cellid_to_embed, f)

    print("Saved geneptw embeddings", flush=True)
    sys.stdout.flush()
    
    return geneptw_embeds

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--adata_path", type=str)
    argparse.add_argument("--output_path", type=str)
    args = argparse.parse_args()

    main(args)
