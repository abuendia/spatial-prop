import numpy as np
import scanpy as sc
from tqdm import tqdm
import pickle
import json 
from PIL import Image
import os 
import argparse
from voyageai import Client
from utils import get_voyage_api_key

api_key = get_voyage_api_key()


def get_multimodal_embedding(mouse_id):
    """
    Generates a multimodal embedding for a given text and image.
    """
    vo = Client(api_key=api_key)
    filedir = mouseid_to_filedir_dict[mouse_id]
    sub_adata = adata_sampled[adata_sampled.obs.mouse_id==mouse_id].copy()

    cell_id_to_text_embed = {}
    cell_id_to_multimodal_embed = {}

    for cell_id in tqdm(sub_adata.obs_names):
        umi_counts = adata_sampled[cell_id].X[0].tolist()
        gene_names = adata_sampled.var.index.tolist()

        # sort by umi_counts
        umi_counts, gene_names = zip(*sorted(zip(umi_counts, gene_names), reverse=True))
        
        # create string from gene names
        text_prompt = " ".join(gene_names).upper()
        image_path = f"/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/image_cell_crops/balanced_sample/{cell_id}-z1.jpeg"
        image = Image.open(image_path)

        documents = [
            [text_prompt],
            [text_prompt, image]
        ]
        result = vo.multimodal_embed(
            inputs=documents,
            model="voyage-multimodal-3",
            input_type="document"
        )
        cell_id_to_text_embed[cell_id] = result.embeddings[0]
        cell_id_to_multimodal_embed[cell_id] = result.embeddings[1]
    
    # save to file
    with open(f"/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/voyage_embeds/{mouse_id}_text_embed.pkl", "wb") as f:
        pickle.dump(cell_id_to_text_embed, f)
    with open(f"/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/voyage_embeds/{mouse_id}_multimodal_embed.pkl", "wb") as f:
        pickle.dump(cell_id_to_multimodal_embed, f)
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse_id", type=str, help="Mouse ID")
    args = ap.parse_args()

    print("Starting mouse ID: ", args.mouse_id)
    adata_sampled = sc.read("/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/processed/aging_coronal_balanced.h5ad")
    with open("/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/utils/mouseid_to_filedir_dict.pkl", "rb") as f:
        mouseid_to_filedir_dict = pickle.load(f)

    get_multimodal_embedding(args.mouse_id)
