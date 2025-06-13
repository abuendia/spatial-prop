import numpy as np
import scanpy as sc
from tqdm import tqdm
import pickle
import json 
from PIL import Image
import os 
import argparse
from voyageai import Client
from utils import load_voyage_key
from multiprocessing import Pool

api_key = load_voyage_key()


def query_voyage_model(cell_id, sub_adata, cell_id_to_mouse_id, z_value, save_dir):
    client = Client(api_key=api_key)
    mouse_id = cell_id_to_mouse_id[cell_id]
    image_path = os.path.join(save_dir, f"cell{cell_id}_mouse{mouse_id}_z{z_value}.png")
    celltype_label = sub_adata[cell_id].obs['celltype'].values[0]

    image = Image.open(image_path)
    documents = [[image]]
    result = client.multimodal_embed(
        inputs=documents,
        model="voyage-multimodal-3",
        input_type="document"
    )
    return cell_id, result.embeddings[0], celltype_label
    

def get_multimodal_embedding_parallel(sub_adata, cell_id_to_mouse_id, z_value, save_dir, results_dir):
    with Pool(processes=8) as pool:
        results = pool.starmap(query_voyage_model, tqdm([(cell_id, sub_adata, cell_id_to_mouse_id, z_value, save_dir) for cell_id in sub_adata.obs_names], total=len(sub_adata.obs_names)))

    results_dict = {cell_id: (embed, true_label) for cell_id, embed, true_label in results}
    # save as json
    with open(os.path.join(results_dir, f"embeds_voyageai_neighborhood.json"), "w") as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    save_dir = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/zeroshot_eval_celltype_neighborhood"
    results_dir = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/voyage_embeds_image_only"
    os.makedirs(results_dir, exist_ok=True)
    
    coronal_sampled = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/sampled_adata/aging_coronal_balanced.h5ad"
    adata = sc.read(coronal_sampled)

    z_value = 2
    cell_id_to_mouse_id = adata.obs.mouse_id.to_dict()
    get_multimodal_embedding_parallel(adata, cell_id_to_mouse_id, z_value, save_dir, results_dir)
