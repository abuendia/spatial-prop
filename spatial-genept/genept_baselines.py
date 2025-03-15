from openai import OpenAI
import numpy as np
import scanpy as sc
from tqdm import tqdm
import pickle
import json 
from PIL import Image
import os 
import argparse
import multiprocessing

from utils import load_gpt_key, encode_image

gpt_key = load_gpt_key()
image_model = "gpt-4o"
embedding_model = "text-embedding-ada-002"


def compute_spatial_description(task):
    """Computes the spatial description for a given cell image."""
    cell_id, image_path = task
    client = OpenAI(api_key=gpt_key)
    image_prompt = """
    Describe this DAPI stain using the below template.

    Estimate the density of cells around the center of image (very low, low, medium, high, very high):
    Describe the overall spatial distribution of cells. Be very specific:
    Describe the morphology of cells:
    """
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model=image_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{image_prompt}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
    )
    return cell_id, response.choices[0].message.content


def process_spatial_descriptions(adata, output_path, num_workers):
    """Parallelizes the spatial description computation across multiple processes."""
    image_dir = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/image_cell_crops/balanced_sample_denoised"
    task_list = [
        (cell_id, os.path.join(image_dir, f"{cell_id}-z1.jpeg"))
        for cell_id in adata.obs_names if os.path.exists(os.path.join(image_dir, f"{cell_id}-z1.jpeg"))
    ]
    
    results = {}
    print("Entering multiprocessing...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        for cell_id, result in tqdm(pool.imap(compute_spatial_description, task_list), total=len(task_list), desc="Processing spatial descriptions"):
            results[cell_id] = result
    
    with open(output_path, "w") as f:
        json.dump(results, f)

def compute_genepts_embedding(task):
    """Computes the gene pathway embedding for a given cell."""
    cell_id, adata_sampled, description = task
    client = OpenAI(api_key=gpt_key)
    umi_counts = adata_sampled[cell_id].X[0].tolist()
    gene_names = adata_sampled.var.index.tolist()
    
    # Sort gene names by umi count (highest first)
    umi_counts, gene_names = zip(*sorted(zip(umi_counts, gene_names), reverse=True))
    
    # Create string of gene followed by count 
    gene_names = [f"{gene} {count}" for gene, count in zip(gene_names, umi_counts)]
    # Add the cell description to the input prompt
    input_prompt = description
    # input_prompt = ' '.join(gene_names).upper()
    # if description:
    #     input_prompt += f"\n\n{description}"
    
    response = client.embeddings.create(
        input=input_prompt,
        model=embedding_model,
    )
    return cell_id, response.data[0].embedding, input_prompt


def process_genepts_embeddings(adata, cell_id_to_description, output_path, num_workers):
    """Parallelizes the gene pathway embedding computation across multiple processes."""
    task_list = [
        (cell_id, adata, cell_id_to_description.get(cell_id, "") if cell_id in cell_id_to_description else "")
        for cell_id in adata.obs_names
    ]
    
    results = {}
    print("Entering multiprocessing...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        for cell_id, embedding, input_prompt in tqdm(pool.imap(compute_genepts_embedding, task_list), total=len(task_list), desc="Processing gene embeddings"):
            results[cell_id] = embedding
    
    with open(output_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    adata_path = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/sampled_adata/aging_coronal_balanced.h5ad"
    cell_id_to_description = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/image_descriptions/spatial_descriptions.json"

    print("Reading adata...")
    adata = sc.read(adata_path)

    with open(cell_id_to_description, "r") as f:
        cell_id_to_description = json.load(f)
    process_genepts_embeddings(adata, cell_id_to_description, args.output_path, num_workers=args.num_workers)
