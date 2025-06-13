import os
import json
import pickle
import numpy as np
import scanpy as sc
from tqdm import tqdm
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from openai import OpenAI

from utils import load_gpt_key, encode_image

PIL.Image.MAX_IMAGE_PIXELS = None


gpt_key = load_gpt_key()
image_model = "gpt-4o"
embedding_model = "text-embedding-ada-002"


def process_cell(cell_id, sub_adata, mouseid_to_filedir_dict, cell_id_to_mouse_id, z_value, save_dir):
    try:
        mouse_id = cell_id_to_mouse_id[cell_id]
        filedir = mouseid_to_filedir_dict[mouse_id]

        with open(os.path.join(filedir, "images", "manifest.json"), "rb") as f:
            manifest_dict = json.load(f)

        img = Image.open(os.path.join(filedir, "images", f"mosaic_DAPI_z{z_value}.tif"))

        X_min_micron = sub_adata[cell_id].obs['min_x'].values[0]
        X_max_micron = sub_adata[cell_id].obs['max_x'].values[0]
        X_center_micron = sub_adata[cell_id].obs['center_x'].values[0]
        Y_min_micron = sub_adata[cell_id].obs['min_y'].values[0]
        Y_max_micron = sub_adata[cell_id].obs['max_y'].values[0]
        Y_center_micron = sub_adata[cell_id].obs['center_y'].values[0]

        X_min_pixel = (X_min_micron - manifest_dict['bbox_microns'][0]) / manifest_dict['microns_per_pixel']
        X_max_pixel = (X_max_micron - manifest_dict['bbox_microns'][0]) / manifest_dict['microns_per_pixel']
        Y_min_pixel = (Y_min_micron - manifest_dict['bbox_microns'][1]) / manifest_dict['microns_per_pixel']
        Y_max_pixel = (Y_max_micron - manifest_dict['bbox_microns'][1]) / manifest_dict['microns_per_pixel']
        X_center_pixel = (X_center_micron - manifest_dict['bbox_microns'][0]) / manifest_dict['microns_per_pixel']
        Y_center_pixel = (Y_center_micron - manifest_dict['bbox_microns'][1]) / manifest_dict['microns_per_pixel']
        
        X_width = X_max_pixel - X_min_pixel
        Y_width = Y_max_pixel - Y_min_pixel
        avg_width = (X_width + Y_width) / 2

        scaling_factor = 5
        im1 = img.crop((X_center_pixel - scaling_factor * avg_width, Y_center_pixel - scaling_factor * avg_width, X_center_pixel + scaling_factor * avg_width, Y_center_pixel + scaling_factor * avg_width))
        
        newsize = (224, 224)
        im1 = im1.resize(newsize)

        im1 = np.array(im1)
        im1 = im1 / 256
        im1 = im1.astype('uint8')
        im1 = Image.fromarray(im1)

        save_path = os.path.join(save_dir, f"cell{cell_id}_mouse{mouse_id}_z{z_value}.png")
        plt.imsave(save_path, im1)
    except Exception as e:
        print(f"Error processing cell {cell_id}: {e}")


def crop_images_parallel(sub_adata, mouseid_to_filedir_dict, cell_id_to_mouse_id, z_value, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    with Pool(processes=8) as pool:
        results = pool.starmap(process_cell, tqdm([(cell_id, sub_adata, mouseid_to_filedir_dict, cell_id_to_mouse_id, z_value, save_dir) for cell_id in sub_adata.obs_names], total=len(sub_adata.obs_names)))


def query_gpt_model(cell_id, sub_adata, cell_id_to_mouse_id, z_value, save_dir):
    client = OpenAI(api_key=gpt_key)
    mouse_id = cell_id_to_mouse_id[cell_id]
    image_path = os.path.join(save_dir, f"cell{cell_id}_mouse{mouse_id}_z{z_value}.png")
    celltype_label = sub_adata[cell_id].obs['celltype'].values[0]

    base64_image = encode_image(image_path)
    image_prompt = f"""
    This image shows a DAPI stain of a cell from a mouse brain. Based on the morphology, classify the cell into one of the following types:
    'Neuron-Excitatory', 'Neuron-Inhibitory', 'Neuron-MSN', 'Astrocyte', 'Microglia', 'Oligodendrocyte', 'OPC', 'Endothelial', 'Pericyte', 'VSMC', 'VLMC', 'Ependymal', 'Neuroblast', 'NSC', 'Macrophage', 'Neutrophil', 'T cell', 'B cell'
    Only output the cell type label from the above list. Do not include any other information and do not output that the cell type is unknown.
    """
    
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
    return cell_id, response.choices[0].message.content, celltype_label


def predict_celltype_from_gpt_parallel(sub_adata, cell_id_to_mouse_id, z_value, save_dir, results_dir):
    with Pool(processes=8) as pool:
        results = pool.starmap(query_gpt_model, tqdm([(cell_id, sub_adata, cell_id_to_mouse_id, z_value, save_dir) for cell_id in sub_adata.obs_names], total=len(sub_adata.obs_names)))

    results_dict = {cell_id: (response, true_label) for cell_id, response, true_label in results}
    # save as json
    with open(os.path.join(results_dir, f"celltype_gpt4o_zeroshot.json"), "w") as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    save_dir = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/zeroshot_eval_celltype_neighborhood"
    results_dir = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/gpt_zeroshot_celltype_neighborhood"
    os.makedirs(save_dir, exist_ok=True)
    
    coronal_sampled = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/sampled_adata/aging_coronal_balanced.h5ad"
    adata = sc.read(coronal_sampled)

    
    mouseid_to_filedir_dict = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/utils/mouseid_to_filedir_dict.pkl"
    with open(mouseid_to_filedir_dict, "rb") as f:
        mouseid_to_filedir_dict = pickle.load(f)

    z_value = 2
    cell_id_to_mouse_id = adata.obs.mouse_id.to_dict()
    crop_images_parallel(adata, mouseid_to_filedir_dict, cell_id_to_mouse_id, z_value, save_dir)
    # predict_celltype_from_gpt_parallel(adata, cell_id_to_mouse_id, z_value, save_dir, results_dir)
