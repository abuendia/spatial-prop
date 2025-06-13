import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from cellpose import denoise
import base64
import subprocess


file_to_mouseid_dict = {
    '202301301605_MsBrain-VS62-OC4-OE3-BW_VMSC07101_region_0.h5ad': "OE3",
     '202301301605_MsBrain-VS62-OC4-OE3-BW_VMSC07101_region_1.h5ad': "OC4",
     '202302061111_MsBrain-VS62-YC4-OE4-BW_VMSC07101_region_0.h5ad': "OE4",
     '202302061111_MsBrain-VS62-YC4-OE4-BW_VMSC07101_region_1.h5ad': "YC4",
     '202302061153_MsBrain-VS62-19-30-BW_Beta10_region_0.h5ad': "30",
     '202302061153_MsBrain-VS62-19-30-BW_Beta10_region_1.h5ad': "19",
     '202302061221_MsBrain-VS62-11-38-BW_Beta8_region_0.h5ad': "38",
     '202302061221_MsBrain-VS62-11-38-BW_Beta8_region_1.h5ad': "11",
     '202302071943_MsBrain-VS62-YC3-OC3-BW_VMSC00201_region_0.h5ad': "OC3",
     '202302071943_MsBrain-VS62-YC3-OC3-BW_VMSC00201_region_1.h5ad': "YC3",
     '202302071954_MsBrain-VS62-1-46-BW_VMSC03501_region_0.h5ad': "46",
     '202302071954_MsBrain-VS62-1-46-BW_VMSC03501_region_1.h5ad': "1",
     '202302071956_MsBrain-VS62-7-42-BW_VMSC00701_region_0.h5ad': "42",
     '202302071956_MsBrain-VS62-7-42-BW_VMSC00701_region_1.h5ad': "7",
     '202302072052_MsBrain-VS62-2-BW_Beta10_region_0.h5ad': "2",
     '202302101157_MsBrain-VS62-14-33-BW_Beta10_region_0.h5ad': "14",
     '202302101157_MsBrain-VS62-14-33-BW_Beta10_region_1.h5ad': "33",
     '202302101312_MsBrain-VS62-39-BW_Beta8_region_0.h5ad': "39", # End of Batch 1
     '202308111122_MsBrain-62-VS85-YS_VMSC12502_region_0.h5ad': '62',
     '202308141252_MsBrain-VS85-86-70_VMSC07201_region_0.h5ad': '70',
     '202308141252_MsBrain-VS85-86-70_VMSC07201_region_1.h5ad': '86',
     '202308141339_MsBrain-VS85-53-101_Beta8_region_0.h5ad': '53', # split
     '202308141339_MsBrain-VS85-53-101_Beta8_region_1.h5ad': '101', # split
     '202308141340_MsBrain-VS85-80-75_VMSC12502_region_0.h5ad': '75',
     '202308141340_MsBrain-VS85-80-75_VMSC12502_region_1.h5ad': '80',
     '202308141358_MsBrain-VS85-61-93_VMSC16102_region_0.h5ad': '61', # split
     '202308141358_MsBrain-VS85-61-93_VMSC16102_region_1.h5ad': '93', # split
     '202308181352_MsBrain-VS85-Top-Young-Ctrl2_VMSC07101_region_0.h5ad': 'YC2',
     '202308181352_MsBrain-VS85-Top-Young-Ctrl2_VMSC07101_region_1.h5ad': 'OE2',
     '202308181451_MsBrain-VS85-Top-Young-Ctrl1_VMSC17502_region_0.h5ad': 'OC1',
     '202308181451_MsBrain-VS85-Top-Young-Ctrl1_VMSC17502_region_1.h5ad': 'YC1',
     '202308251220_MsBrain-VS85-Top-Old-Ctrl2_VMSC10802_region_0.h5ad': 'OC2',
     '202308251220_MsBrain-VS85-Top-Old-Ctrl2_VMSC10802_region_1.h5ad': 'OE1',
     '202308291027_MsBrain-VS85-Top-OT902_VMSC12502_region_0.h5ad': 'OT902',
     '202308291027_MsBrain-VS85-Top-OT902_VMSC12502_region_1.h5ad': 'OC903',
     '202309051110_MsBrain-VS85-TopOT1125_VMSC07101_region_0.h5ad': 'OT1125', # split
     '202309051110_MsBrain-VS85-TopOT1125_VMSC07101_region_1.h5ad': 'OC1138', # split
     '202309051447_MsBrain-VS85-Top-YC1989_VMSC17702_region_0.h5ad': 'YC1989',
     '202309051447_MsBrain-VS85-Top-YC1989_VMSC17702_region_1.h5ad': 'YC1975',
     '202309051527_MsBrain-VS85-Top-YC1990_VMSC13402_region_0.h5ad': 'YC1982',
     '202309051527_MsBrain-VS85-Top-YC1990_VMSC13402_region_1.h5ad': 'YC1990',
     '202309091203_MsBrain-VS85-Top-OT1084_VMSC13402_region_0.h5ad': 'OT1084', # split
     '202309091203_MsBrain-VS85-Top-OT1084_VMSC13402_region_1.h5ad': 'OC1083', # split
     '202309091203_MsBrain-VS85-Top-OT1160_VMSC16102_region_0.h5ad': 'OT1160', # split
     '202309091203_MsBrain-VS85-Top-OT1160_VMSC16102_region_1.h5ad': 'OC1226', # split
     '202309091208_MsBrain-VS85-68_VMSC14402_region_0.h5ad': '68',
     '202309141424_MsBrain-VS85-81-S2_VMSC07101_region_0.h5ad': '81',
     '202309141440_MsBrain-VS85-T57-B97_VMSC10802_region_0.h5ad': '57', # split
     '202309141440_MsBrain-VS85-T57-B97_VMSC10802_region_1.h5ad': '97', # split
     '202309150727_MsBrain-VS85-Top-89-S1_VMSC12502_region_0.h5ad': '89', # split
     '202309150727_MsBrain-VS85-Top-89-S1_VMSC12502_region_1.h5ad': '67', # split
     '202309220825_MsBrain-VS85-34_VMSC12502_region_0.h5ad': '34',
}


def sharpen_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file in tqdm(os.listdir(source_dir)):
        if file.endswith(".jpeg"):
            image_path = os.path.join(source_dir, file)
            # read in image
            img = cv2.imread(image_path)
            dn = denoise.DenoiseModel(model_type="deblur_cyto3", gpu=True)
            ret_img = dn.eval(img, channels=[0,0])
            cv2.imwrite(os.path.join(target_dir, file), ret_img)


def crop_images(adata_sampled, mouseid_to_filedir_dict, mouse_id):
    filedir = mouseid_to_filedir_dict[mouse_id]
    sub_adata = adata_sampled[adata_sampled.obs.mouse_id==mouse_id].copy()
        
    with open(os.path.join(filedir, "images", "manifest.json"), "rb") as f:
        manifest_dict = json.load(f)

    img = Image.open(os.path.join(filedir, "images", f"mosaic_DAPI_z1.tif"))
                
    for cell_id in tqdm(sub_adata.obs_names):
        # Define the micron boundaries
        X_min_micron = sub_adata[cell_id].obs['min_x'].values[0]
        X_max_micron = sub_adata[cell_id].obs['max_x'].values[0]
        Y_min_micron = sub_adata[cell_id].obs['min_y'].values[0]
        Y_max_micron = sub_adata[cell_id].obs['max_y'].values[0]

        # Convert the micron boundaries to pixel coordinates
        X_min_pixel = (X_min_micron - manifest_dict['bbox_microns'][0]) / manifest_dict['microns_per_pixel']
        X_max_pixel = (X_max_micron - manifest_dict['bbox_microns'][0]) / manifest_dict['microns_per_pixel']
        Y_min_pixel = (Y_min_micron - manifest_dict['bbox_microns'][1]) / manifest_dict['microns_per_pixel']
        Y_max_pixel = (Y_max_micron - manifest_dict['bbox_microns'][1]) / manifest_dict['microns_per_pixel']

        # Crop the image using the calculated pixel boundaries
        Y_width = Y_max_pixel - Y_min_pixel
        X_width = X_max_pixel - X_min_pixel
        avg_width = (Y_width + X_width) / 2
        im1 = img.crop((X_min_pixel - 5 * avg_width, Y_min_pixel - 5 * avg_width, X_max_pixel + 5 * avg_width, Y_max_pixel + 5 * avg_width))

        # resize image
        newsize = (224, 224)
        im1 = im1.resize(newsize)
        
        # rescale image from 16-bit to 8-bit 
        im1 = np.array(im1)
        im1 = im1/256
        im1 = im1.astype('uint8')
        im1 = Image.fromarray(im1)

        # save cropped image
        save_dir = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/image_cell_crops/"
        save_dir = os.path.join(save_dir, f"balanced_sample")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        im1.save(os.path.join(save_dir,f"{cell_id}-z1.jpeg"),"JPEG")


def sample_adata_object(adata, n_per_cell_type=500):
    sampled_indices = (
        adata.obs.groupby("celltype")
        .apply(lambda x: x.sample(n=n_per_cell_type, replace=False if len(x) >= n_per_cell_type else True, random_state=42))
        .index
    )
    sampled_barcodes = [x[1] for x in sampled_indices]
    adata_sampled = adata[sampled_barcodes, :]
    return adata_sampled


def load_voyage_key():
    with open("/oak/stanford/groups/jamesz/abuen/spatial-rotation/repos/spatial-genept/voyage.key", "r") as f:
        api_key = f.read()
    return api_key


def load_gpt_key():
    with open("/oak/stanford/groups/jamesz/abuen/spatial-rotation/repos/spatial-genept/gpt.key", "r") as f:
        api_key = f.read()
    return api_key


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_available_cpu_ids():
    """Retrieve the list of CPU IDs assigned via numactl."""
    result = subprocess.run("numactl --show | grep 'physcpubind'", shell=True, capture_output=True, text=True)
    cpu_ids = [int(cpu) for cpu in result.stdout.split("physcpubind:")[1].strip().split(' ')]
    return cpu_ids


def get_entrez_gene_summary(
    gene_name, email="alb2281@columbia.edu", organism="human", max_gene_ids=100
):
    """Returns the 'Summary' contents for provided input
    gene from the Entrez Gene database. All gene IDs 
    returned for input gene_name will have their docsum
    summaries 'fetched'.
    
    Args:
        gene_name (string): Official (HGNC) gene name 
           (e.g., 'KAT2A')
        email (string): Required email for making requests
        organism (string, optional): defaults to human. 
           Filters results only to match organism. Set to None
           to return all organism unfiltered.
        max_gene_ids (int, optional): Sets the number of Gene
           ID results to return (absolute max allowed is 10K).
        
    Returns:
        dict: Summaries for all gene IDs associated with 
           gene_name (where: keys → [orgn][gene name],
                      values → gene summary)
    """
    Entrez.email = email

    query = (
        f"{gene_name}[Gene Name]"
        if not organism
        else f"({gene_name}[Gene Name]) AND {organism}[Organism]"
    )
    handle = Entrez.esearch(db="gene", term=query, retmax=max_gene_ids)
    record = Entrez.read(handle)
    handle.close()

    gene_ids = record["IdList"]
    if len(gene_ids) > 1:
        print("More than 1 gene ID returned for gene name")
    gene_id = gene_ids[0]

    # print(
    #     f"{len(gene_ids)} gene IDs returned associated with gene {gene_name}."
    # )
    # print(f"\tRetrieving summary for {gene_id}...")
    handle = Entrez.efetch(db="gene", id=gene_id, rettype="docsum")
    gene_dict = xmltodict.parse(
        "".join([x.decode(encoding="utf-8") for x in handle.readlines()]),
        dict_constructor=dict,
    )
    gene_docsum = gene_dict["eSummaryResult"]["DocumentSummarySet"][
        "DocumentSummary"
    ]
    summary = gene_docsum.get("Summary")
    handle.close()
    time.sleep(0.34)  # Requests to NCBI are rate limited to 3 per second

    return summary

