# %%
ontology_file = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/causal/mgi.gaf"
# %%
import pandas as pd

go_df = pd.read_csv(ontology_file, header=None, sep="\t", comment="!", dtype=str)
# %%
go_df.columns = ["DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID", "DB:Reference", "Evidence", "With", "Aspect", "DB_Object_Name", "Synonym", "DB_Object_Type", "Taxon", "Date", "Assigned_By", "Annotation_Extension", "Gene_Product_Form_ID"]
# %%
go_df["Qualifier"].value_counts()
# %%
go_df["Annotation_Extension"].value_counts()
# %%
go_df_bp = go_df[go_df["Aspect"] == "P"]
# %%
from goatools.obo_parser import GODag

# Load the GO database
go_obo_file = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/repos/spatial-gnn/scripts/data/go.obo"
godag = GODag(go_obo_file)

# Map GO terms to natural language descriptions
go_descriptions = {go: godag[go].name for go in godag}

# Print results
for go, description in go_descriptions.items():
    print(f"{go}: {description}")
# %%
go_descriptions

# %%
go_df_bp
# %%
# add description column
go_df_bp["GO_Description"] = go_df_bp["GO_ID"].map(go_descriptions)
# %%
go_df_bp["GO_Description"].value_counts()
# %%

# filter df to only include rows with "production" or "response" in Description
go_df_bp_filtered = go_df_bp[go_df_bp["GO_Description"].str.contains("production|response", case=False)]
# %%
go_df_bp_filtered["GO_Description"].value_counts()
# %%
# plot distribution of GO terms
import matplotlib.pyplot as plt

go_df_bp_filtered["GO_Description"].value_counts().plot(kind="bar")

# %%
gene_list_path = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/gnn_model/gene_list300.txt"
gene_list = pd.read_csv(gene_list_path, header=None, sep="\t")

# %%
# convert to list
gene_list = [item.upper() for item in gene_list[0].tolist()]
# %%

go_df_bp_filtered['DB_Object_Symbol'] = go_df_bp_filtered['DB_Object_Symbol'].str.upper()
# %%

# filter df to only include rows with genes in gene_list
go_df_bp_filtered_gnn = go_df_bp_filtered[go_df_bp_filtered["DB_Object_Symbol"].isin(gene_list)]
# %%
import pyreadr

# %%

# Load the .rda file
cellchat_mouse = pyreadr.read_r("/oak/stanford/groups/jamesz/abuen/spatial-rotation/repos/CellChat/data/CellChatDB.human.rda")
# %%
cellchat_mouse.keys()
# %%
# load json
import json

coronal_embeds = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/coronal_embeds_all_geneptw.json"
reprogram_embeds = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/reprogramming_embeds_all_geneptw.json"
exercise_embeds = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/data/merfish/genept_embeds/exercise_embeds_all_geneptw.json"

# %%
with open(coronal_embeds, "rb") as f:
    coronal_embeds_dict = json.load(f)
# %%
with open(reprogram_embeds, "rb") as f:
    reprogram_embeds_dict = json.load(f)
# %%
with open(exercise_embeds, "rb") as f:
    exercise_embeds_dict = json.load(f)

# %%
