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
training_results = "/oak/stanford/groups/jamesz/abuen/spatial-rotation/repos/spatial-gnn/scripts/results/gnn/expression_100per_2hop_2C0aug_200delaunay_expressionFeat_TNP_NoneInject/weightedl1_1en04/training.pkl"

# %%
# load pickle
import pickle

with open(training_results, "rb") as f:
    training = pickle.load(f)

# %%
