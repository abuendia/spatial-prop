# %%
aging_coronal = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/gnn_datasets/aging_coronal_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_TNP_NoneInject"
benchmark_base = "/oak/stanford/groups/akundaje/abuen/spatial/spatial-gnn/data/gnn_datasets/benchmark_base_expression_100per_2hop_2C0aug_200delaunay_expressionFeat_TNP_NoneInject"
# %%

# read in .pt files
aging_coronal_train = f"{aging_coronal}/train"
aging_coronal_test = f"{aging_coronal}/test"
benchmark_base_train = f"{benchmark_base}/train"
benchmark_base_test = f"{benchmark_base}/test"
# %%
import os 
import torch 
from tqdm import tqdm 
# read in files 

# %%

all_test_data_aging_coronal = []
for f in tqdm(os.listdir(aging_coronal_test)):
    all_test_data_aging_coronal.append(torch.load(os.path.join(aging_coronal_test, f), weights_only=False))
# %%
len(all_test_data_aging_coronal)
# %%

all_test_data_benchmark_base = []
for f in tqdm(os.listdir(benchmark_base_test)):
    if f.endswith(".pt"):
        batch_list = torch.load(os.path.join(benchmark_base_test, f), weights_only=False)
        all_test_data_benchmark_base.extend(batch_list)
# %%
all_test_data_benchmark_base[0]
# %%
all_test_data_aging_coronal[0]
# %%
len(all_test_data_benchmark_base)
# %%
len(all_test_data_aging_coronal)
# %%
all_test_data_benchmark_base[0]
# %%
