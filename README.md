# [Under Review] GCNs Meet Long-Tail: Embedding Norm Bias in GCN-Based Recommendations

**Note that sometimes, due to connectivity issues with Git anonymous access, certain files may not be activated properly. In such cases, downloading the files will allow you to view them.**

# Requirements
python 3.8.18, cuda 11.8, and the following installations:
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-geometric
pip install six
pip install pandas
pip install PyYAML==6.0.2
pip install numba
```

# Run

##### DNA-LightGCN and LightGCN
```
cd LightGCN
sh run_DNA-LightGCN.sh
```

##### DNA-IMPGCN and IMPGCN
```
cd IMRec
sh run_DNA-IMPGCN.sh
```

##### DNA-LayerGCN and LayerGCN
```
cd IMRec
sh run_DNA-LayerGCN.sh
```

##### DNA-SimGCL and SimGCL
```
cd SELFRec
sh run_DNA-SimGCL.sh
```

##### DNA-XSimGCL and XSimGC
```
cd SELFRec
sh run_DNA-XSimGCL.sh
```

# Compare with our results
The **results4comparison** folder contains the results of our experiment.
Each file includes the loss and performance metrics for every epoch, as well as the hyperparameters, dataset statistics, and training time.
You can compare our results with your own reproduced results.

# Settings
All benchmark models are implemented according to the configurations outlined in their respective original papers. The scaling factor α in our proposed methodology is tuned within the range [1.0, 5.0] using a step size of 0.5. The experiments are conducted using a single NVIDIA GeForce RTX 2080 Ti GPU.
