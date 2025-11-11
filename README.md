# [Applied Soft Computing] GCNs Meet Long-Tail: Embedding Norm Bias in GCN-Based Recommendations

[![View Paper](https://img.shields.io/badge/View%20Paper-PDF-red?logo=adobeacrobatreader)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5412649)

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

# Citation
If you find DNA useful for your research or development, please cite the following our papers:
```
@article{choi5412646gcns,
  title={GCNs Meet Long-Tail: Embedding Norm Bias in GCN-Based Recommendations},
  author={Choi, Yeo Jun and Yun, Woo-Seong and Jeong, Chanwoo and Cho, Yoon-Sik},
  journal={Available at SSRN 5412646}
}
```

# Acknowledgments
This work was partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025- 00553785) and the Institute of Information and Communications Technology Planning and Evaluation (IITP) grant funded by the Korean government (MSIT) (RS-2021-II211341, Artificial Intelligence Graduate School Program of Chung-Ang University).
