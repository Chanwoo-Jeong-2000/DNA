# [RecSys'25, Under Review] GCNs Meet Long-Tail: Embedding Norm Bias in GCN-Based Recommendations

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

##### DNA-LightGCN and LightGCN (original)
```
cd LightGCN
sh run_DNA-LightGCN.sh
```

##### DNA-IMPGCN and IMPGCN (original)
```
cd IMRec
sh run_DNA-IMPGCN.sh
```

##### DNA-LayerGCN and LayerGCN (original)
```
cd IMRec
sh run_DNA-LayerGCN.sh
```

##### DNA-SimGCL and SimGCL (original)
```
cd SELFRec
sh run_DNA-SimGCL.sh
```

##### DNA-XSimGCL and XSimGCL (original)
```
cd SELFRec
sh run_DNA-XSimGCL.sh
```

# Main Results
Enhanced version of Table 1 from the main paper, including mean and standard deviation across 5 independent runs.
All improvements are statistically significant based on paired t-tests (p ≤ 0.05).

| Model         | Yelp         |                | Gowalla      |                | Amazon-CD    |                |
|---------------|--------------|----------------|--------------|----------------|--------------|----------------|
|               | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        |
| LightGCN      | 0.0957 ± 0.0004      | 0.0811 ± 0.0003    | 0.2092 ± 0.0001     | 0.1246 ± 0.0001   | 0.1475 ± 0.0009        | 0.0925 ± 0.0006      |
| + DNA         | 0.0998 ± 0.0004      | 0.0847 ± 0.0005    | 0.2095 ± 0.0001     | 0.1266 ± 0.0002   | 0.1526 ± 0.0007        | 0.0980 ± 0.0005      |
| Improv.       | +4.28%               | +4.44%             | +0.14%               | +1.61%             | +3.46%                 | +5.95%               |
||||
| IMPGCN        | 0.0869 ± 0.0000                   | 0.0735 ± 0.0000                 | 0.2045 ± 0.0000                   | 0.1237 ± 0.0000                 | 0.1345 ± 0.0000                     | 0.0837 ± 0.0000                   |
| + DNA         | 0.0906 ± 0.0000                   | 0.0771 ± 0.0000                 | 0.2140 ± 0.0000                   | 0.1307 ± 0.0001                 | 0.1399 ± 0.0000                     | 0.0898 ± 0.0000                   |
| Improv.       | +4.26%                     | +4.90%                   | +4.65%                     | +5.66%                   | +4.01%                       | +7.29%                     |
||||
| SimGCL        | 0.1087 ± 0.0001                   | 0.0940 ± 0.0005                 | 0.2259 ± 0.0005                   | 0.1382 ± 0.0003                 | 0.1576 ± 0.0006                     | 0.1007 ± 0.0000                   |
| + DNA         | 0.1097 ± 0.0004                   | 0.0943 ± 0.0008                 | 0.2382 ± 0.0007                   | 0.1457 ± 0.0003                 | 0.1700 ± 0.0005                     | 0.1115 ± 0.0003                   |
| Improv.       | +0.92%                     | +0.11%                   | +5.44%                     | +5.43%                   | +7.87%                       | +10.72%                     |
||||
| LayerGCN      | 0.0965 ± 0.0000                   | 0.0811 ± 0.0000                 | 0.1980 ± 0.0000                   | 0.1163 ± 0.0000                 | 0.1420 ± 0.0000                     | 0.0884 ± 0.0000                   |
| + DNA         | 0.1023 ± 0.0002      | 0.0877 ± 0.0001    | 0.2310 ± 0.0002      | 0.1415 ± 0.0002    | 0.1536 ± 0.0001        | 0.1001 ± 0.0001      |
| Improv.       | +6.01%                     | +8.14%                   | +16.67%                     | +21.67%                   | +8.17	%                       | +13.24%                     |
||||
| XSimGCL       | 0.1089 ± 0.0002                   | 0.0938 ± 0.0001                 | 0.2294 ± 0.0006                   | 0.1399 ± 0.0001                 | 0.1579 ± 0.0004                     | 0.1008 ± 0.0004                   |
| + DNA         | 0.1106 ± 0.0003                   | 0.0945 ± 0.0002                 | 0.2405 ± 0.0010                   | 0.1473 ± 0.0004                 | 0.1698 ± 0.0007                     | 0.1121 ± 0.0004                   |
| Improv.       | +1.56%                     | +0.75%                   | +4.84%                     | +5.29%                   | +7.54%                       | +11.21%                     |

# Additional Experiments Addressing Reviewers' Concerns

#### Comparison with Cosine Similarity Baseline (For reviewer 1, 2, and 4)
We compare DNA against cosine similarity baselines using LightGCN as the backbone. We test three configurations:
- Cosine similarity: Standard cosine similarity (normalizing both user and item embeddings)
- Cosine similarity + β: Cosine similarity with an additional learnable bias term β
- DNA-LightGCN: Item normalization with learnable β, preserving user magnitudes
  
Specifically:
- Cosine similarity: ŷ_ui = cos(θ_ui)
- Cosine similarity + β: ŷ_ui = cos(θ_ui) + β_i
- DNA: ŷ_ui = e_u^T · e_i^norm + β_i

|  | Yelp || Gowalla || Amazon-CD ||
|-|-|-|-|-|-|-|
|               | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        |
|Cosine Similarity w/o $\beta$ |0.0836|0.0697|0.1382|0.0740|0.1353|0.0836|
|Cosine Similarity w/ $\beta$ |0.0876|0.0749|0.1767|0.1067|0.1377|0.0869|
|DNA-LightGCN|**0.0993**|**0.0839**|**0.2097**|**0.1268**|**0.1535**|**0.0989**|

#### Impact of GCN Layer Depth on DNA Performance (For reviewer 3)
We investigate how DNA's effectiveness changes with varying GCN depths (1-6 layers) using LightGCN as the base model.

|# layers|| Yelp || Gowalla || Amazon-CD ||
|-|-|-|-|-|-|-|-|
|             |  | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        |
|1 layers|LightGCN |0.0852|0.0716|0.2045|0.1224|0.1317|0.0815|
||DNA-LightGCN|0.0928|0.0781|0.2047|0.1232|0.1412|0.0900|
|2 layers|LightGCN |0.0911|0.0772|0.2116|0.1269|0.1428|0.0895|
||DNA-LightGCN|0.0917|0.0782|0.2019|0.1233|0.1422|0.0912|
|3 layers|LightGCN |0.0952|0.0806|0.2090|0.1245|0.1471|0.0920|
||DNA-LightGCN|0.0993|0.0839|0.2097|0.1268|0.1535|0.0989|
|4 layers|LightGCN |0.0988|0.0839|0.2044|0.1208|0.1516|0.0951|
||DNA-LightGCN|0.0944|0.0807|0.2002|0.1217|0.1466|0.0939|
|5 layers|LightGCN |0.0995|0.0846|||0.1518|0.0949|
||DNA-LightGCN|0.0996|0.0844|0.2036|0.1225|0.1570|0.1005|
|6 layers|LightGCN |0.0989|0.0839|||0.1509|0.0934|
||DNA-LightGCN|0.0924|0.0791|||0.1486|0.0948|

#### Comparison with State-of-the-Art Debiasing Method (For reviewer 4)
We compare DNA-XSimGCL (our best configuration) against Adap-τ, a recent debiasing method that modifies the loss function to sampled softmax (SSM) with global temperature adjustments.

|  | Yelp || Gowalla || Amazon-CD ||
|-|-|-|-|-|-|-|
|               | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        |
|XSimGCL |0.1089|0.0938|0.2294|0.1399|0.1579|0.1008|
|Adap-τ |0.1052|0.0908|0.2329|0.1408|0.1632|0.1049|
|DNA-XSimGCL|**0.1106**|**0.0945**|**0.2405**|**0.1473**|**0.1698**|**0.1121**|
|Improv.|||||||


# Settings
All benchmark models are implemented according to the configurations outlined in their respective original papers. The scaling factor α in our proposed methodology is tuned within the range [1.0, 5.0] using a step size of 0.5. The experiments are conducted using a single NVIDIA GeForce RTX 2080 Ti GPU.
