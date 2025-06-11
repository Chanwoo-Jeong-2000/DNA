# [RecSys'25, Under Review] GCNs Meet Long-Tail: Embedding Norm Bias in GCN-Based Recommendations

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
```

# Results

## Results on Yelp

| Model                                                     | Recall@20  | NDCG@20    |
|:--------------------------------------------------------- |:----------:|:----------:|
| LightGCN | 0.0957 ± 0.0004 |  ±  |
| LightGCN + DNA |  ±  |  ±  |
| Improv. |  |  |
||||
| IMPGCN |  ±  |  ±  |
| IMPGCN + DNA |  ±  |  ±  |
| Improv. |  |  | 
||||
| SimGCL |  ±  |  ±  |
| SimGCL + DNA |  ±  |  ±  |
| Improv. |  |  | 
||||
| LayerGCN |  ±  |  ±  |
| LayerGCN + DNA |  ±  |  ±  |
| Improv. |  |  | 
||||
| XSimGCL |  ±  |  ±  |
| XSimGCL + DNA |  ±  |  ±  |
| Improv. |  |  |

## Results on Gowalla

| Model                                                     | Recall@20  | NDCG@20    |
|:--------------------------------------------------------- |:----------:|:----------:|
| LightGCN |  |  |
| LightGCN + DNA |  |  |
| Improv. |  |  |
||||
| IMPGCN |  |  |
| IMPGCN + DNA |  |  |
| Improv. |  |  | 
||||
| SimGCL |  |  |
| SimGCL + DNA |  |  |
| Improv. |  |  | 
||||
| LayerGCN |  |  |
| LayerGCN + DNA |  |  |
| Improv. |  |  | 
||||
| XSimGCL |  |  |
| XSimGCL + DNA |  |  |
| Improv. |  |  |

## Results on Amazon-CD

| Model                                                     | Recall@20  | NDCG@20    |
|:--------------------------------------------------------- |:----------:|:----------:|
| LightGCN |  |  |
| LightGCN + DNA |  |  |
| Improv. |  |  |
||||
| IMPGCN |  |  |
| IMPGCN + DNA |  |  |
| Improv. |  |  | 
||||
| SimGCL |  |  |
| SimGCL + DNA |  |  |
| Improv. |  |  | 
||||
| LayerGCN |  |  |
| LayerGCN + DNA |  |  |
| Improv. |  |  | 
||||
| XSimGCL |  |  |
| XSimGCL + DNA |  |  |
| Improv. |  |  |


# Run
Instead of **[dataset]**, substitute **Amazon-CD**, **Gowalla**, **Yelp** to run the code.
##### DNA-LightGCN
```
python main_DNA-LightGCN.py --dataset [dataset]
```
