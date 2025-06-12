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
pip install PyYAML==6.0.2
pip install numba
```

# Results

| Model         | Yelp         |                | Gowalla      |                | Amazon-CD    |                |
|---------------|--------------|----------------|--------------|----------------|--------------|----------------|
|               | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        | Recall@20    | NDCG@20        |
| LightGCN      | 0.0957 ± 0.0004      | 0.0811 ± 0.0003    | 0.2103 ± 0.0011      | 0.1255 ± 0.0009    | 0.1475 ± 0.0009        | 0.0925 ± 0.0006      |
| + DNA         | 0.0998 ± 0.0004      | 0.0847 ± 0.0005    | 0.2097 ± 0.0008      | 0.1267 ± 0.0004    | 0.1526 ± 0.0007        | 0.0980 ± 0.0005      |
| Improv.       | +4.28%               | +4.43%             | −0.29%               | +0.96%             | +3.46%                 | +5.95%               |
||||
| IMPGCN        | ±                    | ±                  | ±                    | ±                  | ±                      | ±                    |
| + DNA         | ±                    | ±                  | ±                    | ±                  | ±                      | ±                    |
| Improv.       |                      |                    |                      |                    |                        |                      |
||||
| SimGCL        | ±                    | ±                  | ±                    | ±                  | ±                      | ±                    |
| + DNA         | ±                    | ±                  | ±                    | ±                  | ±                      | ±                    |
| Improv.       |                      |                    |                      |                    |                        |                      |
||||
| LayerGCN      | ±                    | ±                  | ±                    | ±                  | ±                      | ±                    |
| + DNA         | 0.1023 ± 0.0002      | 0.0877 ± 0.0001    | 0.2310 ± 0.0002      | 0.1415 ± 0.0002    | 0.1536 ± 0.0001        | 0.1001 ± 0.0001      |
| Improv.       |                      |                    |                      |                    |                        |                      |
||||
| XSimGCL       | ±                    | ±                  | ±                    | ±                  | ±                      | ±                    |
| + DNA         | ±                    | ±                  | ±                    | ±                  | ±                      | ±                    |
| Improv.       |                      |                    |                      |                    |                        |                      |


# Run
Instead of **[dataset]**, substitute **Amazon-CD**, **Gowalla**, **Yelp** to run the code.
##### DNA-LightGCN
```
python main_DNA-LightGCN.py --dataset [dataset]
```

# Compare with our results
The **results4comparison** folder contains the results of our experiment. Each file includes the loss and performance metrics for every epoch, as well as the hyperparameters, dataset statistics, and training time. You can compare our results with your own reproduced results.
