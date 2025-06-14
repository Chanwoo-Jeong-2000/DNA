
# Yelp
CUDA_VISIBLE_DEVICES=0 python main.py -m DNA_LayerGCN -d yelp
CUDA_VISIBLE_DEVICES=0 python main.py -m LayerGCN -d yelp

# Gowalla
CUDA_VISIBLE_DEVICES=0 python main.py -m DNA_LayerGCN -d gowalla
CUDA_VISIBLE_DEVICES=0 python main.py -m LayerGCN -d gowalla

# Amazon
CUDA_VISIBLE_DEVICES=0 python main.py -m DNA_LayerGCN -d amazon-cd
CUDA_VISIBLE_DEVICES=0 python main.py -m LayerGCN -d amazon-cd