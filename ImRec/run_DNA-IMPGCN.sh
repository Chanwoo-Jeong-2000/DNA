
# Yelp
CUDA_VISIBLE_DEVICES=0 python main.py -m DNA_IMP_GCN -d yelp
CUDA_VISIBLE_DEVICES=0 python main.py -m IMP_GCN -d yelp

# Gowalla
CUDA_VISIBLE_DEVICES=0 python main.py -m DNA_IMP_GCN -d gowalla
CUDA_VISIBLE_DEVICES=0 python main.py -m IMP_GCN -d gowalla

# Amazon
CUDA_VISIBLE_DEVICES=0 python main.py -m DNA_IMP_GCN -d amazon-cd
CUDA_VISIBLE_DEVICES=0 python main.py -m IMP_GCN -d amazon-cd