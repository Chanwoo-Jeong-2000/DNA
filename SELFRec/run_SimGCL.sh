
# Yelp
CUDA_VISIBLE_DEVICES=0 python main.py --conf_path ./conf/SimGCL_yelp.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --conf_path ./conf/DNA_SimGCL_yelp.yaml

# Gowalla
CUDA_VISIBLE_DEVICES=0 python main.py --conf_path ./conf/SimGCL_gowalla.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --conf_path ./conf/DNA_SimGCL_gowalla.yaml

# Amazon
CUDA_VISIBLE_DEVICES=0 python main.py --conf_path ./conf/SimGCL_amazon.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --conf_path ./conf/DNA_SimGCL_amazon.yaml