
# Yelp
CUDA_VISIBLE_DEVICES=1 python main.py --conf_path ./conf/XSimGCL_yelp.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --conf_path ./conf/DNA_XSimGCL_yelp.yaml

# Gowalla
CUDA_VISIBLE_DEVICES=1 python main.py --conf_path ./conf/XSimGCL_gowalla.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --conf_path ./conf/DNA_XSimGCL_gowalla.yaml

# Amazon
CUDA_VISIBLE_DEVICES=1 python main.py --conf_path ./conf/XSimGCL_amazon.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --conf_path ./conf/DNA_XSimGCL_amazon.yaml