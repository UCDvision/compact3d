path_base=/home/navaneet/data/

ncls=4096
ncls_sh=512
ncls_dc=4096
kmeans_iters=10
st_iter=15000
max_iters=30000
output_base=output/exp_001
dset=tandt
scene=train
cuda_device=0
path_source="$path_base"/"$dset"/"$scene"
path_output="$output_base"/"$dset"/"$scene"

CUDA_VISIBLE_DEVICES=$cuda_device python train_kmeans.py \
  -s="$path_source" \
  -m="$path_output" \
  --kmeans_ncls "$ncls" \
  --kmeans_ncls_sh "$ncls_sh" \
  --kmeans_ncls_dc "$ncls_dc" \
  --kmeans_st_iter "$st_iter" \
  --kmeans_iters "$kmeans_iters" \
  --quant_params sh dc rot sc\
  --kmeans_freq 100 \
  --eval
