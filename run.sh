path_base=/home/navaneet/data/

ncls=4096
ncls_sh=512
ncls_dc=4096
kmeans_iters=10
st_iter=15000
max_iters=15100
output_base=output/exp_001
dset=tandt
scene=train
cuda_device=1
port=4060
ckpt=../../comp3d/my_gaussian-splatting/compact3d/output/e001_orig/"$dset"/"$scene"/chkpnt"$st_iter".pth
#ckpt=output/exp_001_noquant/"$dset"/"$scene"/chkpnt"$st_iter".pth
path_source="$path_base"/"$dset"/"$scene"
path_output="$output_base"/"$dset"/"$scene"

CUDA_VISIBLE_DEVICES=$cuda_device python train_kmeans.py \
  --port $port \
  -s="$path_source" \
  -m="$path_output" \
  --start_checkpoint "$ckpt" \
  --kmeans_ncls "$ncls" \
  --kmeans_ncls_sh "$ncls_sh" \
  --kmeans_ncls_dc "$ncls_dc" \
  --kmeans_st_iter "$st_iter" \
  --kmeans_iters "$kmeans_iters" \
  --total_iterations "$max_iters" \
  --quant_params sh dc rot scale\
  --kmeans_freq 100 \
  --eval
