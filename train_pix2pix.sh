GPU_ID=1
CLIP_SIZE=1

if [ "$#" -ne 0 ]
then
  GPU_ID=$1
  echo "Use ${GPU_ID}th GPU"
  CLIP_SIZE=$2
fi
python train.py --dataroot ./datasets/facades --name semifacades_pix2pix_gpu_${GPU_ID} --model semipix2pix --which_direction BtoA --display_id -1 --aligned_random_crop 1 --clip_size ${CLIP_SIZE} --gpu_ids ${GPU_ID}
