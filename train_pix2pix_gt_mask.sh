GPU_ID=0
CLIP_SIZE=1

if [ "$#" -ne 0 ]
then
  GPU_ID=$1
  echo "Use ${GPU_ID}th GPU"
  CLIP_SIZE=$2
  NUM_STREAM=$3
  MASK_CONSTRAINT=$4
  GT_MASK=$5
  GATE_ON_B=$6
fi

python train.py --dataroot ./datasets/facades --checkpoints_dir --name semifacades_pix2pix_gpu_${GPU_ID}_stream_${NUM_STREAM} --model semipix2pix --which_direction BtoA --display_id -1 --aligned_random_crop 1 --clip_size ${CLIP_SIZE} --gpu_ids ${GPU_ID} --num_stream ${NUM_STREAM} --add_constraint ${MASK_CONSTRAINT} --use_gt_mask ${GT_MASK} --gate_on_gt ${GATE_ON_GT}
