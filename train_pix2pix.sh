GPU_ID=1
MASK_CONSTRAINT=0

if [ "$#" -ne 0 ]
then
  GPU_ID=$1
  echo "Use ${GPU_ID}th GPU"
  MASK_CONSTRAINT=$2
fi
python train.py --dataroot ./datasets/facades --name semifacades_pix2pix_gpu_${GPU_ID}_-${NUM_STREAM} --model semipix2pix --which_direction BtoA --display_id -1 --gpu_ids ${GPU_ID} --use_area_constraint ${MASK_CONSTRAINT}
