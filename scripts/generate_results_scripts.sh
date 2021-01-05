echo "2020-49144d_semipix2pix_facades"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix --name 2020-49144d_semipix2pix_facades --which_direction BtoA --gt_crop 1 --which_crop A --display_id -1 --load_data 1 --load_dir_suffix _save

echo "2020-49144d_semipix2pix_facades_gray"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix --name 2020-49144d_semipix2pix_facades_gray --which_direction BtoA --gt_crop 1 --which_crop A --crop_replace gray --display_id -1  --load_data 1 --load_dir_suffix _gray_save

echo "2020-49144d_semipix2pix_facades_area-constrains"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix --name 2020-49144d_semipix2pix_facades_area-constrains --which_direction BtoA --gt_crop 1 --which_crop A --use_area_constraint 1 --display_id -1  --load_data 1 --load_dir_suffix _save

echo "2020-49144d_semipix2pix_facades_gray_area-constrains"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix --name 2020-49144d_semipix2pix_facades_gray_area-constrains --which_direction BtoA --gt_crop 1 --which_crop A --crop_replace gray --use_area_constraint 1 --display_id -1  --load_data 1 --load_dir_suffix _gray_save

echo "2020-49144d_semipix2pix_facades_oneblock"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix --name 2020-49144d_semipix2pix_facades_oneblock --which_direction BtoA --gt_crop 1 --which_crop A  --random_crop random_oneblock_crop --display_id -1 --load_data 1 --load_dir_suffix _oneblock_save

echo "2020-49144d_semipix2pix_facades_oneblock_gray"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix --name 2020-49144d_semipix2pix_facades_oneblock_gray --which_direction BtoA --gt_crop 1 --which_crop A --crop_replace gray  --random_crop random_oneblock_crop --display_id -1 --load_data 1 --load_dir_suffix _oneblock_gray_save

echo "2020-49144d_semipix2pix_facades_oneblock_area-constrains"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix --name 2020-49144d_semipix2pix_facades_oneblock_area-constrains --which_direction BtoA --gt_crop 1 --which_crop A --use_area_constraint 1 --random_crop random_oneblock_crop --display_id -1 --load_data 1 --load_dir_suffix _oneblock_save

echo "2020-49144d_semipix2pix_facades_oneblock_gray_area-constrains"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix --name 2020-49144d_semipix2pix_facades_oneblock_gray_area-constrains --which_direction BtoA --gt_crop 1 --which_crop A --crop_replace gray --use_area_constraint 1  --random_crop random_oneblock_crop --display_id -1 --load_data 1 --load_dir_suffix _oneblock_gray_save 

echo "2020-93b4e2_semipix2pix2_facades"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix2 --name 2020-93b4e2_semipix2pix2_facades --which_direction BtoA --gt_crop 1 --which_crop A --display_id -1 --gpu_id 1 --load_data 1 --load_dir_suffix _save

echo "2020-93b4e2_semipix2pix2_facades_gray"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix2 --name 2020-93b4e2_semipix2pix2_facades_gray --which_direction BtoA --gt_crop 1 --which_crop A --crop_replace gray --display_id -1 --gpu_id 1 --load_data 1 --load_dir_suffix _gray_save

echo "2020-93b4e2_semipix2pix2_facades_oneblock"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix2 --name 2020-93b4e2_semipix2pix2_facades_oneblock --which_direction BtoA --gt_crop 1 --which_crop A  --random_crop random_oneblock_crop --display_id -1 --gpu_id 1 --load_data 1 --load_dir_suffix _oneblock_save

echo "2020-93b4e2_semipix2pix2_facades_oneblock_gray"
python test.py  --results_dir /2tdisk/home/shiyi/2020_results --dataroot /2tdisk/home/shiyi/2020_datasets/facades --checkpoints_dir /2tdisk/home/shiyi/2020_checkpoints  --model semipix2pix2 --name 2020-93b4e2_semipix2pix2_facades_oneblock_gray --which_direction BtoA --gt_crop 1 --which_crop A --crop_replace gray  --random_crop random_oneblock_crop --display_id -1 --gpu_id 1  --load_data 1 --load_dir_suffix _oneblock_gray_save