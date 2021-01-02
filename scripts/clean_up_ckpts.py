#
# run `sudo python scripts/clean_up_ckpts.py /home/yishi/pytorch-CycleGAN-and-pix2pix/checkpoints/`
#


import os
import sys

root_path = sys.argv[1]

def delete_ckpts(current_path):
    file_names = os.listdir(current_path)
    ckpt_names = [file_name for file_name in file_names if file_name.endswith(".pth") and not file_name.startswith("latest")]
    if len(ckpt_names) > 0:
        for ckpt_name_to_delete in ckpt_names:
            ckpt_path_to_delete = os.path.join(current_path, ckpt_name_to_delete)
            os.remove(ckpt_path_to_delete)
        return
    else:
        # Go one level lower.
        for file_name in file_names:
            path = os.path.join(current_path, file_name)
            if os.path.isdir(path):
                delete_ckpts(path)

delete_ckpts(root_path)
