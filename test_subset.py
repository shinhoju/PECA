import pandas as pd
import os
import shutil

df = pd.read_csv("/home/oem/eccv/XY-Restormer/test_subset.csv")
rows = df.to_dict("records")
src = "/home/oem/eccv/benchmark_720/test"
tar = "/home/oem/eccv/datasets/test/benchmark_720_subset"

for row in rows:
    scene_name, frame_id = row['scene_name'], row['frame_num']
    img_name = scene_name + "_" + f"{frame_id:05d}.png"
    lr_file = os.path.join(src, 'input', img_name)
    hr_file = os.path.join(src, 'target', img_name)
    gd_file = os.path.join(src, 'guide', img_name)
    
    shutil.copy(lr_file, os.path.join(tar, 'input', img_name))
    shutil.copy(hr_file, os.path.join(tar, 'target', img_name))
    shutil.copy(gd_file, os.path.join(tar, 'guide', img_name))