import os
import numpy as np
import shutil

# SOURCE_DIR = "data/src"
SOURCE_DIR = "D:/dev/datasets/1_merge_nemo"
TARGET_DIR = "data/face_nemo"

os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(f"{TARGET_DIR}/train", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/train/images", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/train/annotations", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/test", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/test/images", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/test/annotations", exist_ok=True)

# image_limit_per_subdir = 30
image_limit_per_subdir = 20
image_train_count = 0
image_test_count = 0
for pet_class in os.listdir(SOURCE_DIR):
    print(f"{pet_class}:")
    dir = SOURCE_DIR + "/" + pet_class
    class_name = pet_class.replace("new_", "")

    sub_total_cnt = 0
    # get random images
    flist = np.array(os.listdir(dir))
    image_flist = [file for file in flist if file.split(".")[-1] == "jpg" or file.split(".")[-1] == "png"]
    count_max = min(int(len(image_flist) / 2), image_limit_per_subdir)
    random_flist = np.random.choice(image_flist, count_max, replace=False)

    total_count = len(random_flist)
    train_count = int(total_count * 0.8)
    test_count = int(total_count * 0.2)

    # for file in np.array(os.listdir(dir)):
    for file in random_flist:
        fname = file.split(".")[0]
        ext = file.split(".")[-1]

        sub_total_cnt += 1

        if ext != "jpg" and ext != "png":
            continue

        if not os.path.exists(f"{dir}/{fname}.{ext}") or not os.path.exists(f"{dir}/{fname}.json"):
            print(f"missing file: {fname}.{ext} or {fname}.json")
            continue

        if sub_total_cnt <= train_count:
            target_root = f"{TARGET_DIR}/train"
            shutil.copy(f"{dir}/{fname}.{ext}", f"{target_root}/images/{image_train_count}.{ext}")
            shutil.copy(f"{dir}/{fname}.json", f"{target_root}/annotations/{image_train_count}.json")
            image_train_count += 1
        else:
            target_root = f"{TARGET_DIR}/test"
            shutil.copy(f"{dir}/{fname}.{ext}", f"{target_root}/images/{image_test_count}.{ext}")
            shutil.copy(f"{dir}/{fname}.json", f"{target_root}/annotations/{image_test_count}.json")
            image_test_count += 1

    print("train count, test count:", train_count, test_count)

print("All done")
