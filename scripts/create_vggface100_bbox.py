import os, random
import pandas as pd
from PIL import Image
from shutil import copyfile

DATA_SRC_ROOT = "D:/dev/datasets/VGG-Face2"
# DATA_SRC_ROOT = "data/vggface_src"

DATA_DST_ROOT = "data/vggface_dst"
IMAGE_SRC_PATH = f"{DATA_SRC_ROOT}/data/train"
IMAGE_TRAIN_PATH = f"{DATA_DST_ROOT}/train/images"
ANNOTATION_TRAIN_PATH = f"{DATA_DST_ROOT}/train/annotations"
IMAGE_TEST_PATH = f"{DATA_DST_ROOT}/test/images"
ANNOTATION_TEST_PATH = f"{DATA_DST_ROOT}/test/annotations"


person_limit = 100

selected_ids = []

df_bbox = pd.read_csv(f"{DATA_SRC_ROOT}/meta/bb_landmark/loose_bb_train.csv")
df_landmark = pd.read_csv(f"{DATA_SRC_ROOT}/meta/bb_landmark/loose_landmark_train.csv")

selected_ids = []

i = 0
while True:
    selected_item = random.choice(os.listdir(IMAGE_SRC_PATH))
    if selected_item in selected_ids:
        continue

    selected_ids.append(selected_item)
    i += 1

    if i >= person_limit:
        break

os.makedirs(DATA_DST_ROOT, exist_ok=True)
os.makedirs(IMAGE_TRAIN_PATH, exist_ok=True)
os.makedirs(ANNOTATION_TRAIN_PATH, exist_ok=True)
os.makedirs(IMAGE_TEST_PATH, exist_ok=True)
os.makedirs(ANNOTATION_TEST_PATH, exist_ok=True)

image_count = 0
person_count = 0
prev_person_id = ""
mode = "train"
for j, row in df_bbox.iterrows():
    person_id = row["NAME_ID"].split("/")[0]
    image_name = row["NAME_ID"].split("/")[1]

    if person_id not in selected_ids:
        continue

    if person_id != prev_person_id:
        person_count += 1
        if person_count >= person_limit * 0.8:
            mode = "test"

        prev_person_id = person_id

    # Prepare data
    target_image_path = IMAGE_TRAIN_PATH
    target_annotation_path = ANNOTATION_TRAIN_PATH
    if mode == "test":
        target_image_path = IMAGE_TEST_PATH
        target_annotation_path = ANNOTATION_TEST_PATH

    copyfile(f"{IMAGE_SRC_PATH}/{person_id}/{image_name}.jpg", f"{target_image_path}/{image_count}.jpg")

    bbox = [row["X"], row["Y"], row["X"] + row["W"], row["Y"] + row["H"]]

    with open(f"{target_annotation_path}/{image_count}.csv", "w") as f:
        f.write(",".join([str(point) for point in bbox]))

    image_count += 1

    if image_count % 100 == 0:
        print("image_count:", image_count)

print("image_count:", image_count)
