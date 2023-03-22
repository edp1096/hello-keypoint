import os, random
import pandas as pd
from PIL import Image
from shutil import copyfile

DATA_SRC_ROOT = "D:/dev/datasets/VGG-Face2"
# DATA_SRC_ROOT = "data/vggface_src"

IMAGE_SRC_PATH = f"{DATA_SRC_ROOT}/data/train"

DATA_CROP_DST_ROOT = "data/vggface_keypoints_dst"
IMAGE_CROP_TRAIN_PATH = f"{DATA_CROP_DST_ROOT}/train/images"
ANNOTATION_CROP_TRAIN_PATH = f"{DATA_CROP_DST_ROOT}/train/annotations"
# IMAGE_CROP_TEST_PATH = f"{DATA_CROP_DST_ROOT}/test/images"
# ANNOTATION_CROP_TEST_PATH = f"{DATA_CROP_DST_ROOT}/test/annotations"
IMAGE_CROP_TEST_PATH = f"{DATA_CROP_DST_ROOT}/test"

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

os.makedirs(DATA_CROP_DST_ROOT, exist_ok=True)
os.makedirs(IMAGE_CROP_TRAIN_PATH, exist_ok=True)
os.makedirs(ANNOTATION_CROP_TRAIN_PATH, exist_ok=True)
# os.makedirs(IMAGE_CROP_TEST_PATH, exist_ok=True)
# os.makedirs(ANNOTATION_CROP_TEST_PATH, exist_ok=True)
os.makedirs(IMAGE_CROP_TEST_PATH, exist_ok=True)

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

        if mode != "test" and person_count >= person_limit * 0.8:
            mode = "test"
            image_count = 0

        prev_person_id = person_id

    # Prepare data
    target_image_path = IMAGE_CROP_TRAIN_PATH
    target_annotation_path = ANNOTATION_CROP_TRAIN_PATH
    if mode == "test":
        target_image_path = IMAGE_CROP_TEST_PATH

    # Make square bounding box
    if row["W"] > row["H"]:
        row["Y"] -= (row["W"] - row["H"]) / 2
        row["H"] = row["W"]
    elif row["W"] < row["H"]:
        row["X"] -= (row["H"] - row["W"]) / 2
        row["W"] = row["H"]

    bbox = [row["X"], row["Y"], row["X"] + row["W"], row["Y"] + row["H"]]
    landmarks = df_landmark.iloc[j][["P1X", "P1Y", "P2X", "P2Y", "P3X", "P3Y"]].values

    # Prepare cropped image
    cropped_img = Image.open(f"{IMAGE_SRC_PATH}/{person_id}/{image_name}.jpg").crop(bbox)
    cropped_img = cropped_img.resize((96, 96))
    cropped_img.save(f"{target_image_path}/{image_count}.jpg")

    if mode == "test":
        image_count += 1

        if image_count % 100 == 0:
            print(f"{mode} image_count:", image_count)

        continue

    landmarks_crop = landmarks.copy()
    landmarks_crop[0::2] = landmarks_crop[0::2] - bbox[0]
    landmarks_crop[1::2] = landmarks_crop[1::2] - bbox[1]

    # correct landmarks to fit 96x96 image
    landmarks_crop[0::2] = landmarks_crop[0::2] * 96 / row["W"]
    landmarks_crop[1::2] = landmarks_crop[1::2] * 96 / row["H"]

    with open(f"{target_annotation_path}/{image_count}.csv", "a") as f:
        f.write(",".join([str(point) for point in landmarks_crop]))

    image_count += 1

    if image_count % 100 == 0:
        print(f"{mode} image_count:", image_count)

print("Done")
