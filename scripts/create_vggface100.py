import os, random
import pandas as pd
from PIL import Image
from shutil import copyfile

DATA_SRC_ROOT = "D:/dev/datasets/VGG-Face2"
# DATA_SRC_ROOT = "data/vggface_src"

DATA_DST_ROOT = "data/vggface_dst"
IMAGE_SRC_PATH = f"{DATA_SRC_ROOT}/data/train"
IMAGE_PATH = f"{DATA_DST_ROOT}/images"
ANNOTATION_PATH = f"{DATA_DST_ROOT}/annotations"

DATA_CROP_DST_ROOT = "data/vggface_crop_dst"
IMAGE_CROP_PATH = f"{DATA_CROP_DST_ROOT}/images"
ANNOTATION_CROP_PATH = f"{DATA_CROP_DST_ROOT}/annotations"

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
os.makedirs(IMAGE_PATH, exist_ok=True)
os.makedirs(ANNOTATION_PATH, exist_ok=True)
os.makedirs(DATA_CROP_DST_ROOT, exist_ok=True)
os.makedirs(IMAGE_CROP_PATH, exist_ok=True)
os.makedirs(ANNOTATION_CROP_PATH, exist_ok=True)

image_count = 0
for j, row in df_bbox.iterrows():
    person_id = row["NAME_ID"].split("/")[0]
    image_name = row["NAME_ID"].split("/")[1]

    if person_id not in selected_ids:
        continue

    # Prepare data
    copyfile(f"{IMAGE_SRC_PATH}/{person_id}/{image_name}.jpg", f"{IMAGE_PATH}/{image_count}.jpg")

    bbox = [row["X"], row["Y"], row["X"] + row["W"], row["Y"] + row["H"]]
    landmarks = df_landmark.iloc[j][["P1X", "P1Y", "P2X", "P2Y", "P3X", "P3Y"]].values
    merged_points = list(bbox) + list(landmarks)

    with open(f"{ANNOTATION_PATH}/{image_count}.csv", "w") as f:
        f.write(",".join([str(point) for point in merged_points]))

    # Prepare cropped data
    cropped_img = Image.open(f"{IMAGE_PATH}/{image_count}.jpg").crop(bbox)
    cropped_img.save(f"{IMAGE_CROP_PATH}/{image_count}.jpg")

    landmarks_crop = landmarks.copy()
    landmarks_crop[0::2] = landmarks_crop[0::2] - bbox[0]
    landmarks_crop[1::2] = landmarks_crop[1::2] - bbox[1]

    with open(f"{ANNOTATION_CROP_PATH}/{image_count}.csv", "a") as f:
        f.write(",".join([str(point) for point in landmarks_crop]))

    image_count += 1

    if image_count % 100 == 0:
        print("image_count:", image_count)

print("image_count:", image_count)
