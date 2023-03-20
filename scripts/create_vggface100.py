import os, random
import pandas as pd
from shutil import copyfile

DATA_SRC_ROOT = "D:/dev/datasets/VGG-Face2"
# DATA_SRC_ROOT = "data/vggface_src"
DATA_DST_ROOT = "data/vggface_dst"
IMAGE_SRC_PATH = f"{DATA_SRC_ROOT}/data/train"
IMAGE_DST_PATH = f"{DATA_DST_ROOT}/images"
ANNOTATION_PATH = f"{DATA_DST_ROOT}/annotations"

person_limit = 10

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
os.makedirs(IMAGE_DST_PATH, exist_ok=True)
os.makedirs(ANNOTATION_PATH, exist_ok=True)

image_count = 0
for j, row in df_bbox.iterrows():
    person_id = row["NAME_ID"].split("/")[0]
    image_name = row["NAME_ID"].split("/")[1]

    if person_id not in selected_ids:
        continue

    copyfile(f"{IMAGE_SRC_PATH}/{person_id}/{image_name}.jpg", f"{IMAGE_DST_PATH}/{image_count}.jpg")

    bbox = [row["X"], row["Y"], row["X"] + row["W"], row["Y"] + row["H"]]
    landmarks = df_landmark.iloc[j][["P1X", "P1Y", "P2X", "P2Y", "P3X", "P3Y"]].values
    # correct landmarks from bbox
    print("Original:", landmarks[0], landmarks[1], bbox[0], bbox[1])
    # landmarks[0::2] = landmarks[0::2] - bbox[0]
    # landmarks[1::2] = landmarks[1::2] - bbox[1]
    merged_points = list(bbox) + list(landmarks)
    print("Changed:", landmarks[0], landmarks[1], bbox[0], bbox[1])

    with open(f"{ANNOTATION_PATH}/{image_count}.csv", "w") as f:
        f.write(",".join([str(point) for point in bbox]))

    image_count += 1

print("image_count:", image_count)
