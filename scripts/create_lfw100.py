import os, random
from shutil import copyfile

DATA_SRC_ROOT = "data/lfw_src"
DATA_DST_ROOT = "data/lfw_dst"

selected_items = []

os.makedirs(DATA_DST_ROOT, exist_ok=True)

for item in os.listdir(DATA_DST_ROOT):
    os.remove(f"{DATA_DST_ROOT}/{item}")

i = 0
while True:
    selected_item = random.choice(os.listdir(DATA_SRC_ROOT))
    if selected_item in selected_items:
        continue

    selected_items.append(selected_item)
    i += 1

    if i >= 100:
        break

for j, selected_item in enumerate(selected_items):
    selected_image = random.choice(os.listdir(f"{DATA_SRC_ROOT}/{selected_item}"))
    ext = selected_image.split(".")[-1]

    copyfile(f"{DATA_SRC_ROOT}/{selected_item}/{selected_image}", f"{DATA_DST_ROOT}/{j}.{ext}")

    print(f"{j+1}/{len(selected_items)}: {selected_item}/{selected_image}")
