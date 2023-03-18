import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2

import os
import zipfile


src_root = "data/src"
save_root = "data/dst"
os.makedirs(save_root, exist_ok=True)

if not os.path.exists(f"{src_root}/training.csv"):
    with zipfile.ZipFile(f"{src_root}/training.zip", "r") as zip_ref:
        zip_ref.extractall(src_root)
if not os.path.exists(f"{src_root}/test.csv"):
    with zipfile.ZipFile(f"{src_root}/test.zip", "r") as zip_ref:
        zip_ref.extractall(src_root)

train_data = pd.read_csv(f"{src_root}/training.csv")
test_data = pd.read_csv(f"{src_root}/test.csv")

print("data shape")
print(train_data.shape, test_data.shape)
print("train data head")
print(train_data.head(3))
print("test data head")
print(test_data.head())
print("column info")
print(train_data.info())


def getImage(row):
    image_size = 96  # kaggle
    # image_size = 224  # my
    image = row["Image"]
    image = np.fromstring(image, sep=" ").reshape([image_size, image_size])

    return image


def getKeypoints(row):
    point_count = 15  # kaggle, 30 with Image
    # point_count = 13  # my, 26 with Image
    keypoints = pd.DataFrame(row).drop(["Image"], axis=0).values.reshape([point_count, 2])

    return keypoints


def showImage(row):
    image = getImage(row)
    plt.imshow(image, cmap="gray")
    plt.show()


def showImageWithKeypoints(row):
    image = getImage(row)
    keypoints = getKeypoints(row)
    print(keypoints[0])
    plt.imshow(image, cmap="gray")
    # plt.plot(keypoints[:, 0], keypoints[:, 1], "gx")
    plt.plot(keypoints[0][0], keypoints[0][1], "gx")  # right eye from watcher side
    plt.plot(keypoints[1][0], keypoints[1][1], "gx")  # left eye from watcher side
    plt.plot(keypoints[10][0], keypoints[10][1], "gx")  # nose
    plt.show()


def saveImage(row, path):
    image = getImage(row)
    cv2.imwrite(path, image)


selected_row = train_data.iloc[1]
# selected_row = train_data.iloc[2643]
showImageWithKeypoints(selected_row)
# showImage(selected_row)

os.makedirs(f"{save_root}/train", exist_ok=True)
os.makedirs(f"{save_root}/train/images", exist_ok=True)
os.makedirs(f"{save_root}/train/annotations", exist_ok=True)
os.makedirs(f"{save_root}/test", exist_ok=True)

train_data_count = len(train_data)
test_data_count = len(test_data)
nan_founds = []

print(f"train data count: {train_data_count}")
for i, data in enumerate(train_data.values):
    row = pd.Series(data, index=train_data.columns)

    keypoints_raw = getKeypoints(row)

    # Before reshape
    # 0   left_eye_center_x          7039 non-null   float64
    # 1   left_eye_center_y          7039 non-null   float64
    # 2   right_eye_center_x         7036 non-null   float64
    # 3   right_eye_center_y         7036 non-null   float64
    # 20  nose_tip_x                 7049 non-null   float64
    # 21  nose_tip_y                 7049 non-null   float64

    # After reshape - 3 points only. eyes locations are face owner side, not watcher side
    # 1  (2, 3)   -> 0  right_eye_center_x, right_eye_center_y
    # 0  (0, 1)   -> 1  left_eye_center_x, left_eye_center_y
    # 10 (20, 21) -> 2  nose_tip_x, nose_tip_y

    if keypoints_raw[1][0] > keypoints_raw[0][0]:
        print(f"Notice: the location of left eye is right side: {i}")

    # Change eyes from owner side to watcher side
    keypoints = [[
        keypoints_raw[1][0],
        keypoints_raw[1][1],
        keypoints_raw[0][0],
        keypoints_raw[0][1],
        keypoints_raw[10][0],
        keypoints_raw[10][1],
    ]]

    for j, k in enumerate(keypoints[0]):
        if math.isnan(k):
            print(f"nan detected: {i}")
            nan_founds.append(i)
            continue

    saveImage(row, f"{save_root}/train/images/{i}.jpg")
    np.savetxt(f"{save_root}/train/annotations/{i}.csv", keypoints, delimiter=",", fmt="%f")

    if i % 100 == 0 or i == train_data_count - 1:
        print(f"{i+1}/{train_data_count}")


print(f"test data count: {test_data_count}")
for i, data in enumerate(test_data.values):
    row = pd.Series(data, index=test_data.columns)
    saveImage(row, f"{save_root}/test/{i}.jpg")

    if i % 100 == 0 or i == test_data_count - 1:
        print(f"{i+1}/{test_data_count}")

print(f"nan count: {len(nan_founds)}")
print(f"nan founds: {nan_founds}")
print("done")
