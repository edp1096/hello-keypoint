# DATASET_NAME = "dst"
# DATA_ROOT = "data/dst/train"
# DATA_TEST_ROOT = "data/dst/test"

DATASET_NAME = "humanface_keypoints"
DATA_ROOT = "data/vggface_keypoints_dst/train"
DATA_TEST_ROOT = "data/vggface_keypoints_dst/test"

# DATASET_NAME = "face_keypoints"
# DATA_ROOT = "data/face_keypoints_src/train"
# DATA_TEST_ROOT = "data/face_keypoints_src/test/images"

# MODEL_NAME = "resnet18"
MODEL_NAME = "resnet50"
# MODEL_NAME = "efficientnetv2_s"

# IMAGE_SIZE = 384
IMAGE_SIZE = 224

# POINT_NUM = 8  # 4 xy points. bounding box
POINT_NUM = 6  # 3 xy points. facial keypoints - left eye, right eye, nose

USE_AMP = True

OUTPUT_SAVE_ROOT = "weights"
COMMON_FILENAME = f"{OUTPUT_SAVE_ROOT}/{DATASET_NAME}_{MODEL_NAME}"
WEIGHT_FILE = f"{COMMON_FILENAME}.pt"
WEIGHT_INFO_FILE = f"{COMMON_FILENAME}_info.log"
SCATTER_FILE = f"{COMMON_FILENAME}_dist"
LOSS_RESULT_FILE = f"{COMMON_FILENAME}.png"


# BATCH_SIZE = 1920  # resnet18, 28
# BATCH_SIZE = 256  # resnet18, 96
# BATCH_SIZE = 128  # resnet18, 384
BATCH_SIZE = 64
# BATCH_SIZE = 96  # efficientnetv2_s, 96
# BATCH_SIZE = 32  # efficientnetv2_s, 224
# BATCH_SIZE = 13  # efficientnetv2_s, 384

EPOCHS_PRETRAIN = 20
EPOCHS = 50
LEARNING_RATE = 0.01

