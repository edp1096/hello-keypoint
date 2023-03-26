import cv2
import numpy as np
from PIL import Image
import os


def alignFace(using_type, fpath, eyeL, eyeR, nose):
    # src = cv2.imread("data/sample/will_farrell.jpg", cv2.IMREAD_COLOR)
    src = cv2.imread(fpath, cv2.IMREAD_COLOR)

    h_orig, w_orig, c_orig = src.shape
    # w, h = 512, 512
    w, h = 384, 384
    scale_w, scale_h = w / w_orig, h / h_orig
    src = cv2.resize(src, dsize=(w, h), interpolation=cv2.INTER_AREA)

    # ml 데이터 값으로 바까야됨
    # eyeL = [348 * scale_w, 368 * scale_h]
    # eyeR = [496 * scale_w, 380 * scale_h]
    # nose = [430 * scale_w, 490 * scale_h]
    eyeL = [eyeL[0] * scale_w, eyeL[1] * scale_h]
    eyeR = [eyeR[0] * scale_w, eyeR[1] * scale_h]
    nose = [nose[0] * scale_w, nose[1] * scale_h]

    # REFERENCE_FACIAL_POINTS = np.float32([(120, 150), (390, 150), (250, 320)])  # 512. left eye, right eye, nose
    REFERENCE_FACIAL_POINTS = np.float32([(90, 112), (292, 112), (187, 240)])  # 384. left eye, right eye, nose

    # tmpl = (REFERENCE_FACIAL_POINTS).astype(np.int32)
    # data = np.zeros((h, w, 3), dtype=np.uint8)
    # for x, y in tmpl:
    #     data[y, x] = [255, 255, 255]
    # img = Image.fromarray(data, "RGB")
    # img.show()

    srcPTs = np.array([eyeL, eyeR, nose], dtype=np.float32)
    refPTs = REFERENCE_FACIAL_POINTS

    aff = cv2.getAffineTransform(srcPTs, refPTs)
    result = cv2.warpAffine(src, aff, (0, 0))  # 0,0 의미: 입력영상 크기와 동일한 출력 영상 생성

    # cv2.imshow("src", src)
    # cv2.imshow("dst", result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # save image
    fname, fext = os.path.splitext(os.path.basename(fpath))
    dst_fpath = os.path.join(dst_root, using_type, fname + ".jpg")
    cv2.imwrite(dst_fpath, result)


src_root = "data/face_keypoints_src"
dst_root = "data/face_keypoints_dst"

os.makedirs(dst_root, exist_ok=True)
os.makedirs(os.path.join(dst_root, "train"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "test"), exist_ok=True)

for src_sub_root in os.listdir(src_root):
    src_path = os.path.join(src_root, src_sub_root)
    image_root = os.path.join(src_path, "images")
    annotation_root = os.path.join(src_path, "annotations")

    for image_file in os.listdir(image_root):
        fname, fext = os.path.splitext(image_file)
        im_fpath = os.path.join(image_root, image_file)
        an_fpath = os.path.join(annotation_root, fname + ".csv")

        keypoints = np.loadtxt(an_fpath, delimiter=",", dtype=np.float32).reshape(-1, 2)
        eyeL, eyeR, nose = keypoints[0], keypoints[1], keypoints[2]

        alignFace(src_sub_root, im_fpath, eyeL, eyeR, nose)
