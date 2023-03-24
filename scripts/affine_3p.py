import cv2
import numpy as np
from PIL import Image

src = cv2.imread("data/sample/will_farrell.jpg", cv2.IMREAD_COLOR)

h_orig, w_orig, c_orig = src.shape
# w, h = 512, 512
w, h = 384, 384
scale_w, scale_h = w / w_orig, h / h_orig
src = cv2.resize(src, dsize=(w, h), interpolation=cv2.INTER_AREA)

# ml 데이터 값으로 바까야됨
eyeL = [348 * scale_w, 368 * scale_h]
eyeR = [496 * scale_w, 380 * scale_h]
nose = [430 * scale_w, 490 * scale_h]

# REFERENCE_FACIAL_POINTS = np.float32([(120, 150), (390, 150), (250, 320)])  # 512. left eye, right eye, nose
REFERENCE_FACIAL_POINTS = np.float32([(90, 112), (292, 112), (187, 240)])  # 384. left eye, right eye, nose

tmpl = (REFERENCE_FACIAL_POINTS).astype(np.int32)
data = np.zeros((h, w, 3), dtype=np.uint8)
for x, y in tmpl:
    data[y, x] = [255, 255, 255]
img = Image.fromarray(data, "RGB")
img.show()

srcPTs = np.array([eyeL, eyeR, nose], dtype=np.float32)
refPTs = REFERENCE_FACIAL_POINTS

aff = cv2.getAffineTransform(srcPTs, refPTs)
result = cv2.warpAffine(src, aff, (0, 0))  # 0,0 의미: 입력영상 크기와 동일한 출력 영상 생성

cv2.imshow("src", src)
cv2.imshow("dst", result)
cv2.waitKey()
cv2.destroyAllWindows()
