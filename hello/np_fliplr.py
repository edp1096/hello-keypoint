import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

pil_im = Image.open("444.jpg").convert("RGB")
np_im = np.array(pil_im)
flip_im = np.fliplr(np_im)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(np_im)
ax2.imshow(flip_im)
plt.show()
