import cv2 as cv
import numpy as np

img = cv.imdecode(
    np.fromfile('F:/数据集/DIV2K/DIV2K_train_HR/0001.png', dtype=np.uint8),
    cv.IMREAD_COLOR)
print(img)