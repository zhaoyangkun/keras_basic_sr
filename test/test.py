import cv2 as cv
import numpy as np

img = cv.imdecode(
    np.fromfile('./image/head.jpg', dtype=np.uint8),
    cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite('./image/head_gray.jpg', img)