import cv2 as cv

ori_img = cv.imread("./image/bicubic_contract_4.png")
gray_img = cv.cvtColor(ori_img, cv.COLOR_BGR2GRAY)
cv.imwrite("./image/bicubic_contract_4_gray.png", gray_img)