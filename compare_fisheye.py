from utility import resize_rgb
import cv2

rgb = cv2.imread("./2026-06-24-190115.jpg")
rgb = resize_rgb(rgb)
print(rgb.shape)
cv2.imwrite("color_224.png", rgb)