import cv2

cv2.imshow('image', cv2.imread('dataset/s1/1.pgm'))
print("size of image: ", cv2.imread('dataset/s1/1.pgm').shape)
cv2.waitKey(0)
cv2.destroyAllWindows()