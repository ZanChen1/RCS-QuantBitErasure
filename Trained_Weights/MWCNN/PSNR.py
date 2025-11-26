import cv2
import math
import numpy


def PSNR(img1, img2):
    D = numpy.array(img1 - img2, dtype=numpy.int64)
    D[:, :] = D[:, :]**2
    RMSE = D.sum()/img1.size
    psnr = 10*math.log10(float(255.**2)/RMSE)
    return psnr

def psnr2(img1,img2):
    mse = numpy.mean((img1/1. - img2/1.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 255.
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))

if __name__ == "__main__":
    img1 = cv2.imread("original 2D4F.bmp", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("Basic2.jpg", cv2.IMREAD_GRAYSCALE)
    psnr = PSNR(img1, img2)
    print ("The PSNR between the two img of the two is %f" % psnr)

    img1 = cv2.imread("original 2D4F.bmp", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("Final2.jpg", cv2.IMREAD_GRAYSCALE)
    psnr = PSNR(img1, img2)
    print ("The PSNR between the two img of the two is %f" % psnr)

