"""Hierarchy of functions to be applied
1. Brightened the image
2. Sharpen the image
3. Add Contrast to the image
4. Deskew the image


Author : Parul Sharma, Ajinkya Rahane and Shiv @ internship Finance Project iNeuron
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance


# importing utils functions


# Image from ESRGAN / image from the previous module

class ImageEnhancement:

    def __init__(self):
        pass

    def read_image(self):
        plt.rcParams['figure.figsize'] = [12, 8]
        # read the image from previous module
        image = cv2.imread('passbook_esrgan.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title('Original Image form ESRGAN')
        plt.show()
        print(image.shape)
        return image

    def brighten_image(self, image):
        # self.image = image
        # im = Image.open("passbook_esrgan.png")
        enhancer = ImageEnhance.Brightness(image)
        bright_image = enhancer.enhance(1.1)
        plt.imshow(bright_image)
        plt.title('bright Image')
        plt.show()
        return bright_image

    def contrast_image(self,bright_image):
        enhancer2 = ImageEnhance.Contrast(bright_image)
        contrasted_image= enhancer2.enhance(1.3)
        # plt.imshow(contrast_image)
        # plt.title('contrast Image')
        # plt.show()
        return contrasted_image

    def sharpen_image(self, contrast_image):
        enhancer1 = ImageEnhance.Sharpness(contrast_image)
        sharpened_image = enhancer1.enhance(3.0)
        # plt.imshow(sharpened_image)
        # plt.title('sharpe Image')
        # plt.show()
        return sharpened_image

    def deskew_image(self, sharpened_image):

        # convert the image to grayscale and flip the foreground and background to ensure foreground is now "white" and
        # the background is "black"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)

        # threshold the image, setting all foreground pixels to 255 and all background pixels to 0
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # grab the (x, y) coordinates of all pixel values that are greater than zero, then use these coordinates to
        # compute a rotated bounding box that contains all coordinates
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # the `cv2.minAreaRect` function returns values in the range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)

        # otherwise, just take the inverse of the angle to make it positive
        else:
            angle = -angle

        # rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # draw the correction angle on the image so we can validate it
        # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the output image
        print("[INFO] angle: {:.3f}".format(angle))
        # cv2.imshow("Input", image)
        # cv2.imshow("Rotated", deskewed_image)
        return deskewed_image

    cv2.waitKey(10)
    print('-------------------------------------end of code-----------------------------------------')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img_enhancement_object = ImageEnhancement()
image_object = img_enhancement_object.read_image()
image_brightened = img_enhancement_object.brighten_image(image_object)
image_contrast = img_enhancement_object.contrast_image(image_brightened)
image_sharpen = img_enhancement_object.sharpen_image(image_contrast)
image_deskew = img_enhancement_object.deskew_image(image_sharpen)