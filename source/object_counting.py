import cv2
import numpy as np
# from flask import Flask, render_template
from matplotlib import pyplot as plt
from skimage import io
import os
from .utils import get_folder_dir  # Custom function for better directory name handling


def show_original_img(image):
    # Reading Image
    # url = '/content/gdrive/MyDrive/Assets/Images_Proj1_topic1/objets4.jpg'
    # url = '/content/drive/MyDrive/Assets/Images_Proj1_topic1/hatGao.png'

    # image = io.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def medianBlur_unsharp(image):
    # Median blur
    # we can not use parameter of medianBlur > 7, 7 is max
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_blur = cv2.medianBlur(image, 5)
    # cv2.imshow(image_blur)
    gaussian = cv2.GaussianBlur(image_blur, (5, 5), 2.0)
    unsharp_img = cv2.addWeighted(image_blur, 1.5, gaussian, -0.5, 0, image)
    return unsharp_img


def gaussian_grayScale(unsharp_img):
    gaussian = cv2.GaussianBlur(unsharp_img, (1, 1), 2.0)
    print('gaussian')
    # cv2.imshow(gaussian)
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    return gray


def fft(gray):
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift[227:233, 219:225] = 255
    dft_shift[227:233, 236:242] = 255

    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
    img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_back


def find_Canny_edges_Closing(img_back):
    # Find Canny edges
    # tim bien ngoai cung
    # phan vung (lay nguong)
    edged = cv2.Canny(img_back, 90, 200)
    # cv2.imshow(edged)
    # Closing
    # tham so khac
    # kernel = np.ones((13,13),np.uint8)

    kernel = np.ones((1, 1), np.uint8)

    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    return closing


# ================================================= in the last step
def contours_count_hatgao(image, closing):
    # In case of Hat Gao problem
    # Finding Contours
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Contour Retrieval Mode,   stores absolutely all the contour points.
    # https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html

    number = str(len(contours))

    # Draw all contours
    # -1 signifies drawing all contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    cv2.putText(image, "Number of contours = " + number, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image


def contours_count_tools(image, image_blur):
    # In case of counting file objects...

    gray = cv2.cvtColor(image_blur, cv2.COLOR_RGB2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 1)

    # Điều chỉnh giá trị của hàm adaptiveThreshold để thấy kết quả, kiểm tra lại 2 giá trị cuối
    # giá trị cuối nếu >8 thì kết quả counting sẽ bị lệch (tức là sẽ lớn hơn 10 objects)
    img_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 10)

    # Kernel ảnh hưởng đến việc nối các vùng lại mới nhau.
    kernel = np.ones((13, 13), np.uint8)
    # Giãn nở để liên kết các vùng phía trong với nhau
    img = cv2.dilate(img_thresh, kernel)

    # Co lại để tách các object với nhau, tăng độ chính xác của hàm count
    img = cv2.erode(img, kernel)

    img = cv2.medianBlur(img, 5)

    # cv2.RETR_EXTERNAL:
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_of_objects = str(len(contours))
    # print("Number of objects: ", num_of_objects)

    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    cv2.putText(image, "Number of contours = " + num_of_objects, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image
