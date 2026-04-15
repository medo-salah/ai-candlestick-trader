import cv2, numpy as np, pytesseract, os
from PIL import Image

def crop_chart_area(img):
    h, w = img.shape[:2]
    return img[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]

def denoise_and_resize(img, size=(512,512)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def preprocess_image(path, out_path=None, size=(512,512)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError('Cannot read image: '+path)
    cropped = crop_chart_area(img)
    proc = denoise_and_resize(cropped, size=size)
    if out_path:
        cv2.imwrite(out_path, cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
    return proc
