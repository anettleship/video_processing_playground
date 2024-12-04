import sys
import os
from pathlib import Path

import cv2

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# Paths
current_path = Path(os.getcwd())
image_name = Path('image.jpg')

# Read image from your local file system
original_image = cv2.imread(current_path / image_name)

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Load the classifier and create a cascade object for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

detected_faces = face_cascade.detectMultiScale(grayscale_image)

for (column, row, width, height) in detected_faces:
    cv2.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )

resized_original_image = ResizeWithAspectRatio(original_image, height=800)

cv2.imshow('Image', resized_original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
