import cv2

video_capture = cv2.VideoCapture(0)

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

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    resized_original_image = ResizeWithAspectRatio(frame, height=800)

    cv2.imshow('Image', resized_original_image)
    cv2.waitKey(1)