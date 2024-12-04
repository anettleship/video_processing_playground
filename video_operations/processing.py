from numpy import ndarray

import cv2

class VideoProcessor:

    def __init__(self):

        pass

    def process(self, image:ndarray) -> ndarray:

        pass


class FaceDetection(VideoProcessor):

    def __init__(self):

        # Load the classifier and create a cascade object for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    def _add_overlay(self, image: ndarray, detected_faces:list[int, int, int, int]) -> ndarray:

        # Draw Green Rectangles over all detected faces
        for (column, row, width, height) in detected_faces:
            cv2.rectangle(
                image,
                (column, row),
                (column + width, row + height),
                (0, 255, 0),
                2
            )
        return image

    def _preprocess(self, image:ndarray) -> ndarray:

        # Convert color image to grayscale for Viola-Jones
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def process(self, image:ndarray) -> ndarray:

        pre_processed_image = self._preprocess(image)
        detected_faces = self.face_cascade.detectMultiScale(pre_processed_image)
        self._add_overlay(image, detected_faces)
        return image