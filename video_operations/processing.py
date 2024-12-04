from numpy import ndarray

import cv2

class VideoProcessor:

    def __init__(self):

        pass

    def process(self, image:ndarray) -> tuple[ndarray, int|str]:

        pass


class FaceDetection(VideoProcessor):

    def __init__(self):

        # Load the classifier and create a cascade object for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    def _add_overlay_in_place(self, image: ndarray, detected_faces:list[int, int, int, int]) -> None:

        # Draw Green Rectangles over all detected faces - edits image in place.
        for (column, row, width, height) in detected_faces:
            cv2.rectangle(
                image,
                (column, row),
                (column + width, row + height),
                (0, 255, 0),
                2
            )

    def _preprocess(self, image:ndarray) -> ndarray:

        # Convert color image to grayscale for Viola-Jones
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def process(self, image:ndarray) -> tuple[ndarray,None]:

        pre_processed_image = self._preprocess(image)
        detected_faces = self.face_cascade.detectMultiScale(pre_processed_image)
        self._add_overlay_in_place(image, detected_faces)
        return (image,None)

class FaceHorizontalPositionDetector(VideoProcessor):

    def __init__(self, image_divisions:int) -> None:

        # Load the classifier and create a cascade object for face detection
        self.image_divisions = image_divisions
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    def _convert_face_to_position(self, frame: ndarray, detected_faces:list[int, int, int, int]) -> list:

        # divides image into a number of image divisions
        # returns a list of numbers based on the horizontal position of detected faces in the image
        
        face_positions = list()

        image_horizontal_dimension = len(frame[0])
        image_division_width = image_horizontal_dimension / self.image_divisions

        for (column, row, width, height) in detected_faces:
            
            horizontal_offset = float(width) / 2
            horizontal_centre = float(column) + horizontal_offset
            division_index = horizontal_centre / image_division_width
            face_positions.append(round(division_index))

        return face_positions 

    def _preprocess(self, frame:ndarray) -> ndarray:

        # Convert color image to grayscale for Viola-Jones
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def process(self, frame:ndarray) -> ndarray:

        pre_processed_frame = self._preprocess(frame)
        detected_faces = self.face_cascade.detectMultiScale(pre_processed_frame)
        face_indexes = self._convert_face_to_position(frame, detected_faces)
        return (frame, face_indexes)