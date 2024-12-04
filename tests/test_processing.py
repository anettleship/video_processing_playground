import cv2
import os
import pytest

from numpy import ndarray
from ..video_operations.processing import FaceHorizontalPositionDetector

class TestFaceHorizontalPositionDetector:

    image = None

    @pytest.fixture
    def single_face(self):
        cwd = os.getcwd()
        test_image_path = 'test_data/justin.jpg'
        yield cv2.imread(cwd + "/" + test_image_path)

    def test_convert_face_to_position_should_return_frame_and_integer(self, single_face):

        frame = single_face 
        processor = FaceHorizontalPositionDetector(image_divisions=10)

        frame, result = processor.process(frame=frame)

        assert isinstance(frame, ndarray) 
        assert isinstance(result[0], int) 

    def test_convert_face_to_position_should_find_single_face_at_index_three(self, single_face):

        frame = single_face 
        processor = FaceHorizontalPositionDetector(image_divisions=10)

        frame, result = processor.process(frame=frame)

        assert result[0] == 3

    @pytest.fixture
    def five_wide_grid(self):
        cwd = os.getcwd()
        test_image_path = 'test_data/multiple_faces.jpg'
        yield cv2.imread(cwd + "/" + test_image_path)

    def test_convert_face_to_position_should_find_a_face_in_every_division_given_a_grid_of_faces(self, five_wide_grid):

        frame = five_wide_grid 
        processor = FaceHorizontalPositionDetector(image_divisions=5)

        frame, result = processor.process(frame=frame)
        indexes_set = set(result)

        assert len(result) > 18 # There are 20 faces, but at least one is not recognised
        for index in range(5):
            assert index in indexes_set
