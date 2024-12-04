from video_operations.capture import VideoCapture
from video_operations.processing import FaceDetection, FaceHorizontalPositionDetector

face_detection_processor = FaceDetection()
face_horizontal_positioner = FaceHorizontalPositionDetector(image_divisions=100)
video_capture = VideoCapture(show_video=True, print_values=True, output_height=800, video_processors=[face_detection_processor, face_horizontal_positioner])

if __name__ == "__main__":
    video_capture.run()