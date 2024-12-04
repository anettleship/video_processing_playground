from video_operations.capture import VideoCapture
from video_operations.processing import FaceDetection

face_detection_processor = FaceDetection()
video_capture = VideoCapture(show_video=True, output_height=800, video_processors=[face_detection_processor], )

if __name__ == "__main__":
    video_capture.run()