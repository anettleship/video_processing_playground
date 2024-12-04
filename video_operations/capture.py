import cv2
from numpy import ndarray
from typing import Optional

from video_operations.processing import VideoProcessor

class VideoCapture:

    def __init__(
            self, 
            device_index:Optional[int]=0,
            output_width:Optional[int]=None,
            output_height:Optional[int]=None,
            video_processors:Optional[list[VideoProcessor]]=list(),
        ) -> None:

        self.video_capture = cv2.VideoCapture(device_index)
        self.output_height = output_height
        self.output_width = output_width
        
        self.video_processors = video_processors
        
        # Capture while loop runs while true
        self.run_capture = True

    def _resize_with_aspect_ratio(self, image:ndarray, width:int=None, height:int=None, inter:int=cv2.INTER_AREA):

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
    
    def _apply_video_processors(self, image:ndarray) -> ndarray:

        for processor in self.video_processors:
            image = processor.process(image)

        return image

    def _show_image(self, image:ndarray) -> None:

            cv2.imshow('Image', image)
            cv2.waitKey(1)

    def run(self) -> None:

        while self.run_capture:
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            processed_image = self._apply_video_processors(frame)
            if self.output_height or self.output_width:
                resized_frame = self._resize_with_aspect_ratio(frame, height=self.output_height, width=self.output_width)
                self._show_image(resized_frame)
                continue
            self._show_image(processed_image)