from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes, Results

import cv2
import torch
import numpy as np


class Camera:
    def __init__(
        self, height, pitch, pix_w, pix_h, f=5.43, cmos_w=7.44, cmos_h=5.58
    ) -> None:
        """
        Create a camera object

        Args:
            height (float): camera's distance from floor in millimeter
            pitch (float): camera's pitch angle in degree
            pix_w (float): how many pixel in width
            pix_h (float): how many pixel in height
            f (float): focal length in millimeter
            cmos_w (float): cmos width in millimeter
            cmos_h (float): cmos length in millimeter
        """
        self.f = f
        # millimeter per pixel
        self.dx = cmos_w / pix_w
        self.dy = cmos_h / pix_h
        # Image Origin in pixel
        self.origin = [pix_w / 2, pix_h / 2]
        self.height = height
        self.pitch = np.deg2rad(pitch)

    def to_ground_coord(self, xy):
        """
        convert pixel coord to ground coord

        Args:
            xy (numpy.ndarray): pixel ground coord

        Returns:
            gc_xy: ground coord
        """
        # Relocate origin
        xy -= torch.tensor(self.origin, device=xy.device, dtype=xy.dtype)
        wi = xy[:, 0]
        hi = xy[:, 1]
        # Angle between optical y-axis and imaging point ray
        beta = torch.atan(hi * self.dy / self.f)
        # Evaluate ground coord
        gc_y = self.height / torch.tan(self.pitch + beta)
        gc_x = (
            wi
            * self.dx
            * torch.sqrt(self.height**2 + gc_y**2)
            / torch.sqrt((hi * self.dy) ** 2 + self.f**2)
        )

        return torch.vstack((gc_x, gc_y)).T


class GroundCoordBoxes:
    """
    A class for storing and manipulating ground coord boxes.

    Args:
        boxes (Boxes): A Boxes object containing the detection bounding boxes.

    Attributes:
        coords (numpy.ndarray): The ground coords of the boxes
    """

    def __init__(self, boxes, camera) -> None:
        def pixel_ground_coord(xyxy):
            """Get the bottom center of the bounding box

            Args:
                xyxy (numpy.ndarray): pixel coordinate of the bounding box
            
            Returns:
                (numpy.ndarray): Bottom center coordinate of the bounding box
            """
            # Extract x coord
            xx = xyxy[:, [0, 2]]
            # Eval the center x
            x = xx.mean(dim=1)
            # Conbine center x with bottom y
            xy = torch.vstack((x, xyxy[:, 3])).T
            return xy

        self.boxes = boxes
        
        xy = pixel_ground_coord(self.boxes.xyxy)
        self.coords = camera.to_ground_coord(xy)


class Tracker:
    """
    Tracker object
    """

    def __init__(self, video, filters=["person"], device=0) -> None:
        self.model = YOLO("yolov8n.pt")

        self.video = cv2.VideoCapture(video)

        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        self.frame_w = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_h = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.camera = Camera(2000, 10, self.frame_w, self.frame_h)

        names = list(self.model.names.values())
        self.pre_result = self.model.track(
            source=video,
            classes=[names.index(name) for name in filters],
            device=device,
            stream=True,
        )

    def plot(
        result,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        labels=True,
        boxes=True,
        show=True,
    ):
        """
        Plots the detection results on an input RGB image.

        Args:
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            show (bool): Whether to show the annotated image

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.
        """
        from copy import deepcopy
        from ultralytics.yolo.utils.plotting import Annotator, colors

        names = result.names
        annotator = Annotator(
            deepcopy(result.orig_img), line_width, font_size, font, example=names
        )

        res = annotator.result()

        if show:
            cv2.imshow("result", res)
            cv2.waitKey(1)

        return res


tracker = Tracker("Video.mp4")
for result in tracker.pre_result:
    boxes = result.boxes
    GroundCoordBoxes(boxes, tracker.camera)
