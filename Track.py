from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


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
        self.origin_w = pix_w / 2
        self.origin_h = pix_h / 2
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
        wi = xy[:, 0] - self.origin_w
        hi = xy[:, 1] - self.origin_h
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


class TrackResult:
    """
    TrackResult object
    """

    def __init__(self, id, cls, fps, window=1) -> None:
        self.id = id
        self.cls = cls
        self.interval = 1 / fps

        self.last_coord = None
        self.last_xyxy = None
        # Speed filter cache
        self.speeds = deque(maxlen=int(fps * window))

    def update(self, xyxy, coord):
        self.last_xyxy = xyxy

        if self.last_coord is not None:
            speed = (coord - self.last_coord) / self.interval
            self.speeds.append(speed.numpy())
        self.last_coord = coord

        return self

    @property
    def speed(self):
        return np.mean(self.speeds, axis=0)

    @property
    def xyxy(self):
        return self.last_xyxy

    def __str__(self) -> str:
        return f"id: {self.id}, cls: {self.cls}"


class Tracker:
    """
    Tracker object
    """

    def __init__(self, video, filters=["person"], device=0) -> None:
        self.model = YOLO("yolov8n.pt")

        _video = cv2.VideoCapture(video)

        self.fps = _video.get(cv2.CAP_PROP_FPS)

        self.frame_w = _video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_h = _video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.camera = Camera(2000, 10, self.frame_w, self.frame_h)

        names = list(self.model.names.values())
        self.pre_result = self.model.track(
            source=video,
            classes=[names.index(name) for name in filters],
            device=device,
            stream=True,
        )

        self.tracks = {}
        self.orig_img = None

    def result(self):
        for result in self.pre_result:
            boxes = result.boxes
            gcs = GroundCoordBoxes(boxes, self.camera)
            self.orig_img = result.orig_img

            def update(self, gcs):
                boxes = gcs.boxes

                ids = boxes.id.numpy().astype(np.int32) if boxes.id is not None else np.empty(0)
                cls = boxes.cls.cpu().numpy()

                xxyy = boxes.xyxy
                coords = gcs.coords

                for id, cl, xyxy, coord in zip(ids, cls, xxyy, coords):
                    track = (
                        self.tracks[id]
                        if id in self.tracks
                        else TrackResult(id, result.names[cl], self.fps)
                    )
                    self.tracks[id] = track.update(xyxy, coord)

                # Remove tracks that no more exist in current frame
                self.tracks = dict(
                    filter(lambda item: item[0] in ids, self.tracks.items())
                )

            update(self, gcs)
            yield self.tracks.values()

    def plot(
        self,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        boxes=True,
        show=True,
    ):
        """
        Plots the detection results on an input RGB image.

        Args:
            boxes (bool): Whether to plot the bounding boxes.
            show (bool): Whether to show the annotated image

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.
        """
        from copy import deepcopy

        from ultralytics.yolo.utils.plotting import Annotator, colors

        annotator = Annotator(deepcopy(self.orig_img), line_width, font_size, font)

        if boxes:
            for track in self.tracks.values():
                label = str(track)
                annotator.box_label(track.xyxy.squeeze(), label, color=colors(0, True))

                if np.shape(track.speed) == (2,):
                    speed = np.linalg.norm(track.speed)
                    speed = np.round(speed, 0)
                    
                    xy = track.xyxy[[0, 3]].numpy().astype(np.int32)
                    annotator.text(
                        xy,
                        f"speed: {speed}",
                        txt_color=colors(0, True),
                        box_style=True,
                    )

        res = annotator.result()
        if show:
            cv2.imshow("result", res)
            cv2.waitKey(1)
        return res


tracker = Tracker("Video.mp4")

for tracks in tracker.result():
    tracker.plot()
