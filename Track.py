from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class Camera:
    def __init__(
        self,
        height,
        pitch,
        pix_w,
        pix_h,
        origin_x=None,
        origin_y=None,
        f=5.43,
        dx=None,
        dy=None,
        cmos_w=7.44,
        cmos_h=5.58,
    ) -> None:
        """
        Create a camera object

        Args:
            height (float): camera's distance from floor in meter
            pitch (float): camera's pitch angle in degree
            pix_w (float): how many pixel in width
            pix_h (float): how many pixel in height
            origin_x (float): origin x in pixel
            origin_y (float): origin y in pixel
            f (float): focal length in millimeter
            dx (float): pixel width in millimeter
            dy (float): pixel height in millimeter
            cmos_w (float): cmos width in millimeter
            cmos_h (float): cmos length in millimeter
        """
        origin_x = origin_x or pix_w / 2
        origin_y = origin_y or pix_h / 2

        self.f = f / 1000
        # millimeter per pixel
        self.dx = (dx or cmos_w / pix_w) / 1000
        self.dy = (dx or cmos_h / pix_h) / 1000
        # Image Origin in pixel
        self.origin_x = origin_x
        self.origin_y = origin_y
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
        wi = xy[:, 0] - self.origin_x
        hi = xy[:, 1] - self.origin_y
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

        self.pix_gcs = pixel_ground_coord(self.boxes.xyxy)
        self.coords = camera.to_ground_coord(self.pix_gcs)


class TrackResult:
    """
    TrackResult object
    """

    def __init__(self, id, cls, fps, window=1) -> None:
        self.id = id
        self.cls = cls
        self.interval = 1 / fps

        self.last_coord = None
        self.last_pix_gc = None
        self.last_xyxy = None
        # Speed filter cache
        self.speeds = deque(maxlen=int(fps * window))

    def update(self, xyxy, coord, pix_gc):
        self.last_xyxy = xyxy

        if self.last_coord is not None:
            speed = (coord - self.last_coord) / self.interval
            self.speeds.append(speed.numpy())
        self.last_coord = coord
        self.last_pix_gc = pix_gc

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

        self.frame_w = int(_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera = Camera(
            height=2.046,
            pitch=np.rad2deg(0.2888),
            pix_w=self.frame_w,
            pix_h=self.frame_h,
            origin_x=366.51,
            origin_y=305.83,
            dx=0.023,
            dy=0.023,
            f=20.1619,
        )

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

                ids = (
                    boxes.id.numpy().astype(np.int32)
                    if boxes.id is not None
                    else np.empty(0)
                )
                cls = boxes.cls.cpu().numpy()

                xxyy = boxes.xyxy
                coords = gcs.coords
                pix_gcs = gcs.pix_gcs

                for id, cl, xyxy, coord, pix_gc in zip(ids, cls, xxyy, coords, pix_gcs):
                    track = (
                        self.tracks[id]
                        if id in self.tracks
                        else TrackResult(id, result.names[cl], self.fps)
                    )
                    self.tracks[id] = track.update(xyxy, coord, pix_gc)

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

        from ultralytics.yolo.utils.plotting import Annotator, Colors

        annotator = Annotator(deepcopy(self.orig_img), line_width, font_size, font)

        red = Colors.hex2rgb("#0033FF")
        blue = Colors.hex2rgb("#FF3838")
        if boxes:
            for track in self.tracks.values():
                label = str(track)
                annotator.box_label(track.xyxy.squeeze(), label, color=blue)

                if np.shape(track.speed) == (2,):
                    speed = np.linalg.norm(track.speed)

                    xy = track.xyxy[[0, 3]].numpy().astype(np.int32)
                    annotator.text(
                        xy,
                        f"speed: {speed:.2f}",
                        txt_color=blue,
                        box_style=True,
                    )

                    xy[1] -= 20
                    annotator.text(
                        xy,
                        f"coord: [{track.last_coord[0]:.2f}, {track.last_coord[1]:.2f}]",
                        txt_color=blue,
                        box_style=True,
                    )

                    p1 = track.last_pix_gc
                    p2 = track.last_pix_gc + 5
                    xyxy = torch.hstack((p1, p2)).numpy().astype(np.int32)
                    annotator.box_label(xyxy, color=red)

        res = annotator.result()

        if show:
            cv2.imshow("result", res)
            cv2.waitKey(1)
        return res


tracker = Tracker("Video.mp4")

save_path = "Saved.mp4"
writer = cv2.VideoWriter(
    save_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    tracker.fps,
    (tracker.frame_w, tracker.frame_h),
)

for tracks in tracker.result():
    plotted = tracker.plot()
    writer.write(plotted)
