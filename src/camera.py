"""Webcam capture and frame loop.

Handles opening the webcam, reading frames, and releasing the device for the
rest of the real-time inference pipeline. Uses OpenCV for capture.

The :class:`Camera` API is import-safe: OpenCV (``cv2``) is imported lazily
inside the methods, so ``import src.camera`` succeeds even when OpenCV or a
webcam is unavailable. Failure to open a device is signalled by a raised
exception (never ``sys.exit``) so callers control their own exit behaviour.
"""

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 60
BUFFER_SIZE = 1


class CameraError(RuntimeError):
    """Raised when no webcam can be opened across the attempted indices."""


class Camera:
    """Reusable OpenCV webcam capture wrapper.

    Opening by index uses the index-fallback semantics of the Phase 1 trial:
    when ``camera_index == 0`` the indices ``0, 1, 2`` are tried in order;
    otherwise only the explicitly requested index is tried. Each candidate is
    opened with ``cv2.CAP_DSHOW`` (faster device open on Windows). The first
    index that reports ``isOpened()`` is kept; the rest are released. An index
    that has already failed is never re-tried.
    """

    def __init__(self, camera_index=0, width=FRAME_WIDTH, height=FRAME_HEIGHT,
                 fps=FPS, buffer_size=BUFFER_SIZE):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self._cap = None

    def open(self):
        """Open the first available webcam and apply capture properties.

        Returns the index that was opened. Raises :class:`CameraError` if no
        candidate index can be opened.
        """
        import cv2

        indices = [0, 1, 2] if self.camera_index == 0 else [self.camera_index]

        opened_cap = None
        opened_index = None
        for idx in indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                opened_cap = cap
                opened_index = idx
                break
            cap.release()

        if opened_cap is None:
            raise CameraError(
                "Cannot open any camera. Tried indices "
                f"{indices}. Check that a webcam is connected and not in use "
                "by another application."
            )

        opened_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        opened_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        opened_cap.set(cv2.CAP_PROP_FPS, self.fps)
        opened_cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        self._cap = opened_cap
        return opened_index

    def read(self):
        """Read a single frame.

        Returns ``(ret, frame)`` where ``ret`` is a bool and ``frame`` is the
        captured BGR image (or ``None`` on failure), mirroring
        ``cv2.VideoCapture.read``.
        """
        if self._cap is None:
            raise CameraError("Camera is not open. Call open() first.")
        return self._cap.read()

    def release(self):
        """Release the underlying capture device, if open."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self):
        """Whether the capture device is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
