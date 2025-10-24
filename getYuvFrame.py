
import os
import numpy as np

class getYUVFrame:
    """
    Simple YUV420 (I420) frame reader.

    Usage:
        frames = getYUVFrame("akiyo_352x288_30.yuv", width=352, height=288)
        yuv = frames.getFrame(6)  # returns HxWx3 (Y, U, V), dtype=float32 in [0, 255]
    """
    def __init__(self, filepath: str, width: int, height: int, fmt: str = "I420"):
        if fmt.upper() not in ("I420", "YUV420", "YV12"):
            raise ValueError("Only I420/YUV420/YV12 is supported.")
        self.filepath = filepath
        self.width = int(width)
        self.height = int(height)
        self.fmt = fmt.upper()

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"YUV file not found: {self.filepath}")

        # Some raw YUV clips are aligned in memory; use a safe alignment like OpenCV sample.
        # We read using padded width/height, then crop to the target width/height.
        self.fwidth = (self.width + 31) // 32 * 32
        self.fheight = (self.height + 15) // 16 * 16

        # Bytes per frame for 4:2:0 planar (I420 / YV12): 1.5 bytes per pixel
        self.frame_bytes = int(self.fwidth * self.fheight * 3 // 2)

        # Open once and reuse
        self._fp = open(self.filepath, "rb")

        # Calculate total frames (best-effort)
        try:
            file_size = os.path.getsize(self.filepath)
            self.num_frames = file_size // self.frame_bytes
        except OSError:
            self.num_frames = None

    def __del__(self):
        try:
            if hasattr(self, "_fp") and self._fp:
                self._fp.close()
        except Exception:
            pass

    def _read_plane(self, count: int) -> np.ndarray:
        buf = self._fp.read(count)
        if len(buf) != count:
            raise EOFError("Unexpected end of file while reading YUV data.")
        return np.frombuffer(buf, dtype=np.uint8)

    def getFrame(self, index: int) -> np.ndarray:
        """
        Return the 'index'-th frame as a float32 array of shape (H, W, 3) in YUV order.
        Value range is [0, 255] (no offset removed).

        For I420/YUV420:
            Layout = [Y plane][U plane][V plane], with U/V at half resolution.
        For YV12:
            Layout = [Y plane][V plane][U plane].
        """
        if index < 0:
            raise ValueError("Frame index must be non-negative.")

        # Seek to frame start
        self._fp.seek(index * self.frame_bytes, os.SEEK_SET)

        fW, fH = self.fwidth, self.fheight
        W, H = self.width, self.height
        ChW, ChH = fW // 2, fH // 2  # chroma dimensions

        # Read planes
        Y = self._read_plane(fW * fH).reshape((fH, fW))

        if self.fmt in ("I420", "YUV420"):
            U = self._read_plane(ChW * ChH).reshape((ChH, ChW))
            V = self._read_plane(ChW * ChH).reshape((ChH, ChW))
        else:  # YV12
            V = self._read_plane(ChW * ChH).reshape((ChH, ChW))
            U = self._read_plane(ChW * ChH).reshape((ChH, ChW))

        # Upsample U/V to match Y (nearest / 2x repeat)
        U_up = U.repeat(2, axis=0).repeat(2, axis=1)
        V_up = V.repeat(2, axis=0).repeat(2, axis=1)

        # Stack, crop to exact WxH, and convert to float32 (0~255)
        YUV = np.dstack((Y, U_up, V_up))[:H, :W, :].astype(np.float32)

        return YUV

    def frameCount(self) -> int:
        """Return total frame count if determinable, else -1."""
        return int(self.num_frames) if self.num_frames is not None else -1
