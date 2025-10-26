import os
import re
import numpy as np

class getYUVFrame:
    """
    Unified YUV reader for:
      1) Raw YUV420 planar (I420/YUV420/YV12) without headers (.yuv)
      2) YUV4MPEG2 container (.y4m) with text headers

    getFrame(index) returns:
      ndarray (H, W, 3) in YUV order, dtype=float32, range [0, 255]
      U/V are upsampled (nearest 2x repeat) to match Y size.

    Notes:
      - .y4m: skips global header line and each per-frame "FRAME" header line.
      - Only 4:2:0 is supported for .y4m. If 'C' tag is missing, default is 420.
      - .yuv: supports I420/YUV420/YV12 (V/U order differs for YV12).
    """

    def __init__(self, filepath: str, width: int, height: int, fmt: str = "I420"):
        self.filepath = filepath
        self.width = int(width)
        self.height = int(height)
        self.fmt = fmt.upper()
        self.is_y4m = filepath.lower().endswith(".y4m")

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # Open once and reuse
        self._fp = open(self.filepath, "rb")

        if self.is_y4m:
            # -------- Parse Y4M global header (first line ending with \n) --------
            self.header_len, self.y4m_meta = self._read_y4m_header()

            # If header provides W/H, prefer those (safer)
            hw = self._parse_y4m_wh(self.y4m_meta)
            if hw is not None:
                w_hdr, h_hdr = hw
                if (w_hdr != self.width) or (h_hdr != self.height):
                    self.width, self.height = w_hdr, h_hdr

            # Parse chroma; default to 420 if missing 'C...' tag
            self.chroma = self._y4m_chroma(self.y4m_meta)  # '420', '422', '444', etc.
            if self.chroma != '420':
                raise ValueError(
                    f"Only 4:2:0 Y4M is supported, got chroma='{self.chroma}' in header: {self.y4m_meta!r}"
                )

            # For 4:2:0
            self.frame_bytes = int(self.width * self.height * 3 // 2)
            # Per-frame headers are variable-length; random seek by math is unreliable.
            # We'll rescan from data start on each getFrame(index).
            self._data_start = self.header_len
            self.num_frames = None  # unknown unless full scan (skipped for speed)

        else:
            # -------- Raw YUV420 reader (I420/YUV420/YV12) --------
            if self.fmt not in ("I420", "YUV420", "YV12"):
                raise ValueError("Only I420/YUV420/YV12 is supported for raw .yuv")

            # Some raw YUV are aligned in memory; keep safe alignment then crop
            self.fwidth = (self.width + 31) // 32 * 32
            self.fheight = (self.height + 15) // 16 * 16
            self.frame_bytes = int(self.fwidth * self.fheight * 3 // 2)

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

    # ========================= Y4M helpers =========================

    def _read_y4m_header(self):
        """
        Read the first line (global header) of a Y4M file and return (length_in_bytes, header_bytes).
        """
        self._fp.seek(0, os.SEEK_SET)
        header = self._fp.readline()  # includes trailing b'\n'
        if not header or not header.startswith(b"YUV4MPEG2"):
            raise EOFError("Invalid Y4M header (missing 'YUV4MPEG2').")
        return len(header), header

    @staticmethod
    def _parse_y4m_wh(header_bytes):
        """Extract W/H from the Y4M header if present."""
        try:
            text = header_bytes.decode("ascii", errors="ignore")
        except Exception:
            return None
        mW = re.search(r"\bW(\d+)\b", text)
        mH = re.search(r"\bH(\d+)\b", text)
        if mW and mH:
            return int(mW.group(1)), int(mH.group(1))
        return None

    @staticmethod
    def _y4m_chroma(header_bytes) -> str:
        """
        Normalize chroma code from Y4M header:
          - If no 'C' tag, default to '420' (common practice).
          - Map known 420 variants (C420, C420jpeg, C420mpeg2, 420paldv, 420) -> '420'
          - Return '422' / '444' for those tags (unsupported in this reader).
        """
        try:
            text = header_bytes.decode("ascii", errors="ignore")
        except Exception:
            return '420'  # if undecodable, assume 420

        m = re.search(r"\bC([0-9A-Za-z]+)\b", text)
        if not m:
            return '420'  # missing C tag -> assume 420

        raw = m.group(1).lower()
        if raw in ('420', '420jpeg', '420mpeg2', '420paldv'):
            return '420'
        if raw.startswith('420'):
            return '420'
        if raw.startswith('422'):
            return '422'
        if raw.startswith('444'):
            return '444'
        return raw  # unknown; pass through (likely unsupported)

    def _read_exact(self, n: int) -> bytes:
        buf = self._fp.read(n)
        if len(buf) != n:
            raise EOFError("Unexpected end of file.")
        return buf

    # ========================= Public API =========================

    def getFrame(self, index: int) -> np.ndarray:
        """
        Return the 'index'-th frame as float32 YUV (H, W, 3) in [0,255].
        .y4m: seek to data start, skip 'index' frames (FRAME + payload), then read this frame.
        .yuv: seek by index * frame_bytes using aligned fwidth/fheight, then crop to W/H.
        """
        if index < 0:
            raise ValueError("Frame index must be non-negative.")

        if self.is_y4m:
            # Rewind to just after global header
            self._fp.seek(self._data_start, os.SEEK_SET)

            # Skip preceding frames
            for _ in range(index):
                line = self._fp.readline()
                if not line or not line.startswith(b"FRAME"):
                    raise EOFError("Invalid or truncated Y4M per-frame header while skipping.")
                # Skip this frame payload
                self._fp.seek(self.frame_bytes, os.SEEK_CUR)

            # Read this frame's per-frame header
            line = self._fp.readline()
            if not line or not line.startswith(b"FRAME"):
                raise EOFError("Invalid or truncated Y4M per-frame header.")

            # Now read raw 4:2:0 planar payload
            fW, fH = self.width, self.height
            ChW, ChH = fW // 2, fH // 2

            Y = np.frombuffer(self._read_exact(fW * fH), np.uint8).reshape((fH, fW))
            U = np.frombuffer(self._read_exact(ChW * ChH), np.uint8).reshape((ChH, ChW))
            V = np.frombuffer(self._read_exact(ChW * ChH), np.uint8).reshape((ChH, ChW))

            # Upsample chroma (nearest 2x repeat)
            U_up = U.repeat(2, axis=0).repeat(2, axis=1)
            V_up = V.repeat(2, axis=0).repeat(2, axis=1)

            YUV = np.dstack((Y, U_up, V_up)).astype(np.float32)
            return YUV

        # -------- Raw YUV420 (I420/YUV420 or YV12) --------
        # Seek to start of this frame in aligned buffer
        self._fp.seek(index * self.frame_bytes, os.SEEK_SET)

        fW, fH = self.fwidth, self.fheight
        W, H = self.width, self.height
        ChW, ChH = fW // 2, fH // 2

        # Read planes
        Y = self._read_exact(fW * fH)
        Y = np.frombuffer(Y, dtype=np.uint8).reshape((fH, fW))

        if self.fmt in ("I420", "YUV420"):
            U = np.frombuffer(self._read_exact(ChW * ChH), np.uint8).reshape((ChH, ChW))
            V = np.frombuffer(self._read_exact(ChW * ChH), np.uint8).reshape((ChH, ChW))
        else:  # YV12: V then U
            V = np.frombuffer(self._read_exact(ChW * ChH), np.uint8).reshape((ChH, ChW))
            U = np.frombuffer(self._read_exact(ChW * ChH), np.uint8).reshape((ChH, ChW))

        # Upsample chroma and crop to exact WxH
        U_up = U.repeat(2, axis=0).repeat(2, axis=1)[:H, :W]
        V_up = V.repeat(2, axis=0).repeat(2, axis=1)[:H, :W]
        Y = Y[:H, :W]

        YUV = np.dstack((Y, U_up, V_up)).astype(np.float32)
        return YUV

    def frameCount(self) -> int:
        """
        Return total frame count if determinable.
        - Raw .yuv: derived from file size.
        - .y4m: return -1 (unknown) unless scanning whole file (not done here).
        """
        if self.is_y4m:
            return -1
        return int(self.num_frames) if self.num_frames is not None else -1
