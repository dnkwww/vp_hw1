import numpy as np
import cv2
import time

from getYuvFrame import getYUVFrame
from display import displayFrame, yuv2bgr, psnr


def bgr_to_y(img_bgr):
    """
    將 BGR 影像轉為 Y 通道（YCrCb 的 Y）。
    輸入可為 uint8 或 float；此處轉 uint8 再轉回 float32，確保 cvtColor 正常。
    """
    img_u8 = img_bgr.astype(np.uint8)
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    return y


class ebma_halfPel:
    """
    兩階段 EBMA 半畫素：
      1) 原圖上做整數像素全域搜尋 (±R)
      2) 以最佳整數位移為中心，做半畫素 9 點精煉 (±0.5 in {x,y})
    * 一律在原圖尺度運算與視覺化；MV 單位 = 像素（可含 0.5）
    * MAD 與 PSNR 以 Y(亮度) 通道計算，padding 採 reflect
    """

    # N: block size; R: search range
    def __init__(self, video, N, R, flag):
        self.video = video
        self.N = int(N)   # block size (原圖尺度)
        self.R = int(R)   # search range (原圖尺度)
        self.flag = flag

    def _bilinear_block(self, img_pad, r0_float, c0_float, H, W):
        """
        從 pad 後的灰階影像 img_pad 以 (r0_float, c0_float) 為左上角
        取出 HxW 的半畫素樣本（手刻雙線性）。
        回傳 float32 的區塊。
        """
        block = np.zeros((H, W), dtype=np.float32)

        r_int = int(np.floor(r0_float))
        c_int = int(np.floor(c0_float))
        dr = r0_float - r_int
        dc = c0_float - c_int

        # 逐像素雙線性（H,W 都不大，容易閱讀與除錯）
        for rr in range(H):
            r_0 = r_int + rr
            r_1 = r_0 + 1
            for cc in range(W):
                c_0 = c_int + cc
                c_1 = c_0 + 1

                # reflect padding 已避免越界，但保留保險判斷
                if r_1 >= img_pad.shape[0] or c_1 >= img_pad.shape[1]:
                    continue

                p00 = img_pad[r_0, c_0]
                p01 = img_pad[r_0, c_1]
                p10 = img_pad[r_1, c_0]
                p11 = img_pad[r_1, c_1]

                block[rr, cc] = (
                    p00 * (1 - dc) * (1 - dr) +
                    p01 * dc       * (1 - dr) +
                    p10 * (1 - dc) * dr       +
                    p11 * dc       * dr
                )
        return block

    def match(self):
        start_time = time.time()

        # --- 參數與檔名尾碼 ---
        width, height = 352, 288
        N, R = self.N, self.R
        search_params = f'_N{N}_R{R}.jpg'

        # --- 讀取兩幀（原圖尺度）---
        frames = getYUVFrame(self.video, width, height)
        anchor_bgr = yuv2bgr(frames.getFrame(29))   # frame 7 = anchor
        target_bgr = yuv2bgr(frames.getFrame(30))   # frame 8 = target

        # --- 轉 Y 通道（MAD/PSNR 都用亮度）---
        f1_y = bgr_to_y(anchor_bgr)   # anchor Y
        f2_y = bgr_to_y(target_bgr)   # target Y

        H, W = f1_y.shape
        assert (H, W) == (height, width)

        # --- 反射 padding（邊界更穩定）---
        # d 需能涵蓋：搜尋半徑 R + 區塊 N + 半畫素取樣的 1 像素緣
        d = R + N + 1
        f1_pad = np.pad(f1_y, ((d, d), (d, d)), mode='reflect')
        f2_pad = np.pad(f2_y, ((d, d), (d, d)), mode='reflect')

        # --- 區塊網格 ---
        nBh = int(np.ceil(H / N))
        nBw = int(np.ceil(W / N))

        mvx = np.zeros((nBh, nBw), dtype=np.float32)
        mvy = np.zeros((nBh, nBw), dtype=np.float32)
        predict_y = np.zeros((H, W), dtype=np.float32)

        # 半畫素 9 點（含 (0,0) 原點，等下會跳過 (0,0) 以避免重算）
        half_offsets = [
            (-0.5, -0.5), (0.0, -0.5), (0.5, -0.5),
            (-0.5,  0.0), (0.0,  0.0), (0.5,  0.0),
            (-0.5,  0.5), (0.0,  0.5), (0.5,  0.5),
        ]

        # --- 兩階段搜尋 ---
        for bi in range(nBh):
            for bj in range(nBw):
                r0 = bi * N
                c0 = bj * N
                bh = min(N, H - r0)  # 保險：若 H/W 不是 N 整除也能處理
                bw = min(N, W - c0)

                # anchor 區塊（從 pad 後影像取）
                anc_blk = f1_pad[r0 + d: r0 + d + bh, c0 + d: c0 + d + bw]

                # ===== 階段一：整數像素全域搜尋 =====
                MAD_min = 256 * N * N
                best_dy_int = 0
                best_dx_int = 0

                for dy in range(-R, R + 1):
                    for dx in range(-R, R + 1):
                        r_t = r0 + d + dy
                        c_t = c0 + d + dx
                        tar_blk_int = f2_pad[r_t: r_t + bh, c_t: c_t + bw]
                        if tar_blk_int.shape[0] != bh or tar_blk_int.shape[1] != bw:
                            continue
                        MAD = np.sum(np.abs(anc_blk - tar_blk_int))
                        if MAD < MAD_min:
                            MAD_min = MAD
                            best_dy_int = dy
                            best_dx_int = dx

                # ===== 階段二：半畫素 9 點精煉（以最佳整數為中心）=====
                best_dy = float(best_dy_int)
                best_dx = float(best_dx_int)
                MAD_best = MAD_min  # 至少不會比整數更差

                for (ddy, ddx) in half_offsets:
                    if ddy == 0.0 and ddx == 0.0:
                        continue  # (0,0) 即整數點，已算過

                    dy_half = best_dy_int + ddy
                    dx_half = best_dx_int + ddx

                    r_start = r0 + d + dy_half
                    c_start = c0 + d + dx_half

                    tar_blk_half = self._bilinear_block(f2_pad, r_start, c_start, bh, bw)
                    MAD_half = np.sum(np.abs(anc_blk - tar_blk_half))

                    if MAD_half < MAD_best:
                        MAD_best = MAD_half
                        best_dy = dy_half
                        best_dx = dx_half

                # 存 MV（原圖尺度；可為 0.5）
                mvy[bi, bj] = best_dy
                mvx[bi, bj] = best_dx

                # 用最佳 (dx,dy) 重建預測區塊（半畫素雙線性）
                r_start = r0 + d + best_dy
                c_start = c0 + d + best_dx
                pred_blk = self._bilinear_block(f2_pad, r_start, c_start, bh, bw)
                predict_y[r0: r0 + bh, c0: c0 + bw] = pred_blk

        # --- 指標與輸出 ---
        process_time = time.time() - start_time
        # print('[EBMA (Two-Stage Half-pel)]')
        print(f'processing time = {process_time:.4f} s')

        # PSNR 用「target vs predict」更符合補償目標
        psnr_val = psnr(f2_y.astype(np.float32), predict_y.astype(np.float32))
        print(f'PSNR (target_y vs predict_y) = {psnr_val:.4f} dB')

        # 顯示（displayFrame 需要 BGR，這裡把灰階堆 3 通道）
        predict_u8 = np.clip(predict_y, 0, 255).astype(np.uint8)
        predict_bgr = cv2.merge([predict_u8, predict_u8, predict_u8])
        displayFrame(predict_bgr, self.flag, 'predict', 'ebma_halfPel_predict' + search_params)

        # 灰階誤差圖
        err_u8 = np.clip(np.abs(f2_y - predict_y), 0, 255).astype(np.uint8)
        err_bgr = cv2.merge([err_u8, err_u8, err_u8])
        # displayFrame(err_bgr, 'error_gray', self.flag, 'ebma_halfPel_error_gray' + search_params)

        # MV 視覺化：注意用原圖寬高（不要乘 2）
        # draw_motion_field(mvx, mvy, W, H, 'ebma_halfPel_mv' + search_params)

        # 為了和你既有呼叫相容（mvx, mvy = ebma.match()），只回傳 MV
        return mvx, mvy
