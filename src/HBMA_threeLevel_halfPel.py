import numpy as np
import cv2
import time

from getYuvFrame import getYUVFrame
from display import displayFrame, yuv2bgr, psnr

import os
import torch
from torchvision.utils import flow_to_image


def _ceil_div(a, b):
    return (a + b - 1) // b

def bgr_to_y(img_bgr):
    img_u8 = img_bgr.astype(np.uint8)
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0].astype(np.float32)

class hbma_three_level_halfPel:
    """
    讓 A 的 HBMA 行為貼近 B：
      - 全程灰階 MAD/重建
      - reflect padding
      - 等效搜尋範圍：L2=±4 → L1=±4（以 L2*2 為中心）→ L0 整數精煉 ±8 → 半像素 ±0.5
      - 區塊固定 16×16；網格以 L0 步距 16 鋪陣（與 B 相同）
      - PSNR 用灰階（anchor_gray vs predict_gray）
      - match() 仍回傳 (mvx, mvy)
    """
    def __init__(self, video, N, R, flag, anchor_idx=29, target_idx=30):
        self.video = video
        self.N = int(N)
        self.R = int(R)
        self.flag = flag
        self.anchor_idx = int(anchor_idx)
        self.target_idx = int(target_idx)

        self.R_L0_int = 8
        self.R_L1 = 4
        self.R_L2 = 4
        self.R_L0_half = 1.0  # ±1.0（取 0.5 網格）

    @staticmethod
    def _bilinear_sample(gray_img, y, x):
        h, w = gray_img.shape
        y0 = int(np.floor(y)); x0 = int(np.floor(x))
        y1 = y0 + 1;           x1 = x0 + 1
        if y0 < 0 or x0 < 0 or y1 >= h or x1 >= w:
            return 0.0
        dy = y - y0; dx = x - x0
        p00 = gray_img[y0, x0]
        p01 = gray_img[y0, x1]
        p10 = gray_img[y1, x0]
        p11 = gray_img[y1, x1]
        return (p00*(1-dx)*(1-dy) + p01*dx*(1-dy) + p10*(1-dx)*dy + p11*dx*dy)

    def match(self):
        t0 = time.time()
        W, H = 352, 288
        N = self.N
        assert N == 16, "作業規定：所有層級 block size 必須為 16x16。"

        # 讀 YUV → BGR
        frames = getYUVFrame(self.video, W, H)
        anc_bgr = yuv2bgr(frames.getFrame(self.anchor_idx))
        tar_bgr = yuv2bgr(frames.getFrame(self.target_idx))

        # 轉灰階
        anc_gray = cv2.cvtColor(anc_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        tar_gray = cv2.cvtColor(tar_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 金字塔
        f0_L0 = anc_gray
        f1_L0 = tar_gray
        f0_L1 = cv2.resize(f0_L0, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        f1_L1 = cv2.resize(f1_L0, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        f0_L2 = cv2.resize(f0_L0, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        f1_L2 = cv2.resize(f1_L0, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

        # reflect padding
        d_L2 = self.R_L2 + N
        d_L1 = self.R_L1 + N
        d_L0 = self.R_L0_int + int(np.ceil(self.R_L0_half)) + N

        f0_L2_pad = np.pad(f0_L2, ((d_L2,d_L2),(d_L2,d_L2)), mode='reflect')
        f1_L2_pad = np.pad(f1_L2, ((d_L2,d_L2),(d_L2,d_L2)), mode='reflect')
        f0_L1_pad = np.pad(f0_L1, ((d_L1,d_L1),(d_L1,d_L1)), mode='reflect')
        f1_L1_pad = np.pad(f1_L1, ((d_L1,d_L1),(d_L1,d_L1)), mode='reflect')
        f0_L0_pad = np.pad(f0_L0, ((d_L0,d_L0),(d_L0,d_L0)), mode='reflect')
        f1_L0_pad = np.pad(f1_L0, ((d_L0,d_L0),(d_L0,d_L0)), mode='reflect')

        H0, W0 = f0_L0.shape
        num_h = _ceil_div(H0, N)
        num_w = _ceil_div(W0, N)

        mvx_L2 = np.zeros((num_h, num_w), dtype=np.float32)
        mvy_L2 = np.zeros((num_h, num_w), dtype=np.float32)
        mvx_L1 = np.zeros((num_h, num_w), dtype=np.float32)
        mvy_L1 = np.zeros((num_h, num_w), dtype=np.float32)
        mvx = np.zeros((num_h, num_w), dtype=np.float32)
        mvy = np.zeros((num_h, num_w), dtype=np.float32)

        predict_gray = np.zeros_like(f0_L0, dtype=np.float32)

        half_offsets = [(-0.5,-0.5),(0.0,-0.5),(0.5,-0.5),
                        (-0.5, 0.0),(0.0, 0.0),(0.5, 0.0),
                        (-0.5, 0.5),(0.0, 0.5),(0.5, 0.5)]

        # 逐區塊搜尋
        for bi in range(num_h):
            for bj in range(num_w):

                # ------- Level 2: 0.25x 全域搜尋 ±R_L2 -------
                rL2 = bi * (N//4) + d_L2
                cL2 = bj * (N//4) + d_L2
                blk2 = f0_L2_pad[rL2:rL2+N, cL2:cL2+N]
                if blk2.shape != (N, N): 
                    continue
                best = (np.inf, 0, 0)
                for dy in range(-self.R_L2, self.R_L2+1):
                    for dx in range(-self.R_L2, self.R_L2+1):
                        t2 = f1_L2_pad[rL2+dy:rL2+dy+N, cL2+dx:cL2+dx+N]
                        if t2.shape != (N, N): 
                            continue
                        mad = np.sum(np.abs(blk2 - t2))
                        if mad < best[0]:
                            best = (mad, dy, dx)
                mvy_L2[bi, bj], mvx_L2[bi, bj] = best[1], best[2]

                # ------- Level 1: 0.5x 以 L2*2 為中心 ±R_L1 -------
                rL1 = bi * (N//2) + d_L1
                cL1 = bj * (N//2) + d_L1
                blk1 = f0_L1_pad[rL1:rL1+N, cL1:cL1+N]
                if blk1.shape != (N, N): 
                    continue
                pred_dy1 = int(mvy_L2[bi, bj]*2)
                pred_dx1 = int(mvx_L2[bi, bj]*2)
                best = (np.inf, pred_dy1, pred_dx1)
                for ddy in range(-self.R_L1, self.R_L1+1):
                    for ddx in range(-self.R_L1, self.R_L1+1):
                        y = rL1 + pred_dy1 + ddy
                        x = cL1 + pred_dx1 + ddx
                        t1 = f1_L1_pad[y:y+N, x:x+N]
                        if t1.shape != (N, N): 
                            continue
                        mad = np.sum(np.abs(blk1 - t1))
                        if mad < best[0]:
                            best = (mad, pred_dy1+ddy, pred_dx1+ddx)
                mvy_L1[bi, bj], mvx_L1[bi, bj] = best[1], best[2]

                # ------- Level 0: 1.0x 先整數精煉 ±R_L0_int -------
                rL0 = bi * N + d_L0
                cL0 = bj * N + d_L0
                blk0 = f0_L0_pad[rL0:rL0+N, cL0:cL0+N]
                if blk0.shape != (N, N): 
                    continue
                pred_dy0 = int(mvy_L1[bi, bj]*2)
                pred_dx0 = int(mvx_L1[bi, bj]*2)
                best_int = (np.inf, pred_dy0, pred_dx0)
                for ddy in range(-self.R_L0_int, self.R_L0_int+1):
                    for ddx in range(-self.R_L0_int, self.R_L0_int+1):
                        y = rL0 + pred_dy0 + ddy
                        x = cL0 + pred_dx0 + ddx
                        t0_blk = f1_L0_pad[y:y+N, x:x+N]
                        if t0_blk.shape != (N, N): 
                            continue
                        mad = np.sum(np.abs(blk0 - t0_blk))
                        if mad < best_int[0]:
                            best_int = (mad, pred_dy0+ddy, pred_dx0+ddx)

                # ------- Level 0: 半像素(±0.5) 手刻 bilinear 九宮格 -------
                best_half = (best_int[0], float(best_int[1]), float(best_int[2]))
                by, bx = best_int[1], best_int[2]
                for (hdy, hdx) in half_offsets:
                    if hdy == 0.0 and hdx == 0.0:
                        continue
                    y0 = rL0 + by + hdy
                    x0 = cL0 + bx + hdx
                    acc = 0.0
                    for rr in range(N):
                        for cc in range(N):
                            v = self._bilinear_sample(f1_L0_pad, y0+rr, x0+cc)
                            acc += abs(blk0[rr, cc] - v)
                    if acc < best_half[0]:
                        best_half = (acc, by + hdy, bx + hdx)

                # MV
                mvy[bi, bj] = best_half[1]
                mvx[bi, bj] = best_half[2]

                # 半像素重建
                y0 = rL0 + best_half[1]
                x0 = cL0 + best_half[2]
                ph = min(N, H0 - bi*N)
                pw = min(N, W0 - bj*N)
                for rr in range(ph):
                    for cc in range(pw):
                        predict_gray[bi*N + rr, bj*N + cc] = self._bilinear_sample(f1_L0_pad, y0+rr, x0+cc)

        proc = time.time() - t0
        P = psnr(tar_gray, predict_gray)
        print(f"processing time = {proc:.4f} s")
        print(f"PSNR (target_y vs predict_y) = {P:.4f} dB")

        # ===== 視覺化（全部灰圖；送入 displayFrame 前疊成 3 通道）=====
        pred_u8 = np.clip(predict_gray, 0, 255).astype(np.uint8)
        pred_bgr = cv2.merge([pred_u8, pred_u8, pred_u8])
        displayFrame(pred_bgr, self.flag, "hbma3_halfPel_predict_gray", "hbma3_halfPel_predict.jpg")

        err = np.clip(np.abs(tar_gray - predict_gray), 0, 255).astype(np.uint8)
        err_bgr = cv2.merge([err, err, err])
        # displayFrame(err_bgr, self.flag, "hbma3_halfPel_error_gray", f"hbma3_halfPel_error_gray_N{N}_Req{self.R}.jpg")

        # draw_motion_field(mvx, mvy, W, H, f"hbma3_halfPel_mv_N{N}_Req{self.R}.jpg")

        # ==== 產生並存 flow 圖（PyTorch flow_to_image）====
        mvx_interp = cv2.resize(mvx, (W0, H0), interpolation=cv2.INTER_NEAREST)
        mvy_interp = cv2.resize(mvy, (W0, H0), interpolation=cv2.INTER_NEAREST)

        dense_flow = np.zeros((H0, W0, 2), dtype=np.float32)
        dense_flow[:, :, 0] = mvx_interp
        dense_flow[:, :, 1] = mvy_interp

        flow_tensor = torch.from_numpy(dense_flow).float().permute(2, 0, 1)  # (2,H,W)
        flow_img = flow_to_image(flow_tensor).permute(1, 2, 0).numpy().astype(np.uint8)  # RGB

        base_dir = './results/'
        save_dir = os.path.join(base_dir, f'test{self.flag}')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'hbma_flow.jpg'),
                    cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))

        return mvx, mvy
