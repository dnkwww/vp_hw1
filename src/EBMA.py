import numpy as np
import cv2  # 新增：用於 BGR→YCrCb 取得 Y 通道
from getYuvFrame import getYUVFrame
from display import displayFrame, yuv2bgr, psnr
import time
import os
import torch
from torchvision.utils import flow_to_image


def bgr_to_y(img_bgr):
    """
    將 BGR 影像轉為 Y 通道（YCrCb 的 Y）。
    輸入可為 uint8 或 float；此處轉 uint8 再轉回 float32，確保 cvtColor 正常。
    """
    img_u8 = img_bgr.astype(np.uint8)
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    return y


# exhaustive block matching algorithm
class ebma():
    # N: block size; R: search range
    def __init__(self, video, N, R, flag):
        self.video = video
        self.N = N   # block size
        self.R = R  # search range
        self.flag = flag

    def match(self):
        start_time = time.time()
        # video frame width and height
        width = 352
        height = 288
        N = self.N
        search_params = '_'+str(N)+'_'+str(self.R)+'.jpg'

        frames = getYUVFrame(self.video, width, height)
    
        import os
        file_size = os.path.getsize(self.video)
        frame_bytes = int(1.5 * width * height)
        total_frames = file_size // frame_bytes
        print("估算總幀數:", total_frames)

        anchor = yuv2bgr(frames.getFrame(29))   # anchor frame
        target = yuv2bgr(frames.getFrame(30))   # target frame

        d = np.maximum(self.N, self.R)

        # padding：維持原 B 的 zero padding（可保留）
        anchor_2 = np.pad(anchor, ((d, d), (d, d), (0, 0)), 'constant', constant_values=0)
        target_2 = np.pad(target, ((d, d), (d, d), (0, 0)), 'constant', constant_values=0)

        # --- 全流程用 Y 通道做 MAD 與重建 ---
        f1 = bgr_to_y(anchor_2)  # anchor 的 Y (float32)
        f2 = bgr_to_y(target_2)  # target 的 Y (float32)

        numWidthBlks  = int(np.ceil(width  / N))
        numHeightBlks = int(np.ceil(height / N))

        # MV
        mvx = np.zeros([numHeightBlks, numWidthBlks], dtype=np.float32)
        mvy = np.zeros([numHeightBlks, numWidthBlks], dtype=np.float32)

        # 灰階預測影像（單通道）
        predict_gray = np.zeros((height, width), dtype=np.float32)

        for ii in range(d, d - 1 + height, N):
            for jj in range(d, d - 1 + width, N):  # every block in the anchor frame
                MAD_min = 256 * N * N
                dx = 0; dy = 0

                for kk in range(-self.R, self.R + 1):
                    for ll in range(-self.R, self.R + 1):  # every search candidate
                        MAD = np.sum(np.abs(f1[ii: ii+N, jj: jj+N] - 
                                            f2[ii+kk: ii+kk+N, jj+ll: jj+ll+N]))
                        if MAD < MAD_min:
                            MAD_min = MAD
                            dy = kk
                            dx = ll

                # --- 用 target 的 Y 填灰階預測影像 ---
                predict_gray[ii-d: ii-d+N, jj-d: jj-d+N] = \
                    f2[ii+dy: ii+dy+N, jj+dx: jj+dx+N]

                # 記錄 MV
                iblk = int(np.floor((ii - d) / N))
                jblk = int(np.floor((jj - d) / N))
                mvx[iblk, jblk] = dx
                mvy[iblk, jblk] = dy

        # print('integer-EBMA:')
        process_time = time.time() - start_time
        print(f'processing time = {process_time:.4f} s')

        # 顯示/輸出：把灰階疊成 3 通道再顯示
        predict_u8 = predict_gray.clip(0, 255).astype(np.uint8)
        displayFrame(cv2.merge([predict_u8]*3), self.flag, 'predict_gray', 'ebma_predict.jpg')

        # PSNR 與誤差圖：灰階對灰階
        target_gray  = bgr_to_y(target)                   # H×W float32
        error_gray = (np.abs(target_gray - predict_gray)).clip(0, 255).astype(np.uint8)

        psnr_val = psnr(target_gray, predict_gray)
        # displayFrame(cv2.merge([error_gray]*3), self.flag, 'error_gray', 'ebma_error_gray' + search_params)
        print(f'PSNR (target_y vs predict_y) = {psnr_val:.4f} dB')

        # MV 視覺化維持不變
        # draw_motion_field(mvx, mvy, width, height, 'ebma_mv' + search_params)

        # ==== 產生並存 flow 圖（PyTorch flow_to_image）====
        # 將區塊級 MV 放大到像素網格
        mvx_interp = cv2.resize(mvx, (width, height), interpolation=cv2.INTER_NEAREST)
        mvy_interp = cv2.resize(mvy, (width, height), interpolation=cv2.INTER_NEAREST)

        dense_flow = np.zeros((height, width, 2), dtype=np.float32)
        dense_flow[:, :, 0] = mvx_interp
        dense_flow[:, :, 1] = mvy_interp

        flow_tensor = torch.from_numpy(dense_flow).float().permute(2, 0, 1)  # (2,H,W)
        flow_img = flow_to_image(flow_tensor).permute(1, 2, 0).numpy().astype(np.uint8)  # RGB

        # 存檔到 ./results/test{flag}/ebma_flow.jpg
        base_dir = './results/'
        save_dir = os.path.join(base_dir, f'test{self.flag}')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'ebma_flow.jpg'),
                    cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))

        return mvx, mvy

