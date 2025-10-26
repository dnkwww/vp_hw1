import numpy as np
import os

from EBMA import ebma
from EBMA_halfPel import ebma_halfPel
from HBMA_threeLevel_halfPel import hbma_three_level_halfPel

import display as dp

test = ['./test/akiyo_cif.y4m', './test/foreman_cif.y4m']
# b = 'akiyo_352x288_30.yuv'

for i, video_path in enumerate(test, start=1):
    if not os.path.exists(video_path):
        print(f"[警告] 找不到影片：{video_path}")
        continue

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n===== Running EBMA on video {i}: {video_name} =====")

    # 呼叫 EBMA
    eBMA = ebma(video_path, 16, 32, i)
    mvx, mvy = eBMA.match()

    # 呼叫 eBMA_halfPel
    print(f"\n===== Running eBMA_halfPel on video {i}: {video_name} =====")
    eBMA_halfPel = ebma_halfPel(video_path, 16, 32, i)
    mvx, mvy = eBMA_halfPel.match()

    # 呼叫 hbma_halfPel
    print(f"\n===== Running hbma on video {i}: {video_name} =====")
    hbma = hbma_three_level_halfPel(video_path, 16, 32, i)
    mvx, mvy = hbma.match()
