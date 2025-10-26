# VP HW1

---

#### dataset download url:
https://media.xiph.org/video/derf/?hl=zh-TW

###
```
git clone https://github.com/dnkwww/vp_hw1.git
```

#### environment
```
pip install -r requirements.txt
```

---
#### run
```
cd vp_hw1
python src/project1.py
```

#### test results

![results_1026](./test/image.png)

:::success

===== Running EBMA on video 1: akiyo_cif =====
估算總幀數: 300
processing time = 13.2596 s
PSNR (target_y vs predict_y) = 35.2917 dB

===== Running eBMA_halfPel on video 1: akiyo_cif =====
processing time = 29.0489 s
PSNR (target_y vs predict_y) = 43.9340 dB

===== Running hbma on video 1: akiyo_cif =====
processing time = 15.7779 s
PSNR (anchor_gray vs predict_gray) = 35.8798 dB

===== Running EBMA on video 2: foreman_cif =====
估算總幀數: 300
processing time = 18.7184 s
PSNR (target_y vs predict_y) = 21.0730 dB

===== Running eBMA_halfPel on video 2: foreman_cif =====
processing time = 28.0230 s
PSNR (target_y vs predict_y) = 20.1686 dB

===== Running hbma on video 2: foreman_cif =====
processing time = 14.4709 s
PSNR (anchor_gray vs predict_gray) = 21.6273 dB

:::

