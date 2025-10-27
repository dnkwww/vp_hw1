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

![123][image.png]

===== Running EBMA on video 1: akiyo_cif =====
估算總幀數: 300
processing time = 13.5905 s
PSNR (target_y vs predict_y) = 32.3723 dB

===== Running eBMA_halfPel on video 1: akiyo_cif =====
processing time = 22.7277 s
PSNR (target_y vs predict_y) = 35.2301 dB

===== Running hbma on video 1: akiyo_cif =====
processing time = 13.8045 s
PSNR (target_y vs predict_y) = 35.2293 dB

===== Running EBMA on video 2: foreman_cif =====
估算總幀數: 300
processing time = 12.1064 s
PSNR (target_y vs predict_y) = 20.0760 dB

===== Running eBMA_halfPel on video 2: foreman_cif =====
processing time = 20.8205 s
PSNR (target_y vs predict_y) = 20.7797 dB

===== Running hbma on video 2: foreman_cif =====
processing time = 12.6398 s
PSNR (target_y vs predict_y) = 20.6537 dB



[def]: ./results/image.png