# Correlative Channel-Aware Fusion for Multi-View Time Series Classification (C2AF)

This repository is for our AAAI'21 paper:
> Correlative Channel-Aware Fusion for Multi-View Time Series Classification [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/16830) \
> [Yue Bai](https://yueb17.github.io/), [Lichen Wang](https://sites.google.com/site/lichenwang123/), [Zhiqiang Tao](http://ztao.cc/), [Sheng Li](https://sheng-li.org/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/)

This paper proposes to use a learnable fusion strategy to enhance the multi-view time series classification performance. The proposed correlative channel-aware fusion module can be simply realized by a convolutional filter yet effective for final fusion performance. We originally implement it using Tensorflow but it can be easily revised for Pytorch platform.

## Usage
Please directly check the demo.py file which contains the whole pipeline to train the model.

## Extracted feature
In our work, we use a newly proposed multi-view action dataset ([EV-Action](https://arxiv.org/pdf/1904.12602.pdf)). Here, we provide our extracted feature for usage, where RGB, depth, and skeleton features are aligned in temporal dimension and can be used directly to train a model. Please check the google drive link as below.

[Extracted EV-Action feature](https://drive.google.com/drive/folders/1AdtB0AwhkUqk8A0xrVPiJm9L2lfb-NRD?usp=sharing)


## Reference
Please cite this in your publication if our code or feature helps your research. Should you have any questions, welcome to reach out to Yue Bai (bai.yue@northeastern.edu).

```
@inproceedings{bai2021correlative,
  title={Correlative channel-aware fusion for multi-view time series classification},
  author={Bai, Yue and Wang, Lichen and Tao, Zhiqiang and Li, Sheng and Fu, Yun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={8},
  pages={6714--6722},
  year={2021}
}
```
