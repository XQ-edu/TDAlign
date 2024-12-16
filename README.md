# TDAlign

Welcome to the official repository of the TDAlign paper: Modeling Temporal Dependencies within the Target for Long-Term Time Series Forecasting.

TDAlign is a plug-and-play framework without introducing additional learnable parameters to the baseline. It incurs minimal computational overhead, with only linear time complexity and constant space complexity relative to the prediction length.

## Getting Started

### Environment Preparation
```bash
pip install -r requirements.txt
```

### Data Preparation
All seven datasets for TDAlign are available at [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by [Autoformer](https://github.com/thuml/Autoformer).

### Running Experiments
We provide ready-to-use scripts for all major experiments, ensuring easy integration and reproducibility.
- Although these baselines maintain a consistent prediction length for each dataset, the required input lengths may vary depending on the architecture and training strategies of each model. 
- We do not use the **"Drop Last"** operation during testing for all experiments, as suggested by [TFB](https://arxiv.org/pdf/2403.20150).
- The `scripts/origin` folder contains scripts for baselines.
- The `scripts/improve` folder contains scripts for baselines enhanced with TDAlign. 

Example commands:
```bash
bash scripts/origin/run_itransformer_origin.sh
bash scripts/improve/run_itransformer_improve.sh
```

### Develop Your Own Models

Develop steps:
- Add the model file to the folder `models`.
- Include the newly added model in `exp/exp_main_improve.py`.
- Create the corresponding scripts under the folder `scripts/improve`.


## Citation
If you find this repo useful, please cite our paper.
```
@article{xiong2024tdt,
  title={Modeling Temporal Dependencies within the Target for Long-Term Time Series Forecasting},
  author={Xiong, Qi and Tang, Kai and Ma, Minbo, Zhang, Ji and Xu, Jie and Li, Tianrui},
  journal={arXiv preprint arXiv:2406.04777},
  year={2024}
}
```

## Acknowledgement
We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases or datasets:
- [Are Transformers Effective for Time Series Forecasting?](https://github.com/cure-lab/LTSF-Linear)
- [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://github.com/yuqinie98/PatchTST)
- [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://github.com/thuml/iTransformer)
- [MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting](https://github.com/wanghq21/MICN)
- [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)
- [SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting](https://github.com/lss-1138/SegRNN)
- [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://github.com/thuml/Autoformer)
