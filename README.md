# CompFashionIQ

A code base for conducting image retrieval with text feedback experiments on FashionIQ.

Available methods:

- [TIRG (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vo_Composing_Text_and_Image_for_Image_Retrieval_-_an_Empirical_CVPR_2019_paper.pdf)
- [VAL (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Image_Search_With_Text_Feedback_by_Visiolinguistic_Attention_Learning_CVPR_2020_paper.pdf)
- [CoSMo (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_CoSMo_Content-Style_Modulation_for_Image_Retrieval_With_Text_Feedback_CVPR_2021_paper.pdf)
- [RTIC w/o GCN (arXiv 2021)](https://arxiv.org/pdf/2104.03015.pdf)

## Data Preparation

[Here](https://drive.google.com/file/d/1GYeaPjBsLjOavTWcA0bj8U7ZIyKshK7s/view?usp=sharing) is my pre-processed data (including original images and generated vocab embeddings).

Download and unzip into `./train_data`.

## Running

``` bash
python train_net.py --config-file configs/fashioniq/tirg_bigru_init.yaml --use-tensorboard
```

## Results

Please refer to [this table](https://docs.google.com/spreadsheets/d/1l9CRWC7SwPEWE-Z0oWB_0PvnhCAB-V2ZZ4XsQ-cKpLA/edit#gid=0).
