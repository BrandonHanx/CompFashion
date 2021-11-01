# CompFashion

A code base for conducting image retrieval with text feedback experiments.

Available methods:


- For visually compitable garment retrieval:
   - [CSA-Net (CVPR2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Fashion_Outfit_Complementary_Item_Retrieval_CVPR_2020_paper.pdf)

- For text-guided garment retrieval:
   - [TIRG (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vo_Composing_Text_and_Image_for_Image_Retrieval_-_an_Empirical_CVPR_2019_paper.pdf)
   - [VAL (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Image_Search_With_Text_Feedback_by_Visiolinguistic_Attention_Learning_CVPR_2020_paper.pdf)
   - [CoSMo (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_CoSMo_Content-Style_Modulation_for_Image_Retrieval_With_Text_Feedback_CVPR_2021_paper.pdf)
   - [RTIC w/o GCN (arXiv 2021)](https://arxiv.org/pdf/2104.03015.pdf)

## Data Preparation

1. Download FashionIQ dataset according to [this](https://github.com/XiaoxiaoGuo/fashion-iq). Then move data into `train_data/fashioniq/`.

2. Build vocabulary.
   ```bash
   python -m nltk.downloader 'punkt'
   python build_vocab.py
   ```

3. (Optional) Generate GloVe embeddings.
   ```bash
   python -m spacy download en_vectors_web_lg
   python gen_glove.py
   ```

## Running

``` bash
python train_net.py --config-file configs/fashioniq/tirg_bigru_init.yaml --use-tensorboard
```

## Results

coming soon...

## Different Settings in Original Methods

1. VAL uses smaller val split and GloVe.
2. CoSMo constructs captions with both orders for concatenating.
3. ...
