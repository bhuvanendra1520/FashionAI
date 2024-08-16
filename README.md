# Text To Texture

To generate fashion images from the given text description we have adopted state of the art AttnGAN model for our implementation.

# AttnGAN (Python 3, Pytorch 1.0)

Pytorch implementation for reproducing AttnGAN results in the paper [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) by Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He. 

![framework](https://github.com/ChaitanyaGhadling/AttentionGAN/assets/55136558/bb34d72d-a0a9-4dac-87de-b9242dc61f42)
 

### Dependencies
python 3.6+

Pytorch 1.0+

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`



**Data**
In this study, we have used DeepFashion MultiModal dataset, which is a famous dataset in the field of the fashion industry. Four benchmarks are developed by this database that include Attribute Prediction, Consumer-to-shop Clothes
Retrieval, In-shop Clothes Retrieval, and Landmark Detection.

Link to the dataset : https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html


**Training**
- Pre-train DAMSM models:
  - `python pretrain_DAMSM.py --cfg cfg/DAMSM/fashion.yml --gpu 0`
 
- Train AttnGAN models:
  - `python main.py --cfg cfg/fashiom_attn2.yml --gpu 0`

- `*.yml` files are example configuration files for training/evaluation our models.


**Sampling**
- Run `python main.py --cfg cfg/eval_fashion.yml --gpu 0` to generate examples from captions in files listed in "./data/fashion/example_filenames.txt".  
- Input your own sentence in "./data/fashion/example_captions.txt" if you wannt to generate images from customized sentences. 

**Validation**
- To generate images for all captions in the validation dataset, change B_VALIDATION to True in the eval_*.yml. and then run `python main.py --cfg cfg/eval_fashion.yml --gpu 0`
- We compute inception score for models trained on birds using [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model).
- We compute inception score for models trained on coco using [improved-gan/inception_score](https://github.com/openai/improved-gan/tree/master/inception_score).


### Creating an API
[Evaluation code](eval) embedded into a callable containerized API is included in the `eval\` folder.


**Reference**

- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916) [[code]](https://github.com/hanzhanggit/StackGAN-v2)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) [[code]](https://github.com/carpedm20/DCGAN-tensorflow)
