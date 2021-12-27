# DomainAdaption4DependencyParsing

Pytorch implementations of model-centric and data-centric domain-adaption techniques for dependency parsing.

## Datasets
[NLPCC-2019](http://hlt.suda.edu.cn/index.php/Nlpcc-2019-shared-task)依存句法分析领域移植评测任务数据集
源域：平衡语料(BC)  
目标域：
  + 淘宝产品博客(PB)
  + 淘宝产品评论(PC)
  + 网络小说《诛仙》(ZX)

## Core Methods
+ Domain-Adaptive Pretraining：MLM + Finetuning
+ Teacher-Student Learning：pretrained teacher + to-be-trained student
+ Multi-task Tri-training：shared BERT + separated Biaffines


## Others
+ Full Finetuning
+ DistanceNet：Dual BERT + domain distance metrics
+ Domain Adversarial Training：GRL + domain classification
+ VAE Reconstruction：autoencoder
