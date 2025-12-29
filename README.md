# Awesome Industrial & Medical Anomaly Detection

A curated list of papers, datasets, and code resources for anomaly detection in industrial inspection and medical imaging.

---


## Table of Contents



- ⭐⭐ *(2019-CVPR)* **** [[Paper]](), [[ArXiv]](), [[Code]]()
  - Authors: 
  - Brain 
  - Datasets: 





# [Industrial Anomaly Detection](#industrial-anomaly-detection)




## Unsupervised Anomaly Detection Methods


### One-class Classification Methods



- ⭐⭐⭐ *(2022-AAAI)* **Deep One-Class Classification via Interpolated Gaussian Descriptor** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/19915), [[ArXiv]](https://arxiv.org/abs/2101.10043), [[Code]](https://github.com/tianyu0207/IGD)
  - Authors: Yuanhong Chen, Yu Tian, Guansong Pang, Gustavo Carneiro
  - Bralearns a smooth Gaussian descriptor using adversarially interpolated samples so the model can form a robust, compact description of normal data and avoid overfitting or contamination issues.in 
  - Datasets: MNIST, Fashion MNIST, CIFAR-10, MVTec AD, Hyper-Kvasir, LAG





- ⭐⭐ *(2020-ACCV)* **Patch SVDD: Patch-level SVDD for Anomaly Detection and Segmentation** [[Paper]](https://openaccess.thecvf.com/content/ACCV2020/papers/Yi_Patch_SVDD_Patch-level_SVDD_for_Anomaly_Detection_and_Segmentation_ACCV_2020_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2006.16067), [[Code]](https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch)
  - Authors: Jihun Yi, Sungroh Yoon
  - extends deep SVDD to a patch-based, self-supervised setup so the model can both detect abnormal images and pinpoint faulty regions with much higher accuracy. 
  - Datasets: MVTec AD



- ⭐⭐⭐ *(2018-ICLR)* **Deep One-Class Classification** [[Paper]](https://proceedings.mlr.press/v80/ruff18a.html), [[Code]](https://github.com/lukasruff/Deep-SVDD-PyTorch)
  - Authors: Lukas Ruff, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel Müller, Marius Kloft 
  - a deep model trained directly on a one-class objective to tightly enclose normal data, enabling effective identification of inputs that fall outside this learned boundary. 
  - Datasets: MNIST, CIFAR-10

### Reconstruction-based Methods


- ⭐⭐ *(2025-CVPR)* **Exploring Intrinsic Normal Prototypes within a Single Image for  Universal Anomaly Detection** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32601), [[ArXiv]](https://arxiv.org/pdf/2503.02424), [[Code]](https://github.com/luow23/INPFormer)
  - Authors: Wei Luo, Yunkang Cao, Haiming Yao, Xiaotian Zhang, Jianan Lou, Yuqi Cheng, Weiming Shen, Wenyong Yu
  - extracts the intrinsic normal prototypes (INPs) from the image and integrate them into a transformer-based reconstruction model to improve the quality of reconstruction.
  - Datasets: MVTec-AD, VisA, Real-IAD


- ⭐⭐ *(2025-AAAI)* **Unlocking the Potential of Reverse Distillation for Anomaly Detection** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/32601), [[ArXiv]](https://arxiv.org/pdf/2406.07487), [[Code]](https://github.com/hyao1/GLAD)
  - Authors: Xinyue Liu, Jianyuan Wang, Biao Leng, Shuo Zhang
  - strengthens reversed distillation by adding an expert network and guided feature transfer so the model can better distinguish abnormal features and reconstruct clean normal patterns with fewer errors. 
  - Datasets: MVTec AD, MPDD, BTAD


- ⭐⭐ *(2024-ECCV)* **GLAD: Towards Better Reconstruction with  Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection** [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08940-supp.pdf), [[ArXiv]](https://arxiv.org/abs/2412.07579), [[Code]](https://github.com/hito2448/urd)
  - Authors: Hang Yao, Ming Liu, Zhicun Yin, Zifei Yan, Xiaopeng Hong, and Wangmeng Zuo
  - strengthens the reconstruction process of the diffusion model from global and local perspectives.
  - Datasets: MVTec-AD, MPDD, VisA, PCB-Bank



- ⭐⭐⭐ *(2023-NeurIPS)* **ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction** [[Paper]](https://openreview.net/pdf?id=KYxD9YCQBH), [[ArXiv]](https://arxiv.org/abs/2306.02602), [[Code]](https://github.com/guojiajeremy/ReContrast)
  - Authors: Jia Guo, Shuai Lu, Lize Jia, Weihang Zhang, Huiqi Li
  - fine-tunes both encoder and decoder with a contrastive reconstruction strategy to remove biases from pre-trained features and adapt the whole model to the target domain for more accurate identification of unusual patterns. 
  - Datasets: MVTec AD, Visa, OCT2017, APTOS, ISIC2018



- ⭐⭐ *(2023-CVPR)* **Diversity-Measurable Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Diversity-Measurable_Anomaly_Detection_CVPR_2023_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2303.05047), [[Code]](https://github.com/FlappyPeggy/DMAD)
  - Authors: Wenrui Liu, Hong Chang, Bingpeng Ma, Shiguang Shan, Xilin Chen
  - boosts reconstruction diversity for normal patterns using multi-scale deformation fields while preventing abnormal details from being learned, leading to more reliable detection of unusual inputs. 
  - Datasets: Ped2, Avenue, ShanghaiTech, MVTec AD




- ⭐⭐ *(2023-CVPR)* **Revisiting Reverse Distillation for Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf), [[Code]](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)
  - Authors: Tien, Tran Dinh and Nguyen, Anh Tuan and Tran, Nguyen Hoang and Huy, Ta Duc and Duong, Soan T.M. and Nguyen, Chanh D. Tr. and Truong, Steven Q. H.
  - enhances reversed distillation to create a faster, more accurate system for spotting and localizing defects without relying on slow external memory banks. 
  - Datasets: MVTec AD, BTAD, Retinal OCT




- ⭐⭐⭐ *(2022-CVPR)* **Anomaly Detection via Reverse Distillation from One-Class Embedding** [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2201.10703), [[Code]](https://github.com/hq-deng/RD4AD)
  - Authors: Hanqiu Deng, Xingyu Li
  - reverses teacher–student distillation by having the student reconstruct the teacher’s multi-level features from a compact one-class embedding, yielding richer cues for spotting deviations from normality. 
  - Datasets: MVTec AD, MNIST, CIFAR-10, Fashion-MNIST, Caltech-256


- ⭐⭐ *(2022-CVPR)* **Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2111.09099), [[Code]](https://github.com/ristea/sspcab)
  - Authors: Nicolae-Catalin Ristea, Neelu Madan, Radu Tudor Ionescu, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B. Moeslund, Mubarak Shah
  - a self-supervised prediction block that reconstructs masked regions within each receptive field, boosting the performance of many existing frameworks for spotting unusual patterns. 
  - Datasets: MVTec AD, CHUK Avenue, ShanghaiTech
  - Extend to PAMI: *Self-Supervised Masked Convolutional Transformer Block for Anomaly Detection* [[Paper]](https://ieeexplore.ieee.org/document/10273635/), [[ArXiv]](https://arxiv.org/abs/2209.12148), [[Code]](https://github.com/ristea/ssmctb)




- ⭐ *(2022-TMM)* **Self-Supervised Masking for Unsupervised Anomaly Detection and Localization** [[Paper]](https://ieeexplore.ieee.org/document/9779083), [[ArXiv]](https://arxiv.org/abs/2205.06568)
  - Authors: Chaoqin Huang, Qinwei Xu, Yanfeng Wang, Yu Wang, Ya Zhang
  - rains a model to restore randomly masked images and then uses progressive mask refinement at test time to efficiently uncover normal areas and pinpoint abnormal regions. 
  - Datasets: Retinal-OCT, MVTec AD


- ⭐ *(2021-CVPR)* **Glancing at the Patch: Anomaly Localization with Global and Local Feature Comparison** [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Glancing_at_the_Patch_Anomaly_Localization_With_Global_and_Local_CVPR_2021_paper.pdf)
  - Authors: Shenzhi Wang, Liwei Wu, Lei Cui, Yujun Shen
  - compares each patch’s features with those of its surrounding context using global–local networks and multi-head discrepancy scoring to precisely locate abnormal regions. 
  - Datasets: MVTec AD, CIFAR-10




- ⭐ *(2021-PR)* **Reconstruction by inpainting for visual anomaly detection** [[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320320305094), [[Unoff-Code]](https://github.com/plutoyuxie/Reconstruction-by-inpainting-for-visual-anomaly-detection)
  - Authors: Vitjan Zavrtanik, Matej Kristan, Danijel Skčaj
  - reframes reconstruction as an inpainting task so the model learns to fill in missing regions from normal patterns, making anomalies harder to reconstruct and therefore easier to detect. 
  - Datasets: MVTec AD, UCSD Ped2, CUHK Avenue





- ⭐⭐ *(2020-CVPR)* **Learning Memory-guided Normality for Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2003.13228), [[Code]](https://github.com/cvlab-yonsei/MNAD)
  - Authors: Hyunjong Park, Jongyoun Noh, Bumsub Ham
  - limits the model’s ability to reconstruct unusual events by using a memory module that stores diverse normal patterns and enforces compact, well-separated features, making deviations easier to spot. 
  - Datasets: UCSD Ped2, CUHK Avenue, ShanghaiTech



- ⭐ *(2020-TMM)* **Attribute Restoration Framework for Anomaly Detection** [[Paper]](https://ieeexplore.ieee.org/document/9311201), [[ArXiv]](https://arxiv.org/abs/1911.10676)
  - Authors: Chaoqin Huang, Fei Ye, Jinkun Cao, Maosen Li, Ya Zhang, Cewu Lu
  - forces the model to restore images with missing attributes so it must learn meaningful semantics, causing abnormal inputs to produce larger restoration errors. 
  - Datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR100 and ImageNet, MVTec AD, ShanghaiTech



- ⭐⭐⭐ *(2019-ICCV)* **Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf), [[ArXiv]](https://arxiv.org/abs/1904.02639), [[Code]](https://github.com/donggong1/memae-anomaly-detection)
  - Authors: Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, Anton van den Hengel
  - enhances autoencoders with a memory module that reconstructs inputs using stored normal patterns, making abnormal inputs stand out through larger reconstruction errors. 
  - Datasets: MNIST, CIFAR-10, UCSD-Ped2, CUHK Avenue, ShanghaiTech, KDDCUP99


- ⭐ *(2019-AAAI)* **Learning Competitive and Discriminative Reconstructions for Anomaly Detection** [[Paper]](https://cdn.aaai.org/ojs/4451/4451-13-7490-1-10-20190706.pdf), [[ArXiv]](https://arxiv.org/abs/1903.07058)
  - Authors: Kai Tian, Shuigeng Zhou, Jianping Fan, Jihong Guan
  - trains an encoder with two competing decoders that separately model normal and abnormal patterns, allowing the system to infer labels for unlabeled data without needing a hand-crafted threshold. 
  - Datasets: KDD99, MNIST, CIFAR-10, Fashion-MNIST, ImageNet-20, Caltech-101, Caltech-256




- ⭐⭐ *(2018-ICLR)* **Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection** [[Paper]](https://openreview.net/forum?id=BJJLHbb0-), [[ArXiv]](), [[Unoff-Code]](https://github.com/danieltan07/dagmm)
  - Authors: Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, Haifeng Chen
  - unifies autoencoding and density modeling into a single learned system so it can better capture normal data structure and flag points that fall outside this learned distribution. 
  - Datasets: KDDCUP, Thyroid, Arrhythmia, KDDCUP-Rev





### Synthesis-based/Data Augmentation Methods

- ⭐⭐ *(2025-NeurIPS)* **FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis**[[ArXiv]](https://arxiv.org/pdf/2509.20295), [[Code]](https://github.com/Chhro123/fast-foreground-aware-anomaly-synthesis)
  - Authors: Xiaolei Wang, Xiaoyang Wang, Huihui Bai, Eng Gee Lim, Jimin Xiao
  - improves the speed of anomaly synthesis and differentiate between the foreground and background of images. 
  - Datasets: MVTec-AD, BTAD


- ⭐ *(2025-ICCV)* **DecAD: Decoupling Anomalies in Latent Space for Multi-Class Unsupervised Anomaly Detection**[[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_DecAD_Decoupling_Anomalies_in_Latent_Space_for_Multi-Class_Unsupervised_Anomaly_ICCV_2025_paper.pdf)
  - Authors: Xiaolei Wang, Xiaoyang Wang, Huihui Bai, Eng Gee Lim, Jimin Xiao
  - introduces adversarial training to generate more realistic synthetic anomalies, and then use INN (Invertible Neural Network) to ensure that the model can only reconstruct normal features. 
  - Datasets: MVTec AD, VisA, Real-IAD


- ⭐⭐⭐ *(2023-CVPR)* **SimpleNet: A Simple Network for Image Anomaly Detection and Localization** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2303.15140), [[Code]](https://github.com/DonaldRR/SimpleNet)
  - Authors: Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang
  - a lightweight system that adapts pretrained features to the target domain and discriminates between normal and noise-perturbed features, achieving fast and highly accurate detection of abnormal patterns. 
  - Datasets: MVTec AD, CIFAR10


- ⭐⭐ *(2023-CVPR)* **DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2211.11317), [[Code]](https://github.com/apple/ml-destseg)
  - Authors: Xuan Zhang, Shiyu Li, Xi Li, Ping Huang, Jiulong Shan, Ting Chen
  - strengthens teacher–student learning with denoising and supervised segmentation so the model can more robustly distinguish abnormal patterns and localize them accurately. 
  - Datasets:  MVTec AD





- ⭐⭐⭐ *(2021-ICCV)* **DRAEM -- A discriminatively trained reconstruction embedding for surface anomaly detection** [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2108.07610), [[Code]](https://github.com/VitjanZ/DRAEM)
  - Authors: Vitjan Zavrtanik, Matej Kristan, Danijel Skočaj
  - learns a joint representation of an image and its reconstruction to directly separate normal from abnormal regions, enabling accurate defect localization without heavy post-processing. 
  - Datasets: MVTec AD, DAGM





- ⭐⭐ *(2021-CVPR)* **CutPaste: Self-Supervised Learning for Anomaly Detection and Localization** [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html), [[ArXiv]](https://arxiv.org/abs/2104.04015), [[Code]](https://github.com/LilitYolyan/CutPaste)
  - Authors: Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, Tomas Pfister
  - learns features by training a model to spot CutPaste perturbations and then builds a one-class classifier on top, enabling strong detection of unseen defects using only normal data. 
  - Datasets: MVTec AD



- ⭐⭐ *(2018-NeurIPS)* **Deep Anomaly Detection Using Geometric Transformations** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2018/file/5e62d03aec0d17facfc5355dd90d441c-Paper.pdf), [[ArXiv]](https://arxiv.org/abs/1805.10917), [[Code]](https://github.com/izikgo/AnomalyDetectionTransformations)
  - Authors: Izhak Golan, Ran El-Yaniv
  - trains a model to recognize many geometric transformations, and this learned visual understanding helps it spot images that don't fit the normal class. 
  - Datasets:  CIFAR-10, CIFAR-100, Fashion-MNIST, CatsVsDogs




### Feature-based/Memory Methods











## Supervised/Openset/Noise Anomaly Detection Methods

### Supervised/Openset Methods


- ⭐⭐⭐ *(2025-NeurIPS)* **Normal-Abnormal Guided Generalist Anomaly Detection** [[ArXiv]](https://arxiv.org/pdf/2510.00495), [[Code]](https://github.com/JasonKyng/NAGL)
  - Authors: Yuexin Wang, Xiaolei Wang, Yizheng Gong, Jimin Xiao 
  - a memory-based method utilizes normal and abnormal as reference.
  - Datasets: MVTecAD, VisA, BraTS


- ⭐⭐⭐ *(2025-ICCV)* **MultiADS: Defect-aware Supervision for Multi-type Anomaly Detection and  Segmentation in Zero-Shot Learning** [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Sadikaj_MultiADS_Defect-aware_Supervision_for_Multi-type_Anomaly_Detection_and_Segmentation_in_ICCV_2025_paper.pdf), [[Code]](https://github.com/boschresearch/MultiADS)
  - Authors: Ylli Sadikaj, Hongkuan Zhou, Lavdim Halilaj, Stefan Schmid, Steffen Staab, Claudia Plant
  - perform multi-classification based on different types of anomalies.
  - Datasets: MVTec-AD, VisA, MPDD, MAD, RealIAD


- ⭐⭐ *(2025-CVPR)* **Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jung_TailedCore_Few-Shot_Sampling_for_Unsupervised_Long-Tail_Noisy_Anomaly_Detection_CVPR_2025_paper.pdf), [[Code]](https://github.com/fuyunwang/DPDL)
  - Authors: Fuyun Wang, Tong Zhang, Yuanzhi Wang, Yide Qiu, Xin Liu, Xu Guo, Zhen Cui
  - a prototype-driven latent space that tightly clusters normal samples while pushing abnormal ones away, enabling clearer separation even with limited anomaly examples. 
  - Datasets: MVTec AD, Optical, SDD, AITEX, ELPV, Mastcam, BrainMRI, HeadCT, Hyper-Kvasir


- ⭐ *(2025-AAAI)* **Qsco: A Quantum Scoring Module for Open-set Supervised Anomaly Detection** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/34190), [[ArXiv]](https://arxiv.org/abs/2405.16368)
  - Authors: Yifeng Peng, Xinyi Li, Zhiding Liang, Ying Wang
  - a quantum-enhanced scoring module that uses variational quantum circuits to better handle uncertainty and improve the detection of unfamiliar abnormal patterns. 
  - Datasets:  MVTec AD, AITEX, SDD, ELPV, Optical, Mastcam, BrainMRI, HeadCT, Hyper-Kvasir



- ⭐⭐ *(2024-CVPR)* **Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Anomaly_Heterogeneity_Learning_for_Open-set_Supervised_Anomaly_Detection_CVPR_2024_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2310.12790), [[Code]](https://github.com/mala-lab/AHL)
  - Authors: Jiawen Zhu, Choubo Ding, Yu Tian, Guansong Pang
  - Simulates diverse abnormal patterns so models can learn a more general notion of abnormality and better handle unseen, unpredictable defect types. 
  - Datasets: MVTec AD, AITEX, SDD, ELPV, Optical, Mastcam, BrainMRI, HeadCT, Hyper-Kvasir



- ⭐ *(2024-CVPR)* **Supervised Anomaly Detection for Complex Industrial Images** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.html), [[ArXiv]](https://arxiv.org/abs/2405.04953), [[Code]](https://github.com/abc-125/segad)
  - Authors: Aimira Baitieva, David Hurych, Victor Besnier, Olivier Bernard
  - A new industrial defect dataset and a method that combines pixel-level signals with a boosted classifier to better identify real manufacturing defects. 
  - Datasets: Valeo Anomaly Dataset (VAD), VisA



- ⭐⭐ *(2024-TIP)* **Target before Shooting: Accurate Anomaly Detection and Localization under One Millisecond via Cascade Patch Retrieval** [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10678861), [[ArXiv]](https://arxiv.org/abs/2308.06748), [[Code]](https://github.com/flyinghu123/CPR)
  - Authors: Hanxi Li, Jianfei Hu, Bo Li, Hao Chen, Yongbin Zheng, Chunhua Shen
  - a fast cascade retrieval system that finds the most similar reference images and patches to score test samples, achieving both top accuracy and extremely high processing speed. 
  - Datasets: MVTec AD, MVTec-3D AD, BTAD




- ⭐ *(2023-CVPR)*  **Prototypical Residual Networks for Anomaly Detection and Localization** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2212.02031), [[Unoff-Code]](https://github.com/xcyao00/PRNet)
  - Authors: Hui Zhang, Zuxuan Wu, Zheng Wang, Zhineng Chen, Yu-Gang Jiang
  - learns multi-scale residual patterns to highlight defective regions and uses diverse synthetic variations to help the system generalize beyond a few available abnormal examples. 
  - Datasets: MVTec AD, DAGM, BTAD, KolektorSDD2




- ⭐⭐ *(2023-CVPR)* **Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yao_Explicit_Boundary_Guided_Semi-Push-Pull_Contrastive_Learning_for_Supervised_Anomaly_Detection_CVPR_2023_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2207.01463), [[Code]](https://github.com/xcyao00/BGAD)
  - Authors: Xincheng Yao, Ruoqi Li, Jing Zhang, Jun Sun, Chongyang Zhang
  - a boundary-guided contrastive learning method that uses a few known abnormal examples to learn clearer feature separation while avoiding bias toward those seen abnormalities
  - Datasets: MVTecAD, BTAD, AITEX, ELPV, BrainMRI, HeadCT



- ⭐⭐⭐ *(2022-CVPR)* **Catching Both Gray and Black Swans: Open-Set Supervised Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2203.14506), [[Code]](https://github.com/Choubo/DRA)
  - Authors: Choubo Ding, Guansong Pang, Chunhua Shen
  - separates different types of abnormal cues—both known and synthetically created—to help the model recognize not only familiar but also entirely new forms of abnormality. 
  - Datasets: MVTec AD, AITEX, SDD, ELPV, Optical, Mastcam, BrainMRI, HeadCT, Hyper-Kvasir


- ⭐⭐ *(2021-ICLR)* **Explainable Deep One-Class Classification** [[Paper]](https://openreview.net/forum?id=A5VV3UyIQz), [[ArXiv]](https://arxiv.org/abs/2007.01760), [[Code]](https://github.com/liznerski/fcdd)
  - Authors: Philipp Liznerski, Lukas Ruff, Robert A. Vandermeulen, Billy Joe Franks, Marius Kloft, Klaus-Robert Müller
  - turns its own feature mapping into an explanation heatmap, enabling both strong detection performance and interpretable localization of unusual regions. 
  - Datasets: Fashion-MNIST, CIFAR-10, ImageNet, MVTec AD





- ⭐⭐⭐ *(2020-ICLR)* **Deep Semi-Supervised Anomaly Detection** [[Paper]](), [[ArXiv]](https://arxiv.org/abs/1906.02694), [[Code]](https://github.com/lukasruff/Deep-SAD-PyTorch)
  - Authors: Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Binder, Alexander and M{\"u}ller, Emmanuel and M{\"u}ller, Klaus-Robert and Kloft, Marius 
  - uses a small number of known normal and abnormal examples to guide the model in separating normal from non-normal patterns.
  - Datasets: MNIST, Fashion-MNIST, and CIFAR-10 




- ⭐ *(2019-ICLR)* **Deep Anomaly Detection with Outlier Exposure** [[Paper]](https://openreview.net/forum?id=HyxCxhRcY7), [[ArXiv]](https://arxiv.org/abs/1812.04606), [[Code]](https://github.com/hendrycks/outlier-exposure)
  - Authors: Dan Hendrycks, Mantas Mazeika, Thomas Dietterich
  - improves detection of unusual inputs by training models with large external datasets of out-of-distribution examples so they can better recognize unseen irregular cases. 
  - Datasets: SVHN, CIFAR-10, CIFAR-100, Tiny ImageNet, Places365, 20 Newsgroups, TREC, SST, 80 Million Tiny Images, ImageNet-22K, WikiText-2



- ⭐⭐ *(2019-KDD)* **Deep Anomaly Detection with Deviation Networks** [[Paper]](https://dl.acm.org/doi/10.1145/3292500.3330871), [[ArXiv]](https://arxiv.org/abs/1911.08623), [[Code]](https://github.com/mala-lab/deviation-network-image)
  - Authors: Guansong Pang, Chunhua Shen, Anton van den Hengel
  - an end-to-end model that directly learns anomaly scores using a few labeled outliers, ensuring their scores statistically stand out from normal data for more efficient and accurate detection. 
  - Datasets: donors, census, fraud, celeba, backdoor, URL, campaign, news20, thyroid 


### Noise Anomaly Detection Methods


- ⭐⭐ *(2025-NeurIPS)* **An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination** [[ArXiv]](https://arxiv.org/pdf/2510.21296) [[code]](https://github.com/sukanyapatra1997/EPHAD)
  - Authors: Sukanya Patra, Souhaib Ben Taieb
  - proposes a test-time adaptation framework that updates the outputs of AD models trained on contaminated datasets using evidence gathered at test time. 
  - Datasets: MVTecAD, MPDD, ViSA, RealIAD


- ⭐ *(2025-ICCV)* **Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning** [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Aqeel_Towards_Real_Unsupervised_Anomaly_Detection_Via_Confident_Meta-Learning_ICCV_2025_paper.pdf) 
  - Authors: Muhammad Aqeel, Shakiba Sharifi, Marco Cristani, Francesco Setti
  - proposes Soft Confident Learning to filtering noisy sample and utilizes meta-learning to improve the ability to quickly adapt to new data.
  - Datasets: MVTec-AD, VIADUCT, KSDD2


- ⭐⭐ *(2025-CVPR)* **TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Jung_TailedCore_Few-Shot_Sampling_for_Unsupervised_Long-Tail_Noisy_Anomaly_Detection_CVPR_2025_paper.pdf), [[Code]](https://github.com/jungyg/TailedCore)
  - Authors: Yoon Gyo Jung, Jaewoo Park, Jaeho Yoon, Kuan-Chuan Peng, Wonchul Kim, Andrew Beng Jin Teoh, Octavia Camps
  - an improved model based on the softpatch to address the issues of tail class and noise obfuscation. 
  - Datasets: MVTecAD, VisA


- ⭐⭐ *(2025-PR)* **SoftPatch+: Fully Unsupervised Anomaly Classification and Segmentation** [[Paper]](https://www.sciencedirect.com/science/article/pii/S003132032401046X?ref=pdf_download&fr=RR-3&rr=9ada506ebe6397fb), [[Code]](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch)
  - Authors: Chengjie Wanga, Xi Jiangc, Bin-Bin Gaob, Zhenye Ganb, Yong Liub, Feng Zhengc, Lizhuang Maa
  - a memory-based unsupervised AD methods which efficiently denoise the data at the patch level.
  - Datasets: MVTecAD, BTAD


- ⭐⭐⭐ *(2022-NeurIPS)* **SoftPatch: Unsupervised Anomaly Detection with Noisy Data** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/637a456d89289769ac1ab29617ef7213-Paper-Conference.pdf), [[Code]](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch)
  - Authors: Xi Jiang, Jianlin Liu, Jinbao Wang, Qian Nie, Kai Wu, Yong Liu, Chengjie Wang, Feng Zheng
  - propose an patch-level denoise strategy to address the noisy data during training 
  - Datasets: MVTecAD, BTAD


## Zero/Few-shot Anomaly Detection

- ⭐⭐ *(2025-ICCV)* **ReMP-AD: Retrieval-enhanced Multi-modal Prompt Fusion for Few-Shot Industrial Visual Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.pdf), [[Code]](https://github.com/cshcma/ReMP-AD.git)
  - Authors: Hongchi Ma, Guanglei Yang, Debin Zhao, Yanli Ji, Wangmeng Zuo
  - proposes Retrieval-enhanced Multi-modal Prompt Fusion Anomaly Detection to improve matching between query images and reference images in few-shot anomaly detection.
  - Datasets: MVTec-AD, VisA, PCB-Bank


- ⭐ *(2025-ICCV)* **FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection and Segmentation** [[Paper]]([https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.pdf)
  - Authors: Tao Gong, Qi Chu, Bin Liu, Wei Zhou, Nenghai Yu
  - combines frequency information with the CLIP model.
  - Datasets: MVTec AD, VisA, MPDD, BTAD, DAGM, DTD-Synthetic. CVC-ClinicDB, Kvasir, BrainMRI, Br35H


- ⭐⭐⭐ *(2023-CVPR)* **WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf), [[Unoff-Code]](https://github.com/mala-lab/WinCLIP)
  - Authors: Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer
  - proposes text prompt-guided anoamly detection and first leverage manual text prompt set.
  - Datasets: MVTec-AD, VisA, PCB-Bank


# [Medical Anomaly Detection](#medical-anomaly-detection)




- *(2019-Medical Image Analysis )* **** [[Paper]](), [[ArXiv]](), [[Code]]()
  - Authors: 
  - Brain 
  - Datasets: 





## Reconstruction-based Methods



- ⭐ *(2025-TMI)* **Facing Differences of Similarity: Intra- and Inter-Correlation Unsupervised Learning for Chest X-Ray Anomaly Detection** [[Paper]](CheXpert)
  - Authors: Shicheng Xu; Wei Li; Zuoyong Li; Tiesong Zhao; Bob Zhang
  - An intra- and inter-correlation learning framework with anatomical feature-pyramid fusion to enhance chest X-ray anomaly detection. 
  - Datasets: ZhangLab, RSNA, CheXpert



- ⭐⭐ *(2025-TMI)* **Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction** [[Paper]](https://ieeexplore.ieee.org/document/10979458), [[Code]](https://github.com/TumCCC/E2AD)
  - Authors: Peng Tang; Xiaoxiao Yan; Xiaobin Hu; Kai Wu; Tobias Lasser; Kuangyu Shi
  - Extend to EDC, reducing encoder domain gaps using contrastive learning and dual decoders.
  - Datasets: OCT2017, APTOS, ISIC2018, Br35H



- ⭐⭐⭐ *(2024-TMI)* **Encoder-Decoder Contrast for Unsupervised Anomaly Detection in Medical Images** [[Paper]](https://ieeexplore.ieee.org/document/10296925), [[Code]](https://github.com/guojiajeremy/EDC)
  - Authors: Jia Guo, Shuai Lu, Lize Jia, Weihang Zhang, Huiqi Li
  - A contrastive encoder–decoder reconstruction method that jointly trains the whole network to avoid pattern collapse.
  - Datasets: OCT2017, APTOS, ISIC2018, Br35H
  - Similar to *(2023-NeurIPS) ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction* [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html), [[ArXiv]](https://arxiv.org/abs/2306.02602), [[Code]](https://github.com/guojiajeremy/ReContrast)





- ⭐⭐ *(2023-Medical Image Analysis)* **The role of noise in denoising models for anomaly detection in medical images** [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841523002232), [[ArXiv]](https://arxiv.org/abs/2301.08330), [[Code]](https://github.com/AntanasKascenas/DenoisingAE)
  - Authors: Antanas Kascenas, Pedro Sanchez, Patrick Schrempf, Chaoyang Wang, William Clackett, Shadia S. Mikhael, Jeremy P. Voisey, Keith Goatman, Alexander Weir, Nicolas Pugeault, Sotirios A. Tsaftaris, Alison Q. O'Neil
  - Tuning the spatial scale and magnitude of training noise significantly boosts denoising autoencoders. 
  - Datasets: BraTS 2021, iCAIRD GG&C NHS dataset: Head CT, qure.ai CQ500: Head CT 



- ⭐⭐⭐ *(2023-CVPR&PAMI)* **SQUID Deep Feature In-Painting for Unsupervised Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiang_SQUID_Deep_Feature_In-Painting_for_Unsupervised_Anomaly_Detection_CVPR_2023_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2111.13495), [[Code]](https://github.com/tiangexiang/SQUID)
  - Authors: Tiange Xiang, Yixiao Zhang, Yongyi Lu, Alan L. Yuille, Chaoyi Zhang, Weidong Cai, Zongwei Zhou
  - A space-aware memory–queue inpainting method that models recurring anatomical patterns in radiography images.
  - Datasets: DigitAnatomy, ZhangLab Chest X-ray, Stanford CheXpert
  - Further extension to PAMI: *Exploiting Structural Consistency of Chest Anatomy for Unsupervised Anomaly Detection in Radiography Images* [[Paper]](https://ieeexplore.ieee.org/document/10480307), [[ArXiv]](https://arxiv.org/abs/2403.08689), [[Code]](https://github.com/MrGiovanni/SimSID)



- ⭐⭐ *(2022-Medical Image Analysis)* **Constrained unsupervised anomaly segmentation** [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841522001736#bib0045), [[ArXiv]](https://arxiv.org/abs/2203.01671), [[Code]](https://github.com/jusiro/constrained_anomaly_segmentation)
  - Authors:  Julio Silva-Rodríguez, Valery Naranjo, Jose Dolz
  - Avoids using abnormal images to set thresholds by employing inequality-constrained attention optimization with log-barrier methods and entropy regularization.
  - Datasets: BraTS 2019, Physionet-ICH dataset





- ⭐ *(2021-Medical Image Analysis)* **Normative ascent with local gaussians for unsupervised lesion detection** [[Paper]](https://www.sciencedirect.com/science/article/pii/S136184152100253X)
  - Authors: Xiaoran Chen, Nick Pawlowski, Ben Glocker, Ender Konukoglu
  - Brain MRI lesion detection using autoencoders trained on healthy images.
  - Datasets: CamCAN, BRATS17, ATLAS



- ⭐⭐ *(2020-Medical Image Analysis)* **Unsupervised lesion detection via image restoration with a normative prior** [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841520300773), [[ArXiv]](https://arxiv.org/abs/2005.00031), [[Code]](https://github.com/yousuhang/Unsupervised-Lesion-Detection-via-Image-Restoration-with-a-Normative-Prior)
  - Authors: Xiaoran Chen, Suhang You, Kerem Can Tezcan, Ender Konukoglu
  - Brain MRI lesion detection using autoencoders trained on healthy images.
  - Datasets: CamCAN, BRATS17, ATLAS

---


- ⭐ *(2025-MICCAI)* **StyleGAN-Based Brain MRI Anomaly Detection via Latent Code Retrieval and Partial Swap** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-032-04937-7_54)
  - Authors: Jie Wei, Xiaofei Hu, Shaoting Zhang, Guotai Wang
  - A StyleGAN-based reconstruction method that retrieves similar healthy latent codes and partially swaps anomalous regions.
  - Datasets: OpenBHB dataset, BraTS2020, ATLAS




- ⭐⭐ *(2024-MICCAI)* **Rethinking Autoencoders for Medical Anomaly Detection from A Theoretical Perspective** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_51), [[ArXiv]](https://arxiv.org/abs/2403.09303), [[Code]](https://github.com/caiyu6666/AE4AD)
  - Authors: Yu Cai, Hao Chen, Kwang-Ting Cheng
  - Minimizing latent entropy is key to improving reconstruction-based detection.
  - Datasets: RSNA, VinDr-CXR, Brain Tumor, BraTS2021


- ⭐⭐ *(2024-MICCAI)* **Diffusion Models with Implicit Guidance for Medical Anomaly Detection** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_20), [[ArXiv]](https://arxiv.org/abs/2403.08464), [[Code]](https://github.com/ci-ber/THOR_DDPM)
  - Authors: Cosmin Bercea, Benedikt Wiestler, Daniel Rueckert, Julia A. Schnabel
  - A diffusion-based method that uses temporal anomaly guidance to preserve healthy tissue.
  - Datasets: IXI, ATLAS, Wrist X-ray


- ⭐⭐ *(2024-MICCAI)* **MetaAD: Metabolism-Aware Anomaly Detection for Parkinson’s Disease in 3D F-FDG PET** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_28), [[Code]](https://github.com/MedAIerHHL/MetaAD)
  - Authors: Haolin Huang, Zhenrong Shen, Jing Wang, Xinyu Wang, Jiaying Lu, Huamei Lin, Jingjie Ge, Chuantao Zuo, Qian Wang
  - A metabolism-aware cross-modality translation framework that detects PD anomalies by mapping FDG PET to synthetic healthy CFT PET and back for reconstruction-based anomaly cues. 
  - Datasets: in-house dataset







- ⭐⭐ *(2023-MICCAI)* **Reversing the Abnormal: Pseudo-Healthy Generative Networks for Anomaly Detection** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_29), [[ArXiv]](https://arxiv.org/abs/2303.08452), [[Code]](https://github.com/ci-ber/PHANES)
  - Authors: Cosmin Bercea, Benedikt Wiestler, Daniel Rueckert, Julia A. Schnabel
  - A generative-model–based method that reverses anomalies by masking and inpainting them with pseudo-healthy reconstructions. 
  - Datasets: FastMRI, ATLAS



- ⭐⭐ *(2023-ICLR)* **AE-FLOW: Autoencoders with Normalizing Flows for Medical Images Anomaly Detection** [[Paper]](https://openreview.net/forum?id=9OmCr1q54Z)
  - Authors: Yuzhong Zhao, Qiaoqiao Ding, Xiaoqun Zhang
  - A normalizing-flow autoencoder models combining Authencoder and flow likelihood.
  - Datasets: OCT, Chest X-ray, ISIC2018, BraTS2021, microscopic images MIIC 



- ⭐⭐ *(2022-MICCAI)* **Dual-Distribution Discrepancy for Anomaly Detection in Chest X-Rays** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_56), [[ArXiv]](https://arxiv.org/abs/2206.03935), [[Code]](https://github.com/caiyu6666/DDAD)
  - Authors: Yu Cai, Hao Chen, Xin Yang, Yu Zhou, Kwang-Ting Cheng
  - Leverages both normal and unlabeled chest X-rays by modeling separate normal and mixed distributions and detecting anomalies. 
  - Datasets: 
    - RSNA Pneumonia Detection Challenge dataset
    - VinBigData Chest X-ray Abnormalities Detection dataset
    - Chest X-ray Anomaly Detection （in-house）
  - Extension: *(2023-MedIA) Dual-distribution discrepancy with self-supervised refinement for anomaly detection in medical images* [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841523000555?via%3Dihub), [[ArXiv]](https://arxiv.org/abs/2211.07166), [[Code]](https://github.com/caiyu6666/DDAD-ASR)





- ⭐⭐ *(2019-MICCAI)* **Unsupervised Anomaly Localization using Variational Auto-Encoders** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-32251-9_32), [[ArXiv]](https://arxiv.org/abs/1907.02796), [[Code]](https://github.com/MIC-DKFZ/vae-anomaly-experiments)
  - Authors: David Zimmerer, Fabian Isensee, Jens Petersen, Simon Kohl, Klaus Maier-Hein
  - VAE-based, adding a KL-divergence–derived term, enabling assumption-free, architecture-independent detection.
  - Datasets: FashionMNIST, HCP, BraTS2017





## Feature-based/Memory Methods

- ⭐ *(2020-Medical Image Analysis)* **Regularized siamese neural network for unsupervised outlier detection on brain multiparametric magnetic resonance imaging: Application to epilepsy lesion screening** [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841519301562#bib0004)
  - Authors: Zaruhi Alaverdyan, Julien Jung, Romain Bouet, Carole Lartizien
  - One-class methods. A siamese network of convolutional autoencoders to learn location-consistent healthy brain representations, followed by voxel-wise one-class SVMs for detecting subtle MRI lesions.
  - Datasets: in-house MRI dataset 


---

- ⭐⭐ *(2025-MICCAI)* **Anomaly Detection by Clustering DINO Embeddings Using a Dirichlet Process Mixture** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-032-04947-6_5), [[ArXiv]](https://arxiv.org/abs/2509.19997), [[Code]](https://github.com/NicoSchulthess/anomalydino-dpmm)
  - Authors: Nico Schulthess, Ender Konukoglu
  - Modeling DINOv2 embeddings with a Dirichlet Process Mixture Model.
  - Datasets: BMAD, BTCV+LiTS, RESC



- ⭐⭐⭐  *(2024-CVPR)* **Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Adapting_Visual-Language_Models_for_Generalizable_Anomaly_Detection_in_Medical_Images_CVPR_2024_paper.pdf), [[ArXiv]](https://arxiv.org/abs/2403.12570), [[Code]](https://github.com/MediaBrain-SJTU/MVFA-AD)
  - Authors: Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, Yanfeng Wang
  - CLIP-based, Adding multi-level residual adapters and visual-language alignment losses.
  - Datasets:  BMAD



- ⭐⭐  *(2019-CVPR)* **Cascaded Generative and Discriminative Learning for Microcalcification Detection in Breast Mammograms** [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Cascaded_Generative_and_Discriminative_Learning_for_Microcalcification_Detection_in_Breast_CVPR_2019_paper.pdf), [[Code]](https://github.com/502463708/Microcalcification_Detection)
  - Authors: Fandong Zhang Ling Luo, Xinwei Sun, Zhen Zhou, Xiuli Li, Yizhou Yu, Yizhou Wang
  - Combines a generative network, using reconstruction-residual separation to detect microcalcifications.
  - Datasets: INBreast and in-house mammography dataset









## Synthesis-based Methods


- ⭐⭐ *(2024-TMI)* **Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images** [[Paper]](https://ieeexplore.ieee.org/document/10680156), [[ArXiv]](https://arxiv.org/abs/2308.02062), [[Code]](https://github.com/alessandro-f/Dif-fuse)
  - Authors: Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey
  - Weakly supervised AD, using saliency-guided, region-targeted diffusion editing, combining DDPM for lesion modification and DDIM for healthy-region preservation, to generate healthy counterfactuals and derive pixel-wise anomaly maps.
  - Datasets: IST-3, BraTS 2021, WMH


  ---


- ⭐ *(2025-MICCAI)* **Is Hyperbolic Space All You Need for Medical Anomaly Detection?** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-032-04947-6_30), [[ArXiv]](https://arxiv.org/abs/2505.21228), [[No code, 251128]](https://hyperbolic-anomalies.github.io/)
  - Authors: Alvaro Gonzalez-Jimenez, Simone Lionetti, Ludovic Amruthalingam, Philippe Gottfrois, Fabian Gröger, Marc Pouly, Alexander A. Navarini
  - Projecting features into hyperbolic space, enabling better hierarchy modeling.
  - Datasets: BMAD



- ⭐⭐⭐ *(2024-MICCAI)* **MediCLIP: Adapting CLIP for Few-Shot Medical Image Anomaly Detection** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_43), [[ArXiv]](https://arxiv.org/abs/2405.11315), [[Code]](https://github.com/cnulab/MediCLIP)
  - Authors: Ximiao Zhang, Min Xu, Dehui Qiu, Ruixin Yan, Ning Lang, Xiuzhuang Zhou
  - MediCLIP adapts CLIP for few-shot medical AD by using self-supervised disease-pattern synthesis to transfer CLIP’s generalization ability to medical imaging.
  - Datasets:  CheXpert, BrainMRI, BUSI


- ⭐⭐⭐ *(2023-MICCAI)* **Many Tasks Make Light Work Learning to Localise Medical Anomalies from Multiple Synthetic Tasks** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_16), [[ArXiv]](https://github.com/matt-baugh/many-tasks-make-light-work), [[Code]](https://github.com/matt-baugh/many-tasks-make-light-work)
  - Authors: Matthew Baugh, Jeremy Tan, Johanna P. Müller, Mischa Dombrowski, James Batten, Bernhard Kainz
  - training and validating on multiple visually distinct synthetic-anomaly tasks
  - Datasets: HCP, BraTS 2017,  ISLES 2015, VinDr-CXR




- ⭐ *(2022-MICCAI)* **Fast Unsupervised Brain Anomaly Detection and Segmentation with Diffusion Models** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_67), [[ArXiv]](https://arxiv.org/abs/2206.03461)
  - Authors: Walter H. L. Pinaya, Mark S. Graham, Robert Gray, Pedro F Da Costa, Petru-Daniel Tudosiu, and others
  - Latent-space diffusion models for brain image anomaly detection.
  - Datasets: HeadCT, UKB,  WMH, MSLUB, BRATS






- ⭐⭐ *(2021-MICCAI)* **Implicit Field Learning for Unsupervised Anomaly Detection in Medical Images** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_18), [[ArXiv]](https://arxiv.org/abs/2106.05214), [[Code]](https://github.com/snavalm/ifl_unsup_anom_det)
  - Authors: Sergio Naval Marimont & Giacomo Tarroni 
  - An implicit-fields auto-decoder to model healthy anatomy.
  - Datasets: HCP dataset, BRATS 2018




- ⭐ *(2020-MICCAI)* **SteGANomaly Inhibiting CycleGAN Steganography for Unsupervised Anomaly Detection in Brain MRI** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_69)
  - Authors: Christoph Baur, Robert Graf, Benedikt Wiestler, Shadi Albarqouni & Nassir Navab 
  - A CycleGAN-based style-transfer framework that suppresses anomalies by translating images to a low-entropy domain and detecting pathologies via input–reconstruction residuals.
  - Datasets: in-house, 
    - WHM datasets (2019 TMI)



- ⭐⭐ *(2017-IPMI)* **Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_12), [[ArXiv]](https://arxiv.org/abs/1703.05921), [[Code]](https://github.com/tSchlegl/f-AnoGAN)
  - Authors: Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, Georg Langs
  - A GAN-based unsupervised method that learns the manifold of normal anatomy and detects anomalies by mapping images to latent space and scoring their deviations.
  - Datasets: SD-OCT volumes of the retina
  - Extension: *(2019-Medical Image Analysis)* **f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks** [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841518302640)




## Benchmark 

- ⭐⭐ *(2025-MICCAI)* **BenchReAD: A Systematic Benchmark for Retinal Anomaly Detection** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-032-04937-7_4), [[ArXiv]](https://arxiv.org/abs/2507.10492), [[Dataset]](https://github.com/DopamineLcy/BenchReAD)
  - Fundus and OCT


- ⭐⭐⭐ *(2024-CVPR Workshops)* **BMAD: Benchmarks for Medical Anomaly Detection** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Bao_BMAD_Benchmarks_for_Medical_Anomaly_Detection_CVPRW_2024_paper.html), [[ArXiv]](https://arxiv.org/abs/2306.11876), [[Dataset & Code]](https://github.com/DorisBao/BMAD)
  - Authors: Jinan Bao, Hanshi Sun, Hanqiu Deng, Yinsheng He, Zhaoxiang Zhang, Xingyu Li
  - Brain MRI, Chest X-ray, Liver CT, Retinal OCT, Pathology 



## Surveys

- ⭐ *(2025-Medical Image Analysis)* **MedIAnomaly: A comparative study of anomaly detection in medical images** [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841525000489), [[ArXiv]](https://arxiv.org/abs/2404.04518), [[Code]](https://github.com/caiyu6666/MedIAnomaly?tab=readme-ov-file#reconstruction-based-methods)
  - Authors: Yu Cai, Weiwen Zhang, Hao Chen, Kwang-Ting Cheng
  - A comprehensive survey of Reconstruction-based methods.
  - Datasets: RSNA, VinDr-CXR, Brain Tumor, BraTS2021, LAG, Camelyon16



- ⭐ *(2024-TMI)* **Unsupervised Pathology Detection: A Deep Dive Into the State of the Art** [[Paper]](https://ieeexplore.ieee.org/document/10197302), [[ArXiv]](https://arxiv.org/abs/2303.00609), [[Code]](https://github.com/iolag/UPD_study)
  - Authors: Ioannis Lagogiannis, Felix Meissen, Georgios Kaissis, Daniel Rueckert
  - Benchmarks diverse unsupervised anomaly detection methods across multiple medical datasets
  - Datasets: CamCAN, ATLAS, BraTS2020, CheXpert, DDR 


- ⭐ *(2021-Medical Image Analysis)* **Autoencoders for unsupervised anomaly segmentation in brain MR images A comparative study**  [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841520303169), [[ArXiv]](https://arxiv.org/abs/2004.03271), [[Code]](https://github.com/StefanDenn3r/unsupervised_anomaly_detection_brain_mri)
  - Authors:  Christoph Baur, Stefan Denner, Benedikt Wiestler, Shadi Albarqouni, Nassir Navab
  - This study benchmarks unsupervised brain MRI anomaly detection methods under a unified experimental setup, comparing architectures, data requirements, and robustness while identifying open challenges and future directions. 
  - Datasets: in-house datasets,MSLUB, MSSEG2015




---



## Datasets

- BMAD: https://github.com/DorisBao/BMAD 

- Brain Tumor/Stroke Segmentation
  - BraTS2021 Dataset: https://arxiv.org/abs/2107.02314,  https://www.med.upenn.edu/cbica/brats2021/


  - The ischemic stroke lesion segmentation (ISLES) dataset
    - ISLES 2015:  https://www.sciencedirect.com/science/article/pii/S1361841516301268,   https://www.isles-challenge.org/

  - https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

- Pathology  
  - Camelyon16: https://camelyon17.grand-challenge.org/Data/


- Chest X-ray  
  - RSNA: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

- VinDr-CXR:  https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

- Liver CT
  - BTCV Dataset : https://www.synapse.org/#!Synapse:syn3193805/wiki/217753
  - LiTS Dataset: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation
-   



---

# Updates
- 🍋 2025-12-29: Added Industrial Few/Zero-shot AD papers, unsupervised AD papers(2023-2025, CVPR, ICCV, NeurIPS, etc.).
- 🍋 2025-12-14: Added Industrial noisy AD papers, supervised AD papers (2022-2025, CVPR, PR, NeurIPS, etc.).
- 🦘 2025-12-11: Added Industrial unsupervised AD papers (2018-2025, ICLR, CVPR, ICCV, AAAI, etc.).
- 🦘 2025-12-10: Added Industrial supervised AD papers (2017-2025, ICLR, CVPR, AAAI).
- 🦘 2025-11-29: Added Medical Anomaly Detection papers (2017-2025, TMI,MedIA, MICCAI, CVPR) and datasets.
- 🦘 2025-11-28: Added Medical Anomaly Detection papers (2020-2025, TMI,MedIA, MICCAI, CVPR) and datasets.



