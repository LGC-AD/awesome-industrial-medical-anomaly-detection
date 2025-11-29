# Awesome Industrial & Medical Anomaly Detection

A curated list of papers, datasets, and code resources for anomaly detection in industrial inspection and medical imaging.

---


## Table of Contents




# [Industrial Anomaly Detection](#industrial-anomaly-detection)







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


- 🦘 2025-11-29: Added Medical Anomaly Detection papers (2017-2025, TMI,MedIA, MICCAI, CVPR) and datasets.
- 🦘 2025-11-28: Added Medical Anomaly Detection papers (2020-2025, TMI,MedIA, MICCAI, CVPR) and datasets.



