# Awesome Industrial & Medical Anomaly Detection

A curated list of papers, datasets, and code resources for anomaly detection in industrial inspection and medical imaging.

---

## Table of Contents

* [Industrial Anomaly Detection](#industrial-anomaly-detection)

  * [Surveys](#surveys)
  * [Papers](#papers)
  * [Code](#code)


# [Industrial Anomaly Detection](#industrial-anomaly-detection)







# [Medical Anomaly Detection](#medical-anomaly-detection)

- *(2024-Template)* **** [[Paper]](), [[ArXiv]](), [[Code]]()
  - Authors: 
  - Brain 
  - Datasets: 


## Reconstruction-based Methods



- ⭐ *(2025-TMI)* **Facing Differences of Similarity: Intra- and Inter-Correlation Unsupervised Learning for Chest X-Ray Anomaly Detection** [[Paper]](CheXpert), [[Code]]()
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






## Feature-based/Memory Methods


- ⭐⭐ *(2025-MICCAI)* **Anomaly Detection by Clustering DINO Embeddings Using a Dirichlet Process Mixture** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-032-04947-6_5), [[ArXiv]](https://arxiv.org/abs/2509.19997), [[Code]](https://github.com/NicoSchulthess/anomalydino-dpmm)
  - Authors: Nico Schulthess, Ender Konukoglu
  - Modeling DINOv2 embeddings with a Dirichlet Process Mixture Model.
  - Datasets: BMAD, BTCV+LiTS, RESC



## Synthesis-based Methods


- ⭐ *(2025-MICCAI)* **Is Hyperbolic Space All You Need for Medical Anomaly Detection?** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-032-04947-6_30), [[ArXiv]](https://arxiv.org/abs/2505.21228), [[No code, 251128]](https://hyperbolic-anomalies.github.io/)
  - Authors: Alvaro Gonzalez-Jimenez, Simone Lionetti, Ludovic Amruthalingam, Philippe Gottfrois, Fabian Gröger, Marc Pouly, Alexander A. Navarini
  - Projecting features into hyperbolic space, enabling better hierarchy modeling.
  - Datasets: BMAD



- ⭐⭐⭐ *(2024-MICCAI)* **MediCLIP: Adapting CLIP for Few-Shot Medical Image Anomaly Detection** [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_43), [[ArXiv]](https://arxiv.org/abs/2405.11315), [[Code]](https://github.com/cnulab/MediCLIP)
  - Authors: Ximiao Zhang, Min Xu, Dehui Qiu, Ruixin Yan, Ning Lang, Xiuzhuang Zhou
  - MediCLIP adapts CLIP for few-shot medical AD by using self-supervised disease-pattern synthesis to transfer CLIP’s generalization ability to medical imaging.
  - Datasets:  CheXpert, BrainMRI, BUSI


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



---



## Datasets

- [[BMAD]](https://github.com/DorisBao/BMAD) : https://github.com/DorisBao/BMAD 

- [[BraTS2021 Dataset]](http://braintumorsegmentation.org/): http://braintumorsegmentation.org/
  - https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

- Pathology  
  - Camelyon16: https://camelyon17.grand-challenge.org/Data/


- Chest X-ray  
  - [[RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)]: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

- [[VinDr-CXR]] (https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) :  https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

-  [[Liver CT]]
  - BTCV Dataset : https://www.synapse.org/#!Synapse:syn3193805/wiki/217753
  - LiTS Dataset: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation
-   



---

## License

Specify the license.

