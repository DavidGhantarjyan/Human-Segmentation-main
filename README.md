# Human Semantic Segmentation for Resource-Constrained Environments

This project presents a robust and efficient solution for human semantic segmentation, specifically optimized for real-time applications on devices with limited computational resources, such as mobile phones. It leverages lightweight neural network architectures, advanced data handling techniques, and specialized loss functions to achieve high accuracy segmentation masks. The core innovation lies in combining efficient model architectures with sophisticated training methodologies and post-processing steps to deliver superior performance.

## Demo
Watch a real-time demonstration of our human semantic segmentation model, showcasing its performance across diverse poses, backgrounds, and lighting conditions. The video highlights both the raw segmentation output and the refined masks achieved through the Guided Filter post-processing, alongside real-time inference speed.

Example Markdown for a YouTube video: [Watch the demo video](https://youtu.be/R8EFysgOZIw?si=tAVhUdEhBwHUd2Pn)

## Key Features
- **Lightweight Architectures**: Utilizes UNet with MobileNetV1, V2, or V3 encoders to significantly reduce model parameters and computational complexity (GFLOPs) compared to traditional, heavier segmentation models.
- **Advanced Loss Functions**: Employs a unique and carefully tuned combination of Binary Cross-Entropy (BCE), Tversky Loss or Generalized Dice Loss (GDL), and Boundary Loss for robust training, effectively addressing challenges like class imbalance and refining object boundaries.
- **Effective Post-processing**: Integrates the Guided Filter as a crucial post-processing step to smooth segmentation masks, eliminate artifacts, and ensure precise boundary alignment with original image contours, vital for practical applications.
- **Comprehensive Data Strategy**: Developed a diverse training dataset by combining various real-world (COCO, Mapillary Vistas) and synthetic datasets (custom, TikTok Dataset, VideoMatte240K) with extensive data augmentation, enhancing model generalization and robustness to real-world variations.
- **Real-time Performance**: Achieves high inference speeds on GPU, making it suitable for real-time human segmentation tasks on resource-constrained hardware.

## Methodology
Our approach focuses on a comprehensive pipeline designed to deliver high-accuracy human semantic segmentation while maintaining computational efficiency for real-time applications.

### 1. Data Preparation & Preprocessing
A diverse and representative training dataset was crucial for robust model performance.

- **Dataset Integration**: The training dataset was formed by combining real datasets (COCO and Mapillary Vistas) and synthetic datasets (custom, TikTok Dataset, VideoMatte240K). This combination ensures a wide spectrum of poses, backgrounds, and lighting conditions, significantly enhancing the model's generalization capabilities. The final training dataset comprises 76,000 images.
- **Image Resizing and Mask Binarization**: All images are resized to a target resolution of 320x180 (16:9 aspect ratio). Different interpolation methods (INTER_LINEAR for upscaling, INTER_AREA for downscaling) were strategically used to preserve image details and prevent aliasing. Segmentation masks are binarized and resized using INTER_NEAREST interpolation to prevent distortion of binary labels.
- **Synthetic Data Refinement**: For synthetic images, a Gaussian filter was applied to the alpha channel during object placement. This technique creates smoother transitions between foreground objects and backgrounds, improving realism and mitigating visual artifacts that can arise from synthetic generation.

### 2. Data Augmentation
Data augmentation plays a vital role in enhancing the model's robustness to real-world variations in illumination, object orientation, occlusion, and image quality.

- **Techniques**: The Albumentations library was utilized to apply a range of geometric and photometric transformations. These include:
  - **Geometric**: HorizontalFlip, Rotate (with varying angles), RandomCropNearRandomObjBBox (to simulate partial visibility), and CoarseDropout (to model occlusions). Geometric transformations are applied simultaneously to both images and their corresponding masks to maintain consistency.
  - **Photometric**: RandomShadow, Gamma Correction, CLAHE (Contrast Limited Adaptive Histogram Equalization), Sigmoid Contrast, HueSaturationValue, GaussNoise, and GaussianBlur (for background blurring).
- **Intensity Adaptation**: Mild augmentations were applied to real datasets to preserve their naturalness, while more intense augmentations and additional effects were used for synthetic data to maximize diversity and build robustness against potential generation artifacts.

### 3. Architectural Solutions
The core of our segmentation solution is the UNet_MobileNet model, an optimized version of the classical UNet architecture.

- **Base Architecture**: The UNet architecture was chosen for its proven effectiveness in segmentation tasks due to its skip connections that help retain fine-grained spatial information.
- **Efficient Encoder**: The standard convolutional blocks in the UNet encoder were replaced with lightweight blocks from the MobileNet family (V1, V2, or V3). This substitution drastically reduces the model's parameter count and computational complexity (FLOPs) compared to a classical UNet (~60 million parameters).
  - The encoder processes 320x180 RGB images and extracts four feature levels with progressively decreasing spatial resolution.
  - Downsampling is performed using conv_dw_block where the first block uses stride=2 and subsequent blocks use stride=1 for additional feature processing.
- **Decoder with Skip Connections**: The decoder utilizes transposed convolutions for upsampling to restore the original resolution. It incorporates skip connections from the encoder to combine high-level semantic features with low-level spatial details, crucial for precise boundary prediction.
- **Channel Scaling (alpha parameter)**: The alpha parameter (e.g., 1.0, 0.45, 0.3) is used to scale the number of channels within the MobileNet blocks, allowing flexible control over the model's size and complexity to meet specific resource constraints.
- **Output Layer**: A 1x1 convolutional layer transforms the decoder's output into a single-channel binary mask (320x180), where pixel values represent probabilities of belonging to the foreground (human) class.

### 4. Training & Validation Process
The training and validation process was meticulously designed to optimize segmentation accuracy under computational constraints.

- **Combined Loss Function**: Training utilized a combined loss function to address various segmentation challenges:
  - **Initial Phase**: Binary Cross-Entropy (BCE) and Tversky Loss (with α=0.3, β=0.7) or Generalized Dice Loss (GDL) were combined in a 60:40 ratio. BCE provides stable gradients, while Tversky/GDL are robust to class imbalance and improve mask overlap accuracy.
  - **Boundary Refinement Phase**: Once the validation mIoU plateaued, Boundary Loss was gradually introduced into the combined loss function (its weight increased by 1% per epoch). Boundary Loss is critical for refining the geometric structure and spatial integrity of object boundaries, particularly for small objects against large backgrounds. The signed distance map for Boundary Loss was computed on the CPU to minimize GPU load.
- **Optimizer**: The AdamW optimizer was chosen for its stability and effectiveness in deep learning models. The learning rate was maintained at 10e-3 initially and reduced to 3e-4 upon the introduction of Boundary Loss to ensure smooth integration.
- **Training Configuration**: All models (UNet + MobileNetV1/V2/V3) were trained under consistent conditions for 150 epochs, using a batch size of 12 with 8 gradient accumulation steps, resulting in an effective batch size of 96.
- **Validation Metrics**: Validation was performed on a 9,000-image subset (3,000 from COCO, 6,000 from Mapillary Vistas) after each epoch. Key metrics evaluated included:
  - Mean Intersection over Union (mIoU)
  - Dice Coefficient (F1-score)
  - Accuracy
  - Precision
  - Recall
  - Area Under the ROC Curve (AUROC)

## Results
Extensive experimental studies were conducted comparing UNet models with MobileNetV1 (α=1.0) and MobileNetV3 (α=0.3) blocks. Post-processing with the Guided Filter was applied to all models to enhance boundary clarity.

| Model                | Parameters | GFLOPs | Val IoU | Val F1 | FPS (GPU) |
|----------------------|------------|--------|---------|--------|-----------|
| UNet + MobileNetV1   | ~12M       | 10.13  | 0.766   | 0.867  | 50        |
| UNet + MobileNetV3   | ~3M        | 2.88   | 0.775   | 0.871  | 45        |

### UNet + MobileNetV1 Model (α=1.0):
With approximately 12 million parameters and 10.13 GFLOPs, this model achieved strong performance metrics, including an IoU of 0.766, an F1-score of 0.867, and an AUROC of 0.990 when trained with the BCE + Tversky + Boundary loss function. It sustained an impressive inference speed of 50 FPS on GPU.

### UNet + MobileNetV3 Model (α=0.3):
Featuring significantly fewer parameters (~3 million) and lower computational complexity (2.88 GFLOPs), this model, also trained with BCE + Tversky + Boundary loss, demonstrated superior segmentation quality. It achieved higher IoU (0.775) and F1-score (0.871) values compared to MobileNetV1, along with stable validation metrics. Its inference speed on GPU was 45 FPS.

### Comparative Analysis:
- **Segmentation Quality**: The UNet + MobileNetV3 model generally exhibited improved segmentation quality (higher IoU, F1-score, Recall) compared to MobileNetV1. This indicates more accurate segmentation and a better ability to detect target objects, even with complex poses. This superior quality is attributed to MobileNetV3's advanced architectural features, such as squeeze-and-excitation blocks and h-swish activation. Both models achieved a similar high AUROC of 0.990.
- **Inference Speed**: Despite having roughly four times fewer parameters and significantly lower GFLOPs, the MobileNetV3 model's GPU inference speed (45 FPS) was slightly lower than MobileNetV1 (50 FPS). This observation highlights that GFLOPs and parameter count are not the sole determinants of actual GPU inference speed. MobileNetV1 demonstrates better optimization for parallel computations on GPU, leveraging cuDNN library optimizations more efficiently. In contrast, MobileNetV3, due to its more complex architectural elements (SE modules, h-swish activation), proves to be more efficient on CPU for sequential operations, offering nearly double the inference speed compared to MobileNetV1 on such platforms.

## Conclusion
The research conclusively demonstrates that integrating lightweight MobileNetV1 and MobileNetV3 blocks into the U-Net architecture provides a highly effective solution for accurate human semantic segmentation under limited computational resources. The MobileNetV3-based UNet excels in overall segmentation quality (Recall, F1, IoU) with a remarkably lower parameter count and computational complexity. Conversely, the MobileNetV1-based UNet shows better GPU inference speed, likely due to more efficient utilization of cuDNN library optimization and parallel processing capabilities inherent in its design. Notably, MobileNetV3 exhibits superior performance on CPU platforms, making it a versatile choice depending on the deployment environment.

## Future Work
- **Differentiable Guided Filter**: A promising direction is to integrate the Guided Filter directly into the neural network architecture as a differentiable layer. This would allow for end-to-end training, enabling the filter parameters to be learned jointly with the network weights, potentially leading to further improvements in segmentation quality.
- **Spatio-Temporal Dependencies**: Incorporating recurrent neural network blocks, such as ConvLSTM or ConvGRU, into the segmentation architecture could enhance performance by enabling the model to account for spatio-temporal dependencies and local context, particularly beneficial for video-based human segmentation.

## License
This project is released under the **MIT License** — you are free to use, modify, and distribute the code, provided that this copyright notice is included.

However, please note that some **datasets and external resources** used during development are governed by their own licenses:

- **COCO Dataset**  
  Licensed under [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
  ([COCO on PapersWithCode](https://paperswithcode.com/dataset/coco))

- **Mapillary Vistas Dataset**  
  Licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
  ([Mapillary Vistas](https://www.mapillary.com/dataset/vistas))

- **TikTok Dataset**  
  Licensed under [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)  
  (as listed on [Papers with Code](https://paperswithcode.com/dataset/tiktok-dataset)) — for **non-commercial use only**.

- **VideoMatte240K Dataset**  
  Released under the [MIT License](https://github.com/PeterL1n/VideoMatting#license) — free use with attribution.

- **Boundary Loss (LIVIAETS)**  
  Released under the [MIT License](https://github.com/LIVIAETS/boundary-loss/blob/master/LICENSE) — free to use and redistribute with attribution.

- **Deep Guided Filter (wuhuikai)**  
  Also released under the [MIT License](https://github.com/wuhuikai/DeepGuidedFilter/blob/master/LICENSE).  
