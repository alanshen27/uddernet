# UdderNet: A Dual-Modality Neural Approach to Bovine Mastitis Detection

## Abstract

Bovine mastitis, an inflammatory condition of the mammary gland, remains one of the most economically significant diseases in dairy production, with losses attributable to reduced milk yield, discarded milk, veterinary intervention, and premature culling. Early and reliable detection is therefore of considerable practical importance. In this work we present *UdderNet*, a dual-modality framework comprising (i) a convolutional neural network (CNN) for the visual classification of udder imagery and (ii) a fully connected classification network operating on routinely collected milk-sensor measurements. On a dataset of 800 milk samples characterised by seven physiological and physicochemical features, the tabular classifier attains perfect discrimination on a held-out test set (accuracy, precision, recall, and F1 all equal to 1.00). The convolutional pathway is architecturally validated and prepared for training upon availability of labelled image data. We describe the data, model architectures, and training protocol, and discuss the limitations inherent to the present evaluation.

## 1. Introduction

Mastitis detection in commercial dairy herds has traditionally relied on the California Mastitis Test, somatic cell counting, and visual inspection by trained personnel. These approaches, while effective, are labour-intensive and poorly suited to continuous monitoring at scale. Recent advances in machine learning offer an opportunity to automate detection using two complementary information sources: in-line milk-sensor data, which capture the physicochemical signature of intramammary infection, and visual imagery of the udder, which may reveal clinical signs such as swelling and erythema.

The contributions of this work are threefold:

1. A compact convolutional architecture (`UdderCNN`) for binary classification of udder images into *positive* (mastitic) and *negative* (healthy) classes.
2. A feed-forward classification network (`MastitisMLP`) for tabular milk-sensor data.
3. An end-to-end implementation, including reproducible training pipelines and an inference service with a browser-based interface for interactive evaluation.

## 2. Materials and Methods

### 2.1 Tabular dataset

The tabular dataset comprises 800 milk-sample records, each annotated with a binary diagnosis (`class1`; 0 = healthy, 1 = mastitis). The class distribution is imbalanced, with 631 healthy (78.9%) and 169 mastitic (21.1%) samples. Each record contains seven predictor variables: day of lactation, milk temperature (°C), milk pH, electrical conductivity, somatic cell count, daily milk yield, and a binary clotting indicator.

Exploratory analysis reveals pronounced between-class separation. Mastitic samples exhibit, on average, elevated milk temperature (38.1 °C vs. 35.5 °C), higher pH (7.11 vs. 6.65), markedly reduced yield (9.4 L vs. 20.2 L), and clotting in 82% of cases against 0% in healthy samples. These differences are consistent with the established pathophysiology of intramammary infection.

### 2.2 Image dataset

Udder images are organised in a directory structure with one folder per class (`data/images/positive/`, `data/images/negative/`). Images are resized to 128 × 128 pixels and normalised using ImageNet channel statistics. During training, stochastic augmentation (horizontal flipping, rotation up to ±10°, and brightness/contrast jitter) is applied to mitigate overfitting.

### 2.3 Model architectures

**UdderCNN.** The convolutional network consists of four blocks, each comprising two 3 × 3 convolutions with batch normalisation and ReLU activation, followed by 2 × 2 max-pooling. Channel width doubles per block (32 → 64 → 128 → 256). Global average pooling is followed by a two-layer classifier head (256 → 128 → 2) with dropout (p = 0.3). The network contains approximately 1.2 M trainable parameters.

**MastitisMLP.** The tabular classifier is a three-layer perceptron (7 → 64 → 32 → 2) with batch normalisation, ReLU activations, and dropout (p = 0.2) after each hidden layer. Inputs are standardised to zero mean and unit variance using statistics estimated exclusively on the training partition.

### 2.4 Training protocol

Both networks are trained by minimising the cross-entropy loss with the AdamW optimiser (learning rate 10⁻³, weight decay 10⁻⁴). The tabular data are partitioned into training (64%), validation (16%), and test (20%) sets with class-stratified sampling and a fixed random seed (42) to ensure reproducibility. The MLP is trained for 100 epochs with a batch size of 64; the model achieving the highest validation accuracy is retained for final evaluation. The CNN training pipeline employs an 80/20 train–validation split, cosine-annealed learning-rate scheduling, and checkpointing on best validation accuracy.

### 2.5 Implementation

All models are implemented in PyTorch 2.x and trained under Python 3.13. Inference is exposed via a FastAPI service offering JSON endpoints for both modalities and a lightweight browser interface for interactive testing.

## 3. Results

### 3.1 Tabular classification

The MLP converged rapidly, exceeding 99% training accuracy within ten epochs. On the held-out test set (n = 160; 126 healthy, 34 mastitic), the best checkpoint achieved:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Healthy (0) | 1.00 | 1.00 | 1.00 | 126 |
| Mastitis (1) | 1.00 | 1.00 | 1.00 | 34 |
| **Accuracy** | | | **1.00** | 160 |

### 3.2 Image classification

The convolutional pathway has been verified architecturally (forward-pass dimensionality and parameter count) and is pending empirical evaluation upon acquisition of a labelled image corpus.

## 4. Discussion

The perfect test-set performance of the tabular classifier warrants cautious interpretation. The between-class differences in the dataset are large relative to within-class variance — most notably the near-deterministic association between clotting and the positive class — rendering the classification task close to linearly separable. Such separability is plausible for clinically overt mastitis but is unlikely to hold for subclinical cases, where sensor signatures overlap substantially with the healthy distribution. Consequently, the reported figures should be regarded as an upper bound on field performance.

Several limitations merit emphasis. First, the dataset is of modest size and its provenance (single herd, sensor calibration, sampling protocol) is unspecified, limiting external validity. Second, repeated measurements from the same animal, if present across partitions, could inflate performance estimates; cow-level splitting would provide a more conservative evaluation. Third, the image pathway remains untrained, and the proposed fusion of the two modalities is left to future work.

## 5. Conclusion

We have presented a dual-modality neural framework for bovine mastitis detection, comprising a convolutional network for udder imagery and a fully connected classifier for milk-sensor data, together with a reproducible training pipeline and an interactive inference service. The tabular classifier achieves ceiling performance on the available dataset; validation on larger, multi-herd cohorts with subclinical cases, and the training and evaluation of the image pathway, constitute the principal directions for future work.

## References

1. Halasa, T., Huijps, K., Østerås, O., & Hogeveen, H. (2007). Economic effects of bovine mastitis and mastitis management: A review. *Veterinary Quarterly*, 29(1), 18–31.
2. Hogeveen, H., Kamphuis, C., Steeneveld, W., & Mollenhorst, H. (2010). Sensors and clinical mastitis—The quest for the perfect alert. *Sensors*, 10(9), 7991–8009.
3. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.
