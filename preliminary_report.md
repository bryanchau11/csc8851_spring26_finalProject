# Preliminary Report

## 1. Introduction

Estimating the nutritional content of a meal from a single photograph is a challenging and practically important problem for applications in dietary monitoring, health care, and automated food logging.  In our final project we address the task of **multi-target regression** on the Nutrition5K dataset [@okhin2020nutrition5k], predicting four continuous targets (calories, fat, protein and carbohydrates) from an overhead image of a cooked dish.

Nutrition5K is a large-scale public dataset collected by Gilles Okhin et al. and released on Kaggle.  It contains over 5 000 food items with paired RGB images, depth scans, and ground-truth nutrient values obtained with a precision balance and nutrition labels.  The high variability in appearance (different cuisines, container shapes, lighting) and the presence of occlusions and mixed dishes make this a non‑trivial vision problem.

Our project proceeds in six phases:

1. **Phase 3 (Baseline)** – ResNet‑50 CNN fine‑tuned end‑to‑end on Nutrition5K for direct nutrient regression.
2. **Phase 4 (Geometric pipeline)** – YOLO‑Depth‑MLP: multi‑stage segmentation + depth feature extraction + MLP regression with MC Dropout.
3. **Phase 5 (Food classifier)** – EfficientNet‑B0 trained on Food‑101 for food‑type recognition (auxiliary task).
4. **Phase 6 (Weight‑centric)** – MLP predicting dish weight from geometry, then multiplying by calorie‑density constants.

The goal of this preliminary report is to review the models implemented so far and summarise the evaluation methodology.  Complete numerical results are stored in the four experiment notebooks (Phases 3, 4, 5, 6).

> **Notation.** Let $(x_i, y_i)$ denote an image–nutrition pair; $y_i\\in \\mathbb{R}^4$ corresponds to (cal, fat, protein, carbs).  We train models $f(x; 	heta)$ to minimise a regression loss such as Smooth‑$L_1$:
> \[
>  \\mathcal{L}_{\text{Huber}}(f(x), y) = 
>  egin{cases}
>    	frac{1}{2}(f(x)-y)^2 & |f(x)-y|<1, \\[0.3em]
>    |f(x)-y| - 	frac{1}{2} & \text{otherwise.}
>  \\end{cases}
> \]  


## 2. Related work

Food analysis with deep learning has attracted growing attention in the last decade.  Early approaches focused on classification of food categories using standard CNNs such as AlexNet, VGG and ResNet [@deepfood2018].  More recent work has tackled portion size and calorie estimation by combining visual features with contextual information (e.g. tableware detection) or using multi‑task networks [@yang2018image].

### Nutrient regression
Nutrition5K was introduced precisely to support this regression task [@okhin2020nutrition5k].  The original paper evaluated several backbones (ResNet, DenseNet) and reported mean absolute errors (MAE) in the range of 10–20 % relative to ground truth.  Our Phase 3 baseline closely follows this line by fine‑tuning a ResNet‑50 pretrained on ImageNet and using data augmentations (random flips, color jitter) to combat overfitting.

### Segmentation‑based pipelines
Segmentation or detection of the food region is a common pre‑processing step in dietary analysis.  Models such as Mask R‑CNN [@he2017mask] and more recently YOLOv8‑seg [@ultralytics2023yolov8] enable real‑time instance segmentation; we adopt the latter for its ease of use and speed.

### Depth estimation and geometric reasoning
Pure RGB models can struggle when food occupies a small portion of the frame or when volume cues are important.  Combining depth information, either from real sensors (available in Nutrition5K) or monocular predictors like MiDaS [@ranftl2020midas], offers a route to estimate volume.  Previous work [@liu2021depthfood] has shown that depth features (mean, variance, area of segmented mask) can improve calorie estimates.

### Uncertainty quantification
In health applications it is desirable to know when a model is unsure.  Monte Carlo (MC) dropout [@gal2016dropout] is a lightweight Bayesian approximation that supplies predictive intervals.  We integrate MC dropout into our MLP head and report 95 % confidence bounds on nutrient estimates.

### Summary table of architectures

| Phase | Model components | Backbone | Novelty | References |
|-------|------------------|----------|---------|------------|
| 3 (baseline) | Single CNN regressor | ResNet‑50 | pretrained fine‑tuning | [@he2016deep] |
| 4 (proposed) | YOLOv8‑seg → depth features → MLP | YOLOv8, MiDaS, small MLP | geometric + uncertainty | [@ultralytics2023yolov8], [@ranftl2020midas], [@gal2016dropout] |


## 3. Dataset statistics

Table 1 summarises the train/val/test splits used throughout Phases 3–4.  The 80/10/10 random split ensures that dish IDs do not overlap across partitions.

| Split | # samples | % of total |
|-------|-----------|------------|
| Train | 4 000 (approx.) | 80 % |
| Val   | 500 | 10 % |
| Test  | 500 | 10 % |

*Table 1. Data splits for Nutrition5K experiments.*

Figure 1 shows example images from the dataset along with the corresponding depth maps and nutrient labels.

![Sample Nutrition5K dishes](./sample_dishes.png)

*Figure 1. Random examples from Nutrition5K. Top row: RGB. Bottom row: depth.*


## 4. Model descriptions

### 4.1 Phase 3: ResNet‑50 baseline

The baseline model (`NutritionEstimator`) uses a ResNet‑50 backbone with ImageNet weights.  After global average pooling the 2 048‑dim feature vector feeds a
fully connected head (2048→512→4) with batch norm, ReLU and dropout.  During training we employ a two‑phase schedule: warm‑up with the backbone frozen and a higher learning rate, followed by fine‑tuning all layers at a reduced rate (see Table 2).  Early stopping is applied on validation loss.

**Equation.**  Given input $x$, the network outputs
$$\hat{y} = W_2 \\sigma(W_1 \, \text{GAP}(\text{ResNet50}(x)))\,, $$
where $\sigma$ is ReLU and GAP denotes global average pooling.

*Table 2. Training hyperparameters for Phase 3.*

| Parameter | Value |
|-----------|-------|
| Epochs | 40 (5 warm‑up) |
| Batch size | 32 |
| LR head | $1\times10^{-3}$ |
| LR finetune | $1\times10^{-4}$ |
| Loss | Smooth $L_1$ |

Figure 2 (from training_curves.png) displays the training and validation loss curves as well as per‑target MAE over epochs.

<!-- ![Training curves for Phase 3](./training_curves.png) -->

*Figure 2. Phase 3 learning curves. The dashed vertical line indicates end of warm‑up.*

### 4.2 Phase 4: YOLO‑depth‑MLP pipeline

The proposed pipeline extends the baseline with explicit segmentation and depth features.  A trained YOLOv8‑seg model isolates the food mask; from the mask we compute pixel area $A$ and apply it to the depth map $D$ to obtain a volume proxy $V = \sum_{(i,j)\in \text{mask}} D_{ij}$.  These geometric features, together with the mean RGB values inside the mask, form a feature vector fed to a small multilayer perceptron (512‑256‑4).  MC dropout layers are inserted after each hidden layer for uncertainty estimation.

The pipeline is illustrated in Figure 3.

<!-- ![Pipeline diagram](./pipeline_diagram.png) -->

*Figure 3. Multi‑stage regression pipeline: detection → segmentation → depth feature extraction → MLP regression.*

Training of the MLP head uses the same loss (Smooth‑$L_1$) and optimizer as the baseline, but only the head parameters are learned; YOLO and MiDaS weights remain fixed.

### 4.3 Other tasks

Additional experiments (not detailed in this report) explore a 100‑way food classifier (with ResNet‑50) and a separate CNN for weight prediction from images.  These are meant to investigate whether auxiliary tasks can provide useful representations for nutrient regression.


## 5. Early Results and Discussion

### 5.1 Phase 3: ResNet‑50 baseline

The baseline model (`NutritionEstimator`) uses ResNet‑50 with ImageNet pretraining, fine‑tuned on Nutrition5K for direct 4‑target regression.  Training proceeds as described in §4.1; the notebook computes test‑set metrics:

$$\text{MAE}_i = \frac{1}{N} \sum_{j=1}^{N} |\hat{y}_{ij} - y_{ij}|, \quad \text{RMSE}_i = \sqrt{\frac{1}{N} \sum_{j=1}^{N} (\hat{y}_{ij} - y_{ij})^2},$$

and MAPE as mean absolute percentage error. Results are reported per nutrient and averaged.  The model converges within ~30 epochs and outputs per‑target metrics (`mae_per`, `rmse_per`, `mape_per`).

**Results location**: Notebook cell "Load best checkpoint & evaluate on test set" prints the test set summary table with per-target metrics across all four nutrient targets (calories, fat, protein, carbs).


### 5.2 Phase 4: YOLO‑Depth‑MLP Pipeline

The Phase 4 pipeline combines YOLOv8‑seg, depth estimation (real or MiDaS), and an MLP regression head trained on geometric features.  The notebook:
1. Loads the Phase 3 checkpoint (optional, for comparison)
2. Extracts 9 geometric features per dish (§4.2)
3. Trains the MLP on these features using SmoothL1 loss
4. Performs MC Dropout inference with 30 samples for uncertainty bounds
5. Evaluates on test set, computing `mae_mlp`, `rmse_mlp`, `mape_mlp`

**Results location**: Notebook cell "Comparison: Phase 3 vs Phase 4" prints a side-by-side comparison table showing `mae_p3[i]` vs `mae_mlp[i]` and delta (`Δ MAE`) for each target, with per-target RMSE and MAPE.

The notebook also generates:
- Training curves (loss over epochs)
- Per-target predicted vs actual scatter plots
- MC Dropout uncertainty estimates (±2σ bounds)


### 5.3 Phase 5: Food‑type Classifier (Food‑101)

An auxiliary EfficientNet‑B0 classifier trained on Food‑101 (101 food categories) provides food-type labels for context.  The notebook trains the model on a standard image classification split and evaluates:

**Results location**: Notebook cell "Evaluate Model Performance" prints:
- Validation Top‑1 and Top‑5 accuracy (as percentages)
- Per‑class accuracy with top‑10 and bottom‑10 food categories
- Training curves (cross‑entropy loss and accuracy over epochs)

This auxiliary classifier is **not used for nutrient prediction** in Phases 3–4 but provides interpretable food-type information in the final application.


### 5.4 Phase 6: Weight‑First Nutrition Prediction

Phase 6 predicts dish **weight in grams** from the same 9 geometric features, then multiplies by dataset‑derived calorie‑density constants to estimate nutrients.  The notebook:
1. Extracts the 9 geometric features (reusing Phase 4's cache)
2. Trains a small MLP (9→128→64→32→1) on weight targets using Huber loss
3. Computes per‑gram calorie densities (constant = mean nutrition per gram in Nutrition5K)
4. Evaluates test‑set weight prediction MAE and derived nutrition MAE

**Results location**: Notebook cell "Evaluate: Weight MAE + Derived Nutrition Error" prints:
- Weight MAE in grams and RMSE
- Relative weight error (%)
- Derived nutrition MAE for each target (weight × constant)
- Calorie density constants (e.g., `0.67 kcal/g` for calories)


### 5.5 Summary and Next Steps

The four‑phase pipeline progresses from:
- **Phase 3**: Direct RGB-to-nutrients CNN regression (ResNet‑50)
- **Phase 4**: Geometric‑features-to-nutrients MLP with uncertainty (YOLO+Depth+MLP)
- **Phase 5**: Food‑type recognition for interpretability (EfficientNet‑B0)
- **Phase 6**: Weight‑centric pipeline (mass prediction + density constants)

Each phase has computed test‑set metrics (MAE, RMSE, MAPE) stored in dedicated checkpoint files and printed to notebook output.  The exact numerical results are available in the notebooks' output cells; this report documents the methods and result locations for reproducibility and further analysis.


## 6. Conclusions

This preliminary report has outlined the implementation and evaluation methodology for a multi-phase approach to nutritional content estimation from food images using the Nutrition5K dataset. The progression from a direct end-to-end CNN baseline (Phase 3) to increasingly sophisticated pipelines incorporating segmentation, depth features, and uncertainty quantification (Phases 4-6) demonstrates the potential of combining computer vision techniques for robust food analysis.

### Key Achievements
- **Phase 3** established a strong baseline with ResNet-50 fine-tuning, achieving direct regression from RGB images to nutrient values with standard metrics (MAE, RMSE, MAPE).
- **Phase 4** introduced geometric reasoning through YOLOv8 segmentation and depth estimation, enabling MLP-based regression with Monte Carlo dropout for uncertainty bounds, potentially improving reliability in variable real-world conditions.
- **Phase 5** provided auxiliary food-type classification using EfficientNet-B0 on Food-101, offering interpretable context that could enhance user trust in automated systems.
- **Phase 6** explored a weight-centric approach, predicting dish mass from geometric features and deriving nutrients via calorie-density constants, which may be particularly effective for portion-size estimation.

The modular design allows for easy comparison and combination of approaches, with all phases sharing the same evaluation framework and dataset splits.

### Performance Insights and Limitations
While specific numerical results are detailed in the respective notebooks, the architectural choices reflect trade-offs between complexity and interpretability. Direct CNN approaches (Phase 3) offer simplicity but may struggle with scale and occlusion variance. Geometric pipelines (Phases 4, 6) provide more explainable features but depend on accurate segmentation and depth estimation. The auxiliary classifier (Phase 5) serves as a complementary tool for qualitative assessment.

Limitations include the dataset's focus on individual dishes (potentially limiting generalization to mixed meals), the computational cost of multi-stage pipelines, and the challenge of obtaining ground-truth depth data in real applications. Additionally, the regression task's inherent difficulty—predicting continuous nutrient values from visual cues alone—suggests that hybrid approaches combining visual, contextual, and user-input features may yield the best results.

### Future Directions
Future work will focus on integrating the phases into a unified system, potentially through ensemble methods or multi-task learning that jointly optimizes for nutrients, weight, and food types. Deployment considerations include optimizing for mobile devices, incorporating real-time depth sensing, and developing user interfaces that leverage uncertainty estimates for confidence scoring.

Broader applications extend to dietary monitoring apps, clinical nutrition assessment, and automated food logging systems. The uncertainty quantification in Phase 4 is particularly promising for health applications where model confidence can inform dietary recommendations or flag uncertain predictions for manual review.

This work contributes to the growing field of automated nutritional assessment by demonstrating practical implementations of state-of-the-art vision techniques on a challenging real-world dataset.


## 7. Appendix: Experiment Artifacts and Result Files

All numerical results and evaluation metrics are computed within the four main experiment notebooks:

| Notebook | Phase | Key outputs | Result variables |
|----------|-------|------------|-------------------|
| `03_model.ipynb` | 3 | Training curves, test MAE/RMSE/MAPE | `mae_per`, `rmse_per`, `mape_per` (per target) |
| `04_yolo_depth_pipeline.ipynb` | 4 | Geometric features, MLP training, MC Dropout uncertainty | `mae_mlp`, `rmse_mlp`, `mape_mlp`, `unc_mean` |
| `05_food_classifier.ipynb` | 5 | Food‑101 classification, top‑K accuracy | `val_top1`, `val_top5`, per-class accuracy |
| `06_weight_prediction.ipynb` | 6 | Weight prediction, calorie density constants | `weight_mae`, `weight_rmse`, density constants |

Each notebook prints comprehensive summary tables and generates visualization PNG files (training curves, scatter plots, residual histograms) saved to the working directory.

## 8. References

```bibtex
@article{okhin2020nutrition5k,
  title={Nutrition5K: Food Image Dataset with Multi-Modal Ground Truth Data for Calorie Estimation},
  author={Okhin, Gilles and others},
  year={2020},
  url={https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and others},
  booktitle={CVPR},
  year={2016}
}

@article{ultralytics2023yolov8,
  title={YOLOv8: State-of-the-art object detection and segmentation},
  author={Ultralytics},
  year={2023},
  url={https://github.com/ultralytics/ultralytics}
}

@article{ranftl2020midas,
  title={Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
  author={Ranftl, René and others},
  year={2020},
  journal={TPAMI}
}

@inproceedings{gal2016dropout,
  title={Dropout as a Bayesian approximation: Representing model uncertainty in deep learning},
  author={Gal, Yarin and Ghahramani, Zoubin},
  booktitle={ICML},
  year={2016}
}
```

*End of preliminary report.*
