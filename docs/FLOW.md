# Execution Flows — EDD v2

## Flow: Two-Stage Inference (Streamlit App)
### Call Sequence
1. `app.py` -> `load_trained_model()`: Loads Stage 2 EfficientNet-B0 model (.h5 or SavedModel).
2. `app.py` -> `load_stage1_model()`: Loads Stage 1 SVM + luminance parameters (pickle + JSON).
3. `app.py` -> `preprocess_image()`: Resizes uploaded image to 224x224, rescales 1/255.
4. `app.py` -> `predict()` (Stage 2): Runs EfficientNet-B0 inference on preprocessed image.
5. `app.py` -> `extract_luminance_features()` (Stage 1): Median filter -> watershed segmentation -> mean luminance of lens ROI.
6. `app.py` -> `svm_predict()` (Stage 1): SVM with Gaussian kernel classifies normal vs cataract based on luminance.
7. `app.py` -> Two-stage logic:
   - If Stage 1 = normal: Display "No cataract detected" (STOP)
   - If Stage 1 = cataract: Run Stage 2, display immature/mature + confidence
8. `app.py` -> `save_result()`: Appends result to `results/streamlit/results.jsonl`.

### Modifying In This Change:
- `app.py` (entire Streamlit app — new file)

### Gaps / Missing Failure Handling:
- Stage 1 model (SVM + luminance) not yet trained — requires normal eye image collection
- No confidence-threshold gating on Stage 1; low-confidence binary predictions treated same as high-confidence
- Grad-CAM heatmaps slow on CPU (>3 min per image)

---

## Flow: Training Pipeline — Stage 2 (EfficientNet-B0, Kaggle GPU)
### Call Sequence
1. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 1: Setup config, load dataset from Kaggle Add Data.
2. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 2: Build ImageDataGenerators from slit-lamp train/val folders.
3. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 3: Build EfficientNet-B0 (frozen backbone, 2-class head).
4. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 4: EpochCheckpoint + EarlyStopping callbacks.
5. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 5: Phase 1 — train classification head (10 epochs).
6. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 6: Phase 2 — fine-tune full backbone (GPU, up to 30 epochs).
7. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 7: Save final Keras model (.h5).
8. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 8: Evaluate on slit-lamp hold-out (confusion matrix, P/R/F1, ROC-AUC, PR-AUC).
9. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 9: Export TensorFlow SavedModel.
10. `src/train_efficientnet_b0_kaggle.ipynb` -> Cell 10: Export ONNX.

### Modifying In This Change:
- `src/train_efficientnet_b0_kaggle.ipynb` (entire notebook)

### Gaps / Missing Failure Handling:
- No automatic resume if Kaggle session times out mid-epoch; must manually set `resume_from_checkpoint`
- Only 2 classes (immature vs mature); no normal class

---

## Flow: Training Pipeline — Stage 1 (Luminance + SVM, Local)
### Call Sequence
1. `src/train_stage1_svm.py` -> `collect_images()`: Loads normal + cataract images from folder structure.
2. `src/train_stage1_svm.py` -> `extract_luminance_features()`: For each image: median filter -> watershed segmentation -> mean luminance of lens ROI.
3. `src/train_stage1_svm.py` -> `train_svm()`: SVM with Gaussian kernel on luminance features, 10-fold cross-validation.
4. `src/train_stage1_svm.py` -> `evaluate()`: Confusion matrix, sensitivity, specificity, accuracy on hold-out.
5. `src/train_stage1_svm.py` -> `save_model()`: Saves SVM (pickle) + luminance parameters (JSON).

### Files to Create:
- `src/train_stage1_svm.py` (new)

### Gaps / Missing Failure Handling:
- Requires normal eye image collection (none in current dataset)
- Watershed segmentation may fail on low-quality mobile photos

---

## Flow: Mobile Validation Evaluation (Two-Stage)
### Call Sequence
1. `src/evaluate_mobile_two_stage.py` -> `load_stage1_model()`: Loads SVM + luminance parameters.
2. `src/evaluate_mobile_two_stage.py` -> `load_stage2_model()`: Loads EfficientNet-B0 SavedModel.
3. `src/evaluate_mobile_two_stage.py` -> `load_mobile_labels()`: Reads ground-truth CSV (image_id, label: normal/immature/mature).
4. `src/evaluate_mobile_two_stage.py` -> For each mobile image:
   a. Stage 1: luminance features -> SVM -> normal/cataract
   b. If cataract: Stage 2: EfficientNet-B0 -> immature/mature
   c. If normal: Stage 2 skipped
5. `src/evaluate_mobile_two_stage.py` -> Metrics: Stage 1 (P/R/F1, sensitivity, specificity), Stage 2 (per-class P/R/F1)
6. `src/evaluate_mobile_two_stage.py` -> Domain-gap comparison: slit-lamp vs mobile results.

### Files to Create:
- `src/evaluate_mobile_two_stage.py` (new)

### Gaps / Missing Failure Handling:
- Stage 1 model not yet trained
- Mobile validation labels need ophthalmologist grading (normal vs cataract vs immature vs mature)
