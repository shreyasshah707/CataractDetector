# Implementation Plan — EDD v2: Cataract Severity Model Research

**Project:** Cataract severity grading model research (slit-lamp trained, mobile validation)  
**Mode:** Research only — not for public use, no accuracy requirements, no disclaimers, no app-store publishing  
**Owner:** Solo developer  
**Budget:** $0 (no paid services)  
**Training:** Kaggle (GPU) for Stage 2; local CPU for Stage 1  
**Deployment:** None — research model export only (TensorFlow SavedModel + ONNX + SVM pickle)  
**Inference:** Python research scripts + Streamlit app (no mobile app)  
**Explainability:** Grad-CAM heatmaps (research/Python side)  
**Architecture:** Two-stage pipeline — Stage 1: Luminance + SVM (normal vs cataract), Stage 2: EfficientNet-B0 (immature vs mature)

---

## 1. Core assumptions (from user answers)

| Item | Decision |
|------|----------|
| Task | Severity grading (multi-class): e.g., normal / incipient / immature / mature / hypermature cataract |
| Training data | Public professional optician / slit-lamp datasets only |
| Validation data | Mobile-captured photos only — **never** used in training |
| Model input | Whole photo (no eye/pupil segmentation) |
| Accuracy requirement | None (pure research) |
| Baseline | None defined |
| Class imbalance | Not handled (per user constraint) |
| Platform | Android only |
| Photo capture | Gallery pick or in-app camera tap-to-capture, no "hold still" guidance |
| App publishing | None — research build only |
| Privacy/storage | Local only, no server |

---

## 2. Data strategy

### 2.1 Important verification: slit-lamp vs mobile photos

**Slit-lamp photos do NOT replicate mobile eye photos well.**

- Slit-lamp: controlled illumination, high resolution, dilated pupil, centered, no flash reflection.
- Mobile: ambient/flash light, low resolution, undilated pupil, off-center, eyelashes, skin, glare, motion blur.

There is **no large public labeled dataset of cataract eyes captured with mobile cameras**. Only small clinical studies exist (smartphone slit-lamp adapters, portable fundus scopes). You will need to collect and label your own mobile validation set.

**Expected research outcome:** A model trained on slit-lamp images will likely drop significantly in accuracy on mobile photos. Measuring and reporting that drop is the core research finding.

### 2.2 Public training datasets to shortlist (verify on Kaggle/source sites)

Candidates to check for slit-lamp / anterior-segment cataract images:

1. **Cataract-101** — slit-lamp images, cataract vs normal labels (verify license).
2. **Kaggle "Cataract" / "Eye Disease" collections** — check modality (slit-lamp vs external vs fundus), label quality, license.
3. **ODIR / RFMiD / APTOS / Messidor-2** — primarily **fundus/retinal** datasets with cataract among other labels; a different imaging modality than slit-lamp and even further from mobile photos.

**Verification step before training:** Download a small sample from each candidate and confirm:
- Image modality (slit-lamp vs fundus vs external)
- Label granularity (binary vs severity grades)
- License (research-only vs commercial)
- Image resolution and class distribution

### 2.3 Mobile validation dataset (no public equivalent exists)

Since no large public mobile-captured cataract dataset exists:
- Collect your own mobile eye photos (rural/field setting, matching intended deployment).
- Get labels from an ophthalmologist (consensus grading).
- Hold this out completely — zero training use, validation only.

---

## 3. Model architecture & comparison

Compare these candidates on Kaggle, tracking stats per model:

| Architecture | Pros | Cons | Android/TFLite compatibility |
|--------------|------|------|------------------------------|
| **MobileNetV3** (Small/Large) | Tiny, fast, built for mobile | Lower accuracy ceiling | Excellent |
| **EfficientNet-B0/B1** | Strong accuracy/size tradeoff | Slightly heavier than MobileNet | Excellent |
| **ResNet-50** | Robust, well-understood | Heavy, slower inference | Good (TFLite) |
| **ViT-tiny / DeiT-tiny** | Modern accuracy | Heavy, needs more data, slower | Limited/Poor on mobile |

**Deliverable:** A comparison table after initial training runs showing:
- Validation accuracy (slit-lamp hold-out)
- Per-class recall (especially severe classes)
- Model size (MB)
- Inference latency (ms) on target Android device class
- TFLite conversion success (float32 / int8 quantization)

**Recommendation to test first:** Start with **EfficientNet-B0** or **MobileNetV3-Large** as baseline; switch based on the stats table.

---

## 4. Training pipeline (Kaggle)

1. **Environment:** Kaggle GPU notebook (free tier), TensorFlow / PyTorch.
2. **Preprocessing:** Resize to model input size, normalize (ImageNet stats or dataset stats), basic augmentation (rotation, flip, brightness, contrast) — keep lightweight since input is whole photo.
3. **Transfer learning:** Pretrained ImageNet weights → fine-tune on cataract dataset.
4. **Class imbalance:** Not handled per user constraint. (Optional note: if you change your mind later, class-weighting or stratified sampling are low-effort mitigations.)
5. **Checkpoints:** Save every 1/5/10 epochs to Kaggle output; resume from latest if session/time limit hits.
6. **Early stopping:** Optional, monitor validation loss.

---

## 5. Evaluation (run ALL suitable methods every time)

For each model checkpoint, on the slit-lamp hold-out set **and** the mobile validation set:

- Confusion matrix (per severity class)
- Accuracy, precision, recall, F1-score (per-class and macro/weighted averages)
- ROC-AUC (one-vs-rest per class)
- Precision-Recall curves (important if classes are imbalanced)
- Grad-CAM heatmaps on sample images (verify model looks at lens/opaque region, not flash glare or eyelashes)
- Feature-space visualization (t-SNE/UMAP) optional
- Failure-mode notes: dark irises, post-surgery IOL, contact lenses, off-center, motion blur

Compare slit-lamp results vs mobile results to quantify the domain gap.

---

## 6. Model Export (Research Formats)

**Stack:** Python, TensorFlow SavedModel, ONNX (via tf2onnx).

**Outputs:**
- TensorFlow SavedModel directory — native format, supports resuming training, easy inspection.
- ONNX file — cross-framework interoperability for research comparison.
- Export manifest JSON — documents architecture, classes, input size, preprocessing.

**No Android app, no TFLite, no mobile deployment.**

---

## 6. Stage 1: Binary Cataract Detection (Luminance + SVM)

**Based on paper:** "Detecting Cataract Using Smartphones" (Askarian et al., IEEE JTEHM, 2021)  
**Paper results:** 96.6% accuracy, 93.75% sensitivity, 93.4% specificity on normal vs cataract

**Stack:** Python, OpenCV, scikit-learn SVM (Gaussian kernel)

**Pipeline:**
1. Median filter (denoising)
2. Watershed transform (lens ROI segmentation)
3. Luminance extraction: BV = 0.299×R + 0.587×G + 0.114×B, then luminance = 2^BV / 0.0929
4. Feature: mean luminance of lens region
5. Classifier: SVM with Gaussian kernel, 10-fold cross-validation

**Outputs:**
- SVM model (pickle)
- Luminance parameters (JSON): healthy/diseased mean luminance, decision threshold
- Evaluation metrics: sensitivity, specificity, accuracy

**Critical data gap:** Current sheemazain dataset has **no normal class** — all 410 images are cataract (immature/mature). Normal/healthy eye images must be collected separately before Stage 1 can be trained.

**No Android app, no TFLite, no mobile deployment.**

---

## 7. Model Export (Research Formats)

**Stack:** Python, TensorFlow SavedModel, ONNX (via tf2onnx).

**Outputs:**
- TensorFlow SavedModel directory — native format, supports resuming training, easy inspection.
- ONNX file — cross-framework interoperability for research comparison.
- Export manifest JSON — documents architecture, classes, input size, preprocessing.

**No Android app, no TFLite, no mobile deployment.**

---

## 8. Suggested timeline (solo, research pace)

| Phase | Weeks | Milestone |
|-------|-------|-----------|
| Data shortlisting & verification | 1–2 | Confirm datasets, licenses, download samples |
| Normal eye image collection | 2–3 | Collect ~200-400 healthy eye photos for Stage 1 |
| Stage 1: Luminance + SVM training | 1–2 | Binary normal vs cataract classifier |
| Baseline training (slit-lamp, Stage 2) | 3–4 | EfficientNet-B0 on Kaggle with checkpoints |
| Stage 1 + Stage 2 evaluation | 2 | Full eval suite on slit-lamp hold-out |
| Mobile validation collection | 2–4 | Collect + ophthalmologist-label mobile photos |
| Two-stage evaluation (slit-lamp + mobile) | 2 | Domain-gap report for both stages |
| Model export (SavedModel + ONNX + SVM) | 1 | Research-ready model artifacts |
| Streamlit app + documentation | 1 | Research UI + final report |

**Total estimate:** ~15–18 weeks part-time.

---

1. **Domain gap:** Slit-lamp → mobile accuracy drop may be large. This is the core research result; manage expectations.
2. **No public mobile dataset:** You must collect and label your own mobile validation set.
3. **No public normal-class dataset:** You must collect normal/healthy eye photos for Stage 1 binary detection.
4. **No class-imbalance handling:** Severe cataract classes may be underrepresented; recall on rare classes may suffer.
5. **Whole-photo input:** Eyelashes, skin, and flash glare may become shortcut features the model learns (check Grad-CAM).
6. **Severity label quality:** Public dataset labels may be noisy; inter-grader disagreement on severity is common.
7. **Android device variability:** Camera quality differs across phones; test on more than one device if possible.

---

## 9. Definition of "done" (research)

Since no success criteria were defined, a reasonable research endpoint:

- Stage 1 (Luminance + SVM): Trained on normal + cataract images, reports sensitivity/specificity/accuracy
- Stage 2 (EfficientNet-B0): Trained on slit-lamp dataset, evaluated on hold-out set
- Two-stage pipeline evaluated on both slit-lamp hold-out and mobile validation
- Grad-CAM heatmaps generated and inspected (Stage 2)
- Streamlit research app loads Stage 1 SVM + Stage 2 model and returns binary + severity prediction
- Short report documenting the slit-lamp → mobile performance gap for both stages
