# Decision Log — EDD v2

## [D-001]: Dataset Selection Strategy — Slit-Lamp, Maximize Photo Count
- Date: 2026-11-03 | Model: nemotron-3.5-super-vl-free
- Context: Need public training data for cataract severity grading. User specified slit-lamp dataset, preferring the one with the most photos; lower resolution is acceptable because final testing is on mobile photos.
- Options Considered:
  1) Cataract-101 — slit-lamp images, cataract vs normal labels; smaller but modality-correct.
  2) Kaggle cataract / eye-disease collections — variable modality, need per-upload verification; potential for larger volume.
  3) Fundus/retinal datasets (ODIR-5K, RFMiD, APTOS, Messidor-2) — contain cataract labels but are a different imaging modality than slit-lamp; rejected as primary training source.
- Chosen Option: Slit-lamp dataset with the highest verified photo count; resolution deprioritized.
- Why: Matches user instruction to prioritize slit-lamp modality and volume over resolution; fundus datasets do not match the intended mobile photo domain closely enough.
- Trade-offs: Lower-resolution training images, possible label noise on severity grades, domain gap to mobile photos remains unaddressed by resolution alone.
- Revisit if: No slit-lamp dataset with sufficient volume is found, or mobile validation accuracy is too low to draw research conclusions.

## [D-002]: Model Architecture Priority — Accuracy Over Size/Speed/Compatibility
- Date: 2026-11-03 | Model: nemotron-3.5-super-vl-free
- Context: Need model architecture(s) for training. User stated slow inference, compatibility, and size are not issues; maximum accuracy is the requirement.
- Options Considered:
  1) MobileNetV3 / EfficientNet-B0 — small, fast, mobile-friendly; lower accuracy ceiling.
  2) EfficientNet-B7 — large CNN, high accuracy, heavy training.
  3) ViT-Large / DeiT-Large — vision transformer, high accuracy potential, needs more data/compute.
  4) ConvNeXt-Large — modern CNN, strong accuracy, heavy.
  5) ResNet-152 / DenseNet-201 — robust, proven on medical imaging, heavy.
- Chosen Option: Accuracy-first comparison of EfficientNet-B7, ViT-Large, ConvNeXt-Large, ResNet-152, and DenseNet-201; MobileNetV3/EfficientNet-B0 only as lightweight baselines.
- Why: Directly satisfies user requirement that accuracy is the sole optimization target; size/speed/compatibility constraints are waived.
- Trade-offs: Heavy GPU memory usage, longer Kaggle training sessions, large exported TFLite models, slow on-device inference (all accepted per user).
- Revisit if: Kaggle free-tier limits prevent completing training, or accuracy plateaus and a smaller model suffices.

## [D-004]: Final Architecture Selection — EfficientNet-B7 (Accuracy-First)
- Date: 2026-11-03 | Model: nemotron-3.5-super-vl-free
- Context: User has narrowed the architecture choice to a single model. EfficientNet-B7 selected as the primary training architecture per accuracy-first requirement (D-002).
- Options Considered:
  1) EfficientNet-B7 — large CNN, highest accuracy candidate in the EfficientNet family, heavy GPU memory.
  2) ViT-Large / ConvNeXt-Large — potential accuracy competitors but higher data/compute demands.
  3) ResNet-152 / DenseNet-201 — robust baselines but lower accuracy ceiling than EfficientNet-B7 on image classification tasks.
- Chosen Option: EfficientNet-B7 as the sole training architecture.
- Why: Direct user instruction to use EfficientNet-B7; aligns with D-002 (accuracy is the only optimization target; size/speed/compatibility waived).
- Trade-offs: Very heavy GPU memory usage on Kaggle, long training time per epoch, large TFLite export (~200+ MB), slow on-device inference — all accepted per user constraints.
- Revisit if: Kaggle free-tier GPU (16–30 GB VRAM depending on tier) cannot fit B7 with reasonable batch size, or training time exceeds Kaggle session limits despite checkpointing.

## [D-005]: Dataset Selection — Kaggle sheemazain only, 2-Class Immature/Mature
- Date: 2026-11-04 | Model: nemotron-3.5-super-vl-free
- Context: User decided to use only Kaggle sheemazain dataset (~400 images, two folders: immature/mature, slit-lamp photos, Apache 2.0 license). Mendeley dataset dropped due to unorganized structure (patient → eye → images, no ready labels).
- Options Considered:
  1) Use only Kaggle sheemazain (~400 images, 2 classes, verified slit-lamp, already organized by class).
  2) Use Kaggle sheemazain + Mendeley combined (rejected: Mendeley needs label extraction + reorganization, high manual effort).
  3) Keep searching for 5-class severity dataset with normal class (rejected: sheemazain is immediately accessible).
- Chosen Option: Kaggle sheemazain only.
- Why: Verified slit-lamp modality, research-friendly Apache 2.0 license, already organized into immature/mature class folders, immediately downloadable.
- Trade-offs: Only 2 classes (immature vs mature) — no normal, incipient, or hypermature. Cannot detect "healthy eye" or early-stage cataract. ~400 images is small for EfficientNet-B7; overfitting risk mitigated by ImageNet transfer learning + augmentation. Train/val split must be created locally (80/20).
- Revisit if: A 5-class dataset with normal class is found, or if ~400 images proves insufficient for convergence.

## [D-008]: Kaggle Notebook for B0 Fine-Tuning
- Date: 2026-11-05 | Model: nemotron-3.5-super-vl-free
- Context: EfficientNet-B0 Phase 1 (frozen backbone) completed locally on CPU but Phase 2 (fine-tuning) timed out due to CPU slowness. Need a Kaggle GPU notebook for Phase 2 fine-tuning and full export.
- Options Considered:
  1) Continue Phase 2 on local CPU — rejected: too slow (2–100 seconds per step, timed out after 10 min).
  2) Create Kaggle notebook with GPU accelerator for Phase 2 — chosen.
  3) Train only Phase 1 and skip fine-tuning — rejected: fine-tuning likely improves accuracy on this small dataset.
- Chosen Option: Kaggle notebook `src/train_efficientnet_b0_kaggle.ipynb` with GPU accelerator.
- Why: Kaggle provides free GPU, making Phase 2 fine-tuning feasible. Notebook includes checkpointing every epoch for resume on session timeout, full evaluation, and SavedModel + ONNX export.
- Trade-offs: Requires manual dataset upload to Kaggle and dataset_slug configuration. Session timeouts still possible but mitigated by checkpoint resume.
- Revisit if: B0 fine-tuning on GPU still doesn't improve accuracy, or if a different architecture is needed.

## [D-007]: Architecture Change — EfficientNet-B0 (Reduce Overfitting)
- Date: 2026-11-05 | Model: nemotron-3.5-super-vl-free
- Context: EfficientNet-B7 (64M parameters) completely collapsed on the sheemazain dataset (~410 images, 2 classes). Validation accuracy was 51.85% — the model predicted only "immature" for all 81 validation images (recall for mature = 0.0). Switched to EfficientNet-B0.
- Options Considered:
  1) EfficientNet-B0 — 5.3M parameters, 224x224 input, smallest EfficientNet variant.
  2) ResNet-18 — 11M parameters, robust baseline.
  3) MobileNetV3-Small — 2.9M parameters, very small.
  4) Continue with EfficientNet-B7 — rejected: proven to overfit catastrophically.
- Chosen Option: EfficientNet-B0.
- Why: 5.3M parameters (~12x smaller than B7) reduces overfitting on ~410 images. Phase 1 (frozen backbone) training completed successfully on CPU.
- Trade-offs: Lower accuracy ceiling on large datasets; fine-tuning Phase 2 is very slow on CPU (timed out after ~10 min).
- Revisit if: B0 underfits on larger dataset, or if fine-tuning on GPU (Kaggle) improves results further.

### EfficientNet-B0 Results (Phase 1 — frozen backbone, 224x224 input)
- **Accuracy: 59.26%** (48/81 correct) vs B7's 51.85%
- **Confusion Matrix:** [[23, 19], [14, 25]] — both classes now predicted
- **Mature recall: 64.1%** vs B7's 0.0%
- **Macro F1: 0.59** vs B7's 0.34
- **ROC-AUC: 0.60** vs B7's NaN
- Model: `models/checkpoints_b0/best.h5` (epoch 5)
- Full results: `results/b0_eval/evaluation_results.json`

## [D-009]: Binary Cataract Detection — Luminance + SVM (Stage 1) + EfficientNet-B0 Severity (Stage 2)
- Date: 2026-11-05 | Model: nemotron-3.5-super-vl-free
- Context: Current model only classifies immature vs mature (both diseased). Need binary normal vs cataract detection first, per paper "Detecting Cataract Using Smartphones" (Askarian et al., 2021) in docs/parsed_paper/.
- Options Considered:
  1) Luminance + SVM (paper's method) — 96.6% accuracy per paper, robust to smartphone/environment, lightweight, no GPU needed. Requires collecting normal eye images.
  2) Retrain EfficientNet-B0 with 3 classes (normal/immature/mature) — single model, but needs normal images + GPU training, prone to overfitting on small data.
  3) Retrain EfficientNet-B0 with 2 classes (normal vs cataract) — simpler, but loses severity grading; would need separate model for immature/mature.
- Chosen Option: Two-stage pipeline — Stage 1: Luminance + SVM (normal vs cataract), Stage 2: EfficientNet-B0 (immature vs mature for diseased eyes only).
- Why: Matches the paper's validated approach (96.6% accuracy), robust to mobile/environmental variations (critical for your mobile validation use case), and lightweight enough to run without GPU. Stage 2 reuses your existing EfficientNet-B0 severity grader.
- Trade-offs: Requires collecting normal/healthy eye images (none in current sheemazain dataset). Two separate models to maintain instead of one.
- Revisit if: Normal image collection proves difficult, or if a single end-to-end model is preferred later.

## [D-008]: Kaggle Notebook for B0 Fine-Tuning
- Date: 2026-11-04 | Model: nemotron-3.5-super-vl-free
- Context: No mobile app will be built. This is pure research. Need a model export format suitable for research: easy to load, inspect, resume training, and compare across frameworks.
- Options Considered:
  1) TensorFlow SavedModel — native TF format, preserves full model graph, supports resuming training, easy inspection in Python.
  2) Keras .h5 — simple single-file format, easy to load/share, but less flexible than SavedModel for custom objects.
  3) ONNX — cross-framework interoperability (PyTorch, TF, etc.), good for research comparison, but may lose some TF-specific ops.
  4) TFLite — mobile-optimized, quantized, small (rejected: no mobile app being built).
- Chosen Option: TensorFlow SavedModel as primary export, ONNX as secondary for cross-framework research.
- Why: SavedModel is the native TF format — best for research workflow (load, fine-tune, inspect, export again). ONNX adds interoperability without much overhead. TFLite is unnecessary since no mobile app is being built.
- Trade-offs: SavedModel is a directory (not a single file), slightly larger than .h5; ONNX conversion may drop unsupported ops.
- Revisit if: Model needs to be deployed to mobile/edge later, or a single-file format is required for sharing.
