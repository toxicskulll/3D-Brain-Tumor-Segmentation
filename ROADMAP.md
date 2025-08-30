# Roadmap: DARe‑Net on top of NVIDIA's BraTS 2021 codebase

This document outlines the end‑to‑end plan to build DARe‑Net (Dual‑Pathway + Attention‑based Robust Fusion) using NVIDIA’s BraTS 2021 winning pipeline as the foundation. It is structured for practical execution with clear deliverables, acceptance criteria, and risks.

Scope and principles
- Foundation: Use NVIDIA’s BraTS 2021 nnU‑Net implementation as the base (NVIDIA’s public code from DeepLearningExamples or the official NVIDIA/MONAI BraTS21 implementation). Keep its preprocessing, training, sliding‑window inference, deep supervision, and TTA intact.
- Novelty focus: Modify the network architecture only (dual‑pathway encoder + cross‑modality fusion) and extend the loss. Preserve the rest of the pipeline to ensure comparability and lower risk.
- Rigor: Reproduce baselines, maintain parameter‑fair comparisons, and run 5‑fold CV with fixed seeds. Report mean±std and paired tests for Dice/HD95.

Compute assumptions
- Preferred: ≥24 GB VRAM GPU (e.g., RTX 6000 Ada/A6000/A100). Mixed precision and gradient checkpointing will be used. If limited to 12–16 GB, we constrain attention to the lowest resolution.

Phased plan

Phase 0 — Repository and foundation setup
- [ ] Decide the exact NVIDIA foundation repo to track (DeepLearningExamples nnU‑Net vs. NVIDIA/MONAI BraTS21). Document the choice.
- [ ] Add the chosen NVIDIA repo as a git submodule under third_party/nvidia_brats21 (or vendor a pinned snapshot). Record commit hash and license.
- [ ] Create a reproducible environment (conda or pip). Pin CUDA/cuDNN/PyTorch versions to match the NVIDIA code.
- [ ] Acquire BraTS 2021 dataset; convert/preprocess per NVIDIA pipeline (consistent orientation, spacing, modality naming).
- [ ] Smoke test: run a short training on a tiny subset to verify data, augmentations, and inference paths.

Deliverable: Baseline code compiles and runs end‑to‑end on a subset.

Phase 1 — Baseline reproduction (NVIDIA pipeline as‑is)
- [ ] Train the baseline 3D nnU‑Net per NVIDIA defaults (5‑fold CV).
- [ ] Log metrics (ET/TC/WT Dice, HD95), VRAM peak, throughput. Fix seeds; store configs/logs.
- [ ] If needed, produce a capacity‑matched “wider” baseline to control for parameter count in later comparisons.

Acceptance: Baseline within reported tolerance; reproducible training and inference.

Phase 2 — DARe‑Net V0: Dual‑pathway + lightweight gating (low risk)
- [ ] Implement Dual‑Pathway Encoder: split input modalities into Path A (T1, T1Gd) and Path B (T2, FLAIR). Independent early conv blocks.
- [ ] Fusion via cross‑modal gating (SE/CBAM‑style) at the bottleneck: concatenate reduced features, produce sigmoid gates to reweight each pathway, then mix with residual.
- [ ] Add modality‑presence mask (4‑dim one‑hot broadcast spatially) to guide fusion and enable masked behavior when modalities are missing.
- [ ] Training robustness: modality dropout during training (randomly drop 1–2 modalities per sample with realistic probabilities).
- [ ] Loss: start with Dice + CE for stability; then switch to Dice + Focal (λ1 ≈ 0.25–0.5) once learning stabilizes.

Acceptance: V0 trains stably and is non‑inferior to baseline; preliminary gains on ET Dice or robustness under single‑modality ablation.

Phase 3 — DARe‑Net V1: Efficient cross‑attention at the bottleneck
- [ ] Insert multi‑head cross‑attention between pathways at the lowest resolution only. Reduce channels (e.g., 64–128) via 1×1×1 convs for Q/K/V.
- [ ] Use modality‑presence mask to zero or mask missing‑modality contributions. Add residual mixing and attention dropout.
- [ ] Keep attention local or axial if full spatial attention is heavy; alternatively, attend on pooled tokens and reweight full‑res features.
- [ ] Replace gating from V0 with attention; compare metrics and VRAM.

Acceptance: Statistically significant improvements on at least one key metric (e.g., ET Dice) without unacceptable compute cost.

Phase 4 — Loss extensions and schedules
- [ ] Add Boundary Loss (λ2 small: 0.01–0.05) using signed distance maps computed on‑the‑fly. Apply primarily on the final head; keep deep supervision with reduced/standard losses.
- [ ] Ramp up λ2 after initial epochs; consider gradient clipping to stabilize.

Acceptance: Boundary metrics improve and HD95 does not regress; training remains stable.

Phase 5 — Robustness and ablation suite
- [ ] Missing‑modality tests: Evaluate 4→3 and 4→2 for all modality combinations; report Dice drops vs. baseline.
- [ ] Perturbation tests: intensity noise, contrast shifts, bias‑field; report resilience.
- [ ] Ablations:
  - [ ] Dual‑pathway + concat (no attention/gating)
  - [ ] Single‑path capacity‑matched UNet
  - [ ] Gating vs. cross‑attention comparisons
  - [ ] Optional HeMIS‑style late fusion baseline

Acceptance: Clear attribution of gains to each component; robustness advantage over baseline demonstrated.

Phase 6 — Explainability
- [ ] Grad‑CAM adapted for segmentation heads (ET/TC/WT) on representative slices and cases.
- [ ] Complement with occlusion sensitivity or integrated gradients to avoid single‑method bias.

Acceptance: Qualitative maps show anatomically plausible focus; documentation ready for reporting.

Phase 7 — Packaging and reporting
- [ ] Save and version model checkpoints; provide inference scripts/configs compatible with the NVIDIA foundation.
- [ ] Final report: mean±std Dice/HD95 across folds (paired tests), robustness curves, compute profile (VRAM/throughput), and calibration (ECE if relevant).
- [ ] Optional: ensemble across folds; consider lightweight distillation for faster deployment.

Acceptance: Publication‑ready artifact with reproducible scripts and documented results.

Task checklist (condensed)
- [ ] Choose NVIDIA foundation repo; add as submodule; pin commit
- [ ] Environment and dataset setup; smoke test
- [ ] Reproduce baseline (5‑fold); capacity‑matched baseline
- [ ] Implement V0 dual‑path + gating + modality dropout/presence mask
- [ ] Implement V1 bottleneck cross‑attention (efficient); compare to V0
- [ ] Integrate Boundary Loss with ramp schedule
- [ ] Robustness suite (missing modalities, perturbations)
- [ ] Ablations and statistical tests
- [ ] Explainability artifacts (Grad‑CAM + one alternative)
- [ ] Package inference and final report

Risks and mitigations
- Attention overhead: confine to lowest resolution; reduce channels; use windowed/axial attention.
- Parameter fairness: include capacity‑matched baseline; report FLOPs/params/VRAM.
- Boundary loss instability: use small λ2 with ramp; gradient clipping; apply mainly to final head.
- Data variance across centers: leverage strong augmentations (bias field, gamma, noise, elastic) and consider simple histogram alignment if needed.

Success criteria
- Primary: Mean Dice improvements (especially ET) and non‑inferior or improved HD95 over baseline, averaged over 5 folds with paired significance tests.
- Secondary: Reduced performance degradation under missing modalities; reasonable compute overhead; explainability consistent with anatomy.

Immediate next actions
- [ ] Confirm the exact NVIDIA BraTS21 codebase to track and add it as a submodule
- [ ] Set up environment and run a short baseline training to validate the pipeline
- [ ] Draft API hooks for swapping the encoder and fusion modules while preserving the rest of the NVIDIA trainer/inference stack
