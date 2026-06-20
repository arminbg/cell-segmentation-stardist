"""
evaluate_all.py — Step 3: Evaluate model separately for each cell class.

Reports three sets of scores:
  - Overall (all cells together)
  - Interphase only (gray cells)
  - Mitotic only (white cells)

This gives a much clearer picture of model performance!

Usage:
    python evaluate_all.py
"""

import os
import numpy as np
from skimage import io
from stardist.models import StarDist2D
from stardist.matching import matching
from csbdeep.utils import normalize
from config import (BASE, SEQUENCES, IMAGES_FOLDER,
                    MASKS_CONV_FOLDER, MASKS_CLS_FOLDER,
                    MODEL_NAME, MODELS_DIR,
                    INTERPHASE_LABEL, MITOTIC_LABEL)

print("=" * 55)
print("  STEP 3: Evaluating — Interphase vs Mitotic")
print("=" * 55)

model = StarDist2D(None, name=MODEL_NAME, basedir=MODELS_DIR)

# Storage for all results
all_results = {}

for seq in SEQUENCES:
    img_folder = os.path.join(BASE, seq, IMAGES_FOLDER)
    msk_folder = os.path.join(BASE, seq, MASKS_CONV_FOLDER)
    cls_folder = os.path.join(BASE, seq, MASKS_CLS_FOLDER)

    if not os.path.exists(msk_folder):
        print(f"\n  WARNING: {msk_folder} missing — skipping {seq}")
        continue

    overall_f1  = []
    inter_f1    = []
    mito_f1     = []
    overall_iou = []
    inter_iou   = []
    mito_iou    = []

    for fname in sorted(os.listdir(img_folder)):
        if not fname.endswith('.png'):
            continue

        mask_name = fname.replace('I_', 'M_')
        msk_path  = os.path.join(msk_folder, mask_name)
        cls_path  = os.path.join(cls_folder,  mask_name)

        if not os.path.exists(msk_path):
            continue

        # Load image and predict
        img  = io.imread(os.path.join(img_folder, fname), as_gray=True).astype(np.float32)
        img  = normalize(img, 1, 99.8, axis=(0, 1))
        pred, details = model.predict_instances(img)

        # Load ground truth instance mask
        true = io.imread(msk_path).astype(np.int32)

        # Overall score (all cells)
        score = matching(true, pred, thresh=0.5)
        overall_f1.append(score.f1)
        overall_iou.append(score.mean_true_score)

        # Class-specific scores (if class mask exists)
        if os.path.exists(cls_path):
            cls_mask = io.imread(cls_path).astype(np.uint8)

            # Interphase mask — only keep interphase cell instances
            inter_mask = true.copy()
            inter_mask[cls_mask != INTERPHASE_LABEL] = 0

            # Mitotic mask — only keep mitotic cell instances
            mito_mask = true.copy()
            mito_mask[cls_mask != MITOTIC_LABEL] = 0

            # Score interphase only
            if inter_mask.max() > 0:
                s_inter = matching(inter_mask, pred, thresh=0.5)
                inter_f1.append(s_inter.f1)
                inter_iou.append(s_inter.mean_true_score)

            # Score mitotic only
            if mito_mask.max() > 0:
                s_mito = matching(mito_mask, pred, thresh=0.5)
                mito_f1.append(s_mito.f1)
                mito_iou.append(s_mito.mean_true_score)

    all_results[seq] = {
        'overall_f1':  np.mean(overall_f1)  if overall_f1  else 0,
        'overall_iou': np.mean(overall_iou) if overall_iou else 0,
        'inter_f1':    np.mean(inter_f1)    if inter_f1    else 0,
        'inter_iou':   np.mean(inter_iou)   if inter_iou   else 0,
        'mito_f1':     np.mean(mito_f1)     if mito_f1     else 0,
        'mito_iou':    np.mean(mito_iou)    if mito_iou    else 0,
    }

    r = all_results[seq]
    print(f"\n  {seq}:")
    print(f"    Overall     — F1: {r['overall_f1']:.3f}  IoU: {r['overall_iou']:.3f}")
    print(f"    Interphase  — F1: {r['inter_f1']:.3f}  IoU: {r['inter_iou']:.3f}")
    print(f"    Mitotic     — F1: {r['mito_f1']:.3f}  IoU: {r['mito_iou']:.3f}")

# Overall averages
print("\n" + "=" * 55)
print("  OVERALL RESULTS (all sequences combined):")
print(f"    Overall     — F1: {np.mean([v['overall_f1']  for v in all_results.values()]):.3f}  "
      f"IoU: {np.mean([v['overall_iou'] for v in all_results.values()]):.3f}")
print(f"    Interphase  — F1: {np.mean([v['inter_f1']    for v in all_results.values()]):.3f}  "
      f"IoU: {np.mean([v['inter_iou']   for v in all_results.values()]):.3f}")
print(f"    Mitotic     — F1: {np.mean([v['mito_f1']     for v in all_results.values()]):.3f}  "
      f"IoU: {np.mean([v['mito_iou']    for v in all_results.values()]):.3f}")
print("=" * 55)
print("  Evaluation complete! ✅")
