import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from config import (BASE, SEQUENCES, IMAGES_FOLDER,
                    MASKS_CONV_FOLDER, MASKS_CLS_FOLDER,
                    MODEL_NAME, MODELS_DIR, RESULTS_DIR,
                    INTERPHASE_LABEL, MITOTIC_LABEL) 


print("=" * 55)
print("  STEP 4: Generating visualizations")
print("=" * 55)

model = StarDist2D(None, name=MODEL_NAME, basedir=MODELS_DIR)

def colorize(instance_mask, class_mask):

    colored = np.zeros((*instance_mask.shape, 3), dtype=np.float32)

    for inst_id in np.unique(instance_mask):
        if inst_id == 0:
            continue  # skip background

        cell_pixels = instance_mask == inst_id

        # Find the class of this instance
        if class_mask is not None:
            cls_vals = class_mask[cell_pixels]
            cell_class = int(np.bincount(cls_vals[cls_vals > 0]).argmax()) \
                         if np.any(cls_vals > 0) else INTERPHASE_LABEL
        else:
            cell_class = INTERPHASE_LABEL

        # Solid blue for interphase, solid red for mitotic
        if cell_class == MITOTIC_LABEL:
            color = [0.85, 0.15, 0.15]  # Red
        else:
            color = [0.15, 0.35, 0.85]  # Blue

        colored[cell_pixels] = color
    return colored

total_saved = 0




for seq in SEQUENCES:
    img_folder  = os.path.join(BASE, seq, IMAGES_FOLDER)
    msk_folder  = os.path.join(BASE, seq, MASKS_CONV_FOLDER)
    cls_folder  = os.path.join(BASE, seq, MASKS_CLS_FOLDER)
    save_folder = os.path.join(RESULTS_DIR, seq)
    os.makedirs(save_folder, exist_ok=True)

    if not os.path.exists(msk_folder):
        print(f"\n  WARNING: {msk_folder} missing — skipping {seq}")
        continue

    print(f"\n  Processing {seq}...")

    for fname in sorted(os.listdir(img_folder)):
        if not fname.endswith('.png'):
            continue


        mask_name = fname.replace('I_', 'M_')
        msk_path  = os.path.join(msk_folder, mask_name)
        cls_path  = os.path.join(cls_folder,  mask_name)

        if not os.path.exists(msk_path):
            continue

        # Load and predict
        img  = io.imread(os.path.join(img_folder, fname), as_gray=True).astype(np.float32)
        img  = normalize(img, 1, 99.8, axis=(0, 1))
        true = io.imread(msk_path).astype(np.int32)
        pred, _ = model.predict_instances(img)

        # Load class mask if available
        cls_mask = io.imread(cls_path).astype(np.uint8) if os.path.exists(cls_path) else None

        # Colorize ground truth using class mask
        true_colored = colorize(true, cls_mask)

        # For prediction: classify each predicted cell by
        # looking at the ORIGINAL image brightness under it
        # Bright = mitotic (white in mask)
        # Darker = interphase (gray in mask)
        pred_cls_mask = np.zeros_like(pred, dtype=np.uint8)
        for inst_id in np.unique(pred):
            if inst_id == 0:
                continue
            cell_px = pred == inst_id
            mean_brightness = img[cell_px].mean()
            if mean_brightness > 0.55:
                pred_cls_mask[cell_px] = MITOTIC_LABEL     # red
            else:
                pred_cls_mask[cell_px] = INTERPHASE_LABEL  # blue

        pred_colored = colorize(pred, pred_cls_mask)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'Original\n{fname}', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(true_colored)
        axes[1].set_title('Ground Truth', fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(pred_colored)
        axes[2].set_title('AI Prediction', fontsize=10)
        axes[2].axis('off')

        # Legend
        legend = [
            mpatches.Patch(color='steelblue', label='Interphase cell (gray in mask)'),
            mpatches.Patch(color='tomato',    label='Mitotic cell (white in mask)'),
            mpatches.Patch(color='black',     label='Background'),
        ]
        fig.legend(handles=legend, loc='lower center', ncol=3,
                   fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

        plt.suptitle(f'StarDist Segmentation — {seq}', fontsize=12, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_folder, f"{fname}_result.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        total_saved += 1
        print(f"    Saved: {fname}")

print("\n" + "=" * 55)
print(f"  Done! Saved {total_saved} comparison images.")
print(f"  Results folder: results/")
print()
print("  Color legend:")
print("    Blue  = Interphase cells (gray in original mask)")
print("    Red   = Mitotic cells (white in original mask)")
print("=" * 55)
print("  Visualization complete! ✅")
