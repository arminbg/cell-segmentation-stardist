"""
main.py — Run the complete pipeline in one command!

Runs all 4 steps automatically:
  Step 1: Convert masks (preserving interphase/mitotic classes)
  Step 2: Train StarDist model
  Step 3: Evaluate (overall + per class)
  Step 4: Visualize (blue=interphase, red=mitotic)

Usage:
    python main.py
"""

import os, sys, time

print("=" * 55)
print("  CELL SEGMENTATION PIPELINE — StarDist")


print("=" * 55)

def run_step(num, name, script):
    print(f"\n{'─'*55}")
    print(f"  STEP {num}: {name}")
    print(f"{'─'*55}")
    start = time.time()
    code  = os.system(f"python {script}")
    mins  = int((time.time()-start)//60)
    secs  = int((time.time()-start)%60)
    if code == 0:
        print(f"\n  ✅ Step {num} done in {mins}m {secs}s")
    else:
        print(f"\n  ❌ Step {num} FAILED! Fix the error above.")
        sys.exit(1)

start_all = time.time()

run_step(1, "Convert Masks (interphase + mitotic)", "convert_masks.py")
run_step(2, "Train StarDist Model",                 "train.py")
run_step(3, "Evaluate (Overall + Per Class)",       "evaluate_all.py")
run_step(4, "Visualize (Blue=Interphase, Red=Mitotic)", "visualize.py")

total = int((time.time()-start_all)//60)
print()
print("=" * 55)
print("PIPELINE COMPLETE!")
print(f"  Total time: {total} minutes")
print()
print("  Results:  results/ folder")
print("  Model:    models/ folder")
print()
print("  Blue cells  = Interphase (gray in original mask)")
print("  Red cells   = Mitotic (white in original mask)")
print("=" * 55)
