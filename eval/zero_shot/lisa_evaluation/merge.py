import json
import os

merged = []
for i in range(int(os.environ['SPLIT_NUM'])):
    with open(f"tmp_ours_test/res_{i}.json", 'r') as f:
        data = json.load(f)
        merged += data

miou = sum(merged) / len(merged)
acc_05 = sum(iou >= 0.5 for iou in merged) / len(merged)
print(f"mIoU: {miou:.4f}")
print(f"Acc@0.5: {acc_05:.4f}")