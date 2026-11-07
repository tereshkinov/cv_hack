"""Optional color evidence from approximate clothing zones, for evaluation only."""
import cv2
import numpy as np

COLOR_MODEL_ID = 'lab-ab-8x8:central-upper-lower:masked-l1:v1'


def clothing_colors(crop, visible):
    """Return chroma histograms with support; a missing region gives no evidence."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    height, width = lab.shape[:2]
    columns = slice(width // 4, width - width // 4)
    result = {}
    for name, start, end in [('upper', 0.20, 0.49), ('lower', 0.53, 0.83)]:
        rows = slice(round(height * start), round(height * end))
        band = lab[rows, columns]
        mask = visible[rows, columns] & (band[:, :, 0] >= 35) & (band[:, :, 0] <= 225)
        count = np.count_nonzero(mask)
        support = count / mask.size if mask.size else 0.0
        if count < 32 or support < 0.35:
            continue
        hist = cv2.calcHist([band], [1, 2], mask.astype(np.uint8), [8, 8], [0, 256, 0, 256])
        result[name] = {'histogram': (hist.ravel() / count).tolist(), 'support': support}
    return result


def clothing_similarity(left, right):
    scores, weights = [], []
    for name in sorted(left.keys() & right.keys()):
        a, b = left[name], right[name]
        scores.append(float(np.minimum(a['histogram'], b['histogram']).sum()))
        weights.append(min(a['support'], b['support']))
    return float(np.average(scores, weights=weights)) if weights else None
