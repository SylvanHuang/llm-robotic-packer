# envs/metrics.py

def compute_metrics(placed_boxes, bin_dims):
    bin_w, bin_h, bin_d = bin_dims
    bin_volume = bin_w * bin_h * bin_d

    total_volume = 0
    for box in placed_boxes:
        sx, sy, sz = box["size"]
        total_volume += sx * sy * sz

    utilization = total_volume / bin_volume
    return {
        "num_boxes": len(placed_boxes),
        "total_volume": total_volume,
        "space_utilization": round(utilization, 4)
    }
