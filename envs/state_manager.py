# envs/state_manager.py

import json

def save_bin_state(placed_boxes, new_box, bin_dims, path):
    state = {
        "bin": {
            "dimensions": bin_dims,
            "placed_boxes": placed_boxes
        },
        "new_box": {
            "size": new_box["size"],
            "color": new_box["color"]
        }
    }

    with open(path, "w") as f:
        json.dump(state, f, indent=2)
