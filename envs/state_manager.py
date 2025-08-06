# envs/state_manager.py

import json

def save_bin_state(placed_boxes, new_box, bin_dims, path="instructions/bin_state.json"):
    state = {
        "bin": {
            "width": bin_dims[0],
            "height": bin_dims[1],
            "depth": bin_dims[2]
        },
        "placed_boxes": placed_boxes,
        "incoming_box": {
            "size": new_box["size"]
        }
    }

    with open(path, "w") as f:
        json.dump(state, f, indent=2)
