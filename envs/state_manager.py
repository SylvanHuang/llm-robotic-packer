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

def check_collision(new_pos, new_size, placed_boxes):
    nx, ny, nz = new_pos
    nsx, nsy, nsz = new_size

    for box in placed_boxes:
        px, py, pz = box["position"]
        psx, psy, psz = box["size"]

        overlap_x = not (nx + nsx <= px or nx >= px + psx)
        overlap_y = not (ny + nsy <= py or ny >= py + psy)
        overlap_z = not (nz + nsz <= pz or nz >= pz + psz)

        if overlap_x and overlap_y and overlap_z:
            return True
    return False

def is_supported(new_pos, new_size, placed_boxes):
    nx, ny, nz = new_pos
    nsx, nsy, nsz = new_size

    if nz == 0:
        return True

    for box in placed_boxes:
        px, py, pz = box["position"]
        psx, psy, psz = box["size"]

        same_x = not (nx + nsx <= px or nx >= px + psx)
        same_y = not (ny + nsy <= py or ny >= py + psy)
        top_z = pz + psz

        if same_x and same_y and abs(nz - top_z) < 0.1:
            return True
    return False
