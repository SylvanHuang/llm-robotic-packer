# envs/state_manager.py

import json

def save_bin_state(placed_boxes, new_box, bin_dims, path="instructions/bin_state.json"):
    anchors = generate_anchor_positions(placed_boxes, new_box["size"], bin_dims)

    state = {
        "bin": {
            "width": bin_dims[0],
            "height": bin_dims[1],
            "depth": bin_dims[2]
        },
        "placed_boxes": placed_boxes,
        "incoming_box": {
            "size": new_box["size"]
        },
        "anchor_positions": anchors
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


def is_within_bounds(pos, size, bin_dims):
    """
    Check if a box at `pos` with `size` fits entirely inside the bin defined by `bin_dims`.
    """
    for i in range(3):
        if pos[i] < 0:
            return False
        if pos[i] + size[i] > bin_dims[i]:
            return False
    return True


def generate_anchor_positions(placed_boxes, new_box_size, bin_dims):
    anchors = []

    bin_w, bin_h, bin_d = bin_dims
    bw, bh, bd = new_box_size

    # Always try floor anchors first
    for x in range(0, bin_w - bw + 1):
        for y in range(0, bin_h - bh + 1):
            z = 0
            if not check_collision([x, y, z], new_box_size, placed_boxes):
                anchors.append([x, y, z])

    # Then try top of other boxes
    for box in placed_boxes:
        px, py, pz = box["position"]
        psx, psy, psz = box["size"]
        top_z = pz + psz

        for dx in range(0, psx - bw + 1):
            for dy in range(0, psy - bh + 1):
                x = px + dx
                y = py + dy
                z = top_z
                if (
                    x + bw <= bin_w and
                    y + bh <= bin_h and
                    z + bd <= bin_d and
                    not check_collision([x, y, z], new_box_size, placed_boxes)
                ):
                    anchors.append([x, y, z])

    return anchors
