# envs/validator.py

def is_within_bounds(pos, size, bin_dims, verbose=False, allow_above=True):
    x, y, z = pos
    dx, dy, dz = size
    bin_w, bin_h, bin_d = bin_dims

    # z-limit: entry path can be above the bin, final placement must be inside
    z_valid = z >= 0 if allow_above else (0 <= z <= bin_d - dz)

    # if verbose:
    #     print(f"  🔍 Checking bounds for pos {pos} with size {size} vs bin {bin_dims}")
    #     print(f"    x in [0, {bin_w - dx}] → {0 <= x <= bin_w - dx}")
    #     print(f"    y in [0, {bin_h - dy}] → {0 <= y <= bin_h - dy}")
    #     if allow_above:
    #         print(f"    z ≥ 0 → {z >= 0}")
    #     else:
    #         print(f"    z in [0, {bin_d - dz}] → {0 <= z <= bin_d - dz}")

    return (
        0 <= x <= bin_w - dx and
        0 <= y <= bin_h - dy and
        z_valid
    )

def boxes_overlap(pos1, size1, pos2, size2, verbose=False):
    x1, y1, z1 = pos1
    dx1, dy1, dz1 = size1
    x2, y2, z2 = pos2
    dx2, dy2, dz2 = size2

    overlap = (
        x1 < x2 + dx2 and x1 + dx1 > x2 and
        y1 < y2 + dy2 and y1 + dy1 > y2 and
        z1 < z2 + dz2 and z1 + dz1 > z2
    )

    if verbose and overlap:
        print(f"  ❗ Overlap detected between box at {pos1} and placed box at {pos2}")

    return overlap

def is_supported(final_pos, box_size, placed_boxes, verbose=False):
    x, y, z = final_pos
    dx, dy, dz = box_size

    if z == 0:
        if verbose:
            print(f"  ✅ Box is on the floor.")
        return True

    below_z = z - 1
    supported = False

    for b in placed_boxes:
        px, py, pz = b["position"]
        sx, sy, sz = b["size"]

        if pz + sz == z:
            x_overlap = not (x + dx <= px or x >= px + sx)
            y_overlap = not (y + dy <= py or y >= py + sy)

            if verbose:
                print(f"  🔍 Checking support from box at {b['position']} with size {b['size']}")
                print(f"    x-overlap: {x_overlap}, y-overlap: {y_overlap}")

            if x_overlap and y_overlap:
                supported = True
                break

    if not supported and verbose:
        print("  ❗ Box is unsupported (floating).")

    return supported

def validate_instruction(box_instruction, placed_boxes, bin_dims):
    size = box_instruction["size"]
    path = box_instruction["path"]
    color = box_instruction.get("color", "unknown")

    print(f"\n🔎 Validating box '{color}' with size {size}")
    print(f"📦 Path length: {len(path)}")

    for i, pos in enumerate(path):
        print(f"➡️  Step {i}: Position {pos}")

        # ✅ Allow z > bin depth during initial descent, but final placement must be inside
        is_final = (i == len(path) - 1)

        if not is_within_bounds(pos, size, bin_dims, verbose=True, allow_above=not is_final):
            print(f"❌ Box '{color}' goes out of bounds at step {i}: {pos}")
            return False

        for b in placed_boxes:
            if boxes_overlap(pos, size, b["position"], b["size"], verbose=True):
                print(f"❌ Box '{color}' overlaps with placed box at step {i}")
                return False

    final_pos = path[-1]
    if not is_supported(final_pos, size, placed_boxes, verbose=True):
        print(f"❌ Box '{color}' is hanging in mid-air at final position {final_pos}")
        return False

    print(f"✅ Box '{color}' passed all validation checks.\n")
    return True
