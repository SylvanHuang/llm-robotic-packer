# llm_generate.py

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

BIN_STATE_PATH = "instructions/bin_state.json"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ---------- Call 1: choose rotation + anchor by ID ----------

SYSTEM_PICK = (
    "You are choosing a placement target for a 3D bin-packing simulation. "
    "Pick EXACTLY ONE rotation and ONE anchor ID from the provided lists. "
    'Return STRICT JSON: {"rotation_index": <int>, "anchor_id": "r<idx>_a<j>"} '
    "No extra keys. No comments. No prose."
)

def choose_rotation_and_anchor(feedback: str = ""):
    """
    Ask the model to pick a rotation and an anchor by ID.
    Uses JSON response mode to ensure strict JSON output.
    """
    with open(BIN_STATE_PATH, "r") as f:
        state = json.load(f)

    # Keep only essentials in the prompt (compact JSON)
    compact = json.dumps(
        {
            "bin": state.get("bin", {}),
            "incoming_box": {
                "original_size": state.get("incoming_box", {}).get("original_size", []),
                "rotations": state.get("incoming_box", {}).get("rotations", []),
            },
            "anchors_indexed": state.get("anchors_indexed", []),
        },
        separators=(",", ":"),
    )

    user = (
        "Select exactly one rotation_index and one anchor_id from anchors_indexed.\n"
        f"{('Feedback: ' + feedback) if feedback else ''}\n"
        f"{compact}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PICK},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},  # ✅ force strict JSON
    )

    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

# ---------- Call 2: generate a path to a fixed final target ----------

SYSTEM_PATH = (
    "You are a path planner for a 3D bin-packing simulation. "
    "Rules: (1) Path must start above the bin and descend (gravity-like). "
    "(2) Final position MUST equal the given target [x,y,z]. "
    'Return STRICT JSON: {"path": [[x,y,z], ...]} '
    "No extra keys. No comments. No prose."
)

def call_gpt4_for_path_to_target(target_pos, feedback: str = ""):
    """
    Ask the model for a gravity-like path that ends exactly at target_pos.
    Uses JSON response mode to ensure strict JSON output.
    """
    user = (
        f"Final target position (must end here): {target_pos}\n"
        f"{('Feedback: ' + feedback) if feedback else ''}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PATH},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},  # ✅ force strict JSON
    )

    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
