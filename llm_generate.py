# llm_generate.py

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

BIN_STATE_PATH = "instructions/bin_state.json"

SYSTEM_PROMPT = """
You are a 3D robotic box-packing assistant. Your job is to place a new box into a bin by generating a realistic path (sequence of 3D coordinates) that starts from above the bin and ends with the box resting securely on the floor or on top of another box.

Here are the physical and logical constraints:
1. The bin has fixed dimensions. Boxes must not go outside these dimensions.
2. Boxes must be fully inside the bin after placement.
3. Boxes cannot float in mid-air. The final box position must be either on the floor (z=0) or on top of another box.
4. The path should simulate gravity — the box starts at z=10 and moves down.
5. There must be no collisions with placed boxes at any point.

⚠️ Strategy Tips:
- Use the provided list of anchor positions. These are valid [x, y, z] locations that are safe to consider for final placement.
- Prefer corners and edges first to preserve central space.

Return only valid JSON like this:
{
  "size": [x, y, z],
  "path": [[x1, y1, z1], ..., [xf, yf, zf]]
}

⚠️ No markdown or explanation — return only valid raw JSON.
"""

def call_gpt4_for_path(feedback=""):
    with open(BIN_STATE_PATH, "r") as f:
        bin_state_json = json.load(f)

    anchor_list = bin_state_json.get("anchor_positions", [])
    bin_state = json.dumps(bin_state_json, indent=2)

    user_prompt = f"""Bin state:
{bin_state}

Anchors (valid final positions):
{anchor_list}

Generate a valid path to place the new box."""

    if feedback:
        user_prompt += f"\n\n⚠️ Feedback: {feedback}"

    print("🧠 Contacting GPT-4o...")

    # noinspection PyTypeChecker
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.3
    )

    reply = response.choices[0].message.content

    try:
        parsed = json.loads(reply)
        print("✅ GPT-4o returned a valid path.")
        return parsed
    except json.JSONDecodeError:
        print("❌ GPT-4o returned an invalid response.")
        return {}
