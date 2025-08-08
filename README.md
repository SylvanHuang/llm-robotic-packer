# 🧠📦 LLM-Guided 3D Bin Packing Simulation

**LLM-Robotic-Packer** is an advanced research project that integrates **Large Language Models (LLMs)** with a **custom 3D bin-packing simulation** built using **OpenAI Gymnasium**.  
It demonstrates how language models can operate as intelligent decision-makers in **complex spatial reasoning tasks**, bridging the gap between AI-driven planning and robotic manipulation.

---

## 🚀 Project Overview

This project simulates a **robotic packing environment** where an LLM is responsible for **deciding how to place boxes inside a 3D bin**.  
It is designed to serve as both a **research platform** and a **proof-of-concept** for LLM-powered decision-making in constrained physical environments.

The environment, decision flow, and evaluation metrics are all **fully automated**, allowing reproducible experiments and scalable extensions for future research in:
- AI-guided robotics
- Warehouse automation
- Spatial reasoning for LLMs
- Reinforcement learning with LLM advisors

---

## 🛠 How It Works

1. **Custom Gymnasium Environment**  
   - A 3D bin is initialized as the packing space.  
   - Boxes of varying dimensions are introduced for placement.  
   - Supports both **static** and (soon) **dynamic** box generation.

2. **LLM as the Planner**  
   - The current bin state and **valid anchor positions** are passed to the LLM.  
   - The LLM outputs a **placement path** (and optional rotation) for the next box.  
   - The system logs rotation decisions for further analysis.

3. **Collision & Boundary Validation**  
   - Placement proposals are **validated** against bin boundaries and existing boxes.  
   - Invalid placements trigger retries until a valid one is found or attempts are exhausted.

4. **Visualization & Snapshots**  
   - After each successful placement, a **3D visualization** of the bin state is generated.  
   - Snapshots are saved to:  
     ```
     snapshots/<date_time>/
     ```

5. **Metrics & Logging**  
   - **Space Utilization** – Filled volume / Total bin volume  
   - **Placement Success Rate** – % of boxes placed successfully  
   - **Average Placement Time** – Time per box placement  
   - **Rotation Count** – Number of placements involving rotation  
   - **Box-to-Bin Fit Ratio** – Volume of placed boxes vs. total available box volume  
   - Metrics are stored in JSON format in:  
     ```
     results/<date_time>/metrics.json
     ```

---

## 📊 Current Capabilities

✅ **Pre-generated box sequences** for a complete simulation run  
✅ **Anchor-based placement guidance** for informed LLM decisions  
✅ **Full rotation support** with logging  
✅ **Collision & boundary checks** for safe placements  
✅ **3D visual snapshots** after each placement  
✅ **Detailed performance metrics** stored in structured JSON format