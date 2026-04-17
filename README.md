# 3D Floorplan Generator

An automated pipeline that leverages Computer Vision (YOLO) and Generative AI (Gemini) to convert standard 2D floorplan images into dynamic, multi-volume 3D architectural models. 

This project bridges the gap between static 2D layouts and 3D architectural visualization by extracting structural geometry and utilizing a Large Language Model to make contextual architectural design decisions.

---

## Features

- **Automated Feature Extraction:** Utilizes a custom-trained YOLO object detection model to accurately identify and extract spatial coordinates for walls and doors from 2D floorplan images.

- **AI-Driven Architectural Design:** Integrates Google's Gemini 2.5 Flash to act as an automated architect. The LLM analyzes the floorplan's footprint to assign optimal roof styles (e.g., flat, split-level, mono-pitch, shed) and structural parameters (overhang, parapets, canopies).

- **Procedural 3D Generation:** Employs `trimesh` and `shapely` for robust computational geometry, constructing precise 3D meshes, handling Boolean intersections, and extruding complex polygons.

- **Advanced Architectural Elements:** Automatically generates exterior features including door-aligned entrance canopies, parapet walls, roof railings, and exterior staircases.

- **Standardized Output:** Exports the final geometry as a standard `.obj` file compatible with Blender, Unity, Unreal Engine, and web-based 3D viewers.

---

## Model Weights & Dataset

This pipeline relies on a custom-trained YOLO model to detect architectural features.

The model weights (`doors.pt`) were trained using the **CubiCasa5k floor plan dataset**:  
https://github.com/CubiCasa/CubiCasa5k

- Contains 5,000 professionally annotated floorplans
- Provides high-quality labels for walls, doors, and spatial layouts

> ⚠️ Ensure that your `doors.pt` file is placed in the root directory before running the pipeline.

---

## Pipeline Overview

1. **Vision Phase**  
   A 2D image (`floorplan.png`) is processed by the custom YOLO model (`doors.pt`).  
   The model outputs bounding boxes and coordinates for structural elements.

2. **Analysis Phase**  
   `ask_gemini.py` sends the layout to the Gemini API, which returns structured architectural decisions in JSON format.

3. **Construction Phase**  
   `builder.py` converts coordinates and AI parameters into procedural 3D meshes.

4. **Export Phase**  
   All generated meshes are combined and exported as:
   ```
   apartment.obj
   ```

---

## Prerequisites

- Python 3.9+
- Google Gemini API Key
- Groq API Key
- Custom YOLO weights file (`doors.pt`)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Executioner501/3d-Floorplans.git
cd 3d-Floorplans
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**macOS/Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Insert your API keys before running:

- Add **Groq API Key** in `main.py`
- Add **Gemini API Key** in `ask_gemini.py`

> ⚠️ Recommended: Use environment variables instead of hardcoding keys.

---

## Usage

### 1. Add input image

Place your floorplan in the root directory:

```
floorplan.png
```

### 2. Add model weights

Ensure this file exists:

```
doors.pt
```

### 3. Run the pipeline

```bash
python main.py
```

### 4. Output

After execution, the generated 3D model will be saved as:

```
apartment.obj
```

---

## Output

The resulting `.obj` file can be opened in:

- Blender
- Unity
- Unreal Engine
- Any standard 3D viewer

### Material Mapping

- **Walls:** Warm Cream  
- **Floors:** Dark Charcoal  
- **Roof/Slabs:** Dark Slate  
- **Columns/Stairs:** Light Concrete  
- **Railings:** Steel  

---

## Tech Stack

- **Computer Vision:** YOLO (custom-trained)
- **Dataset:** CubiCasa5k
- **LLM:** Gemini 2.5 Flash
- **Geometry:** trimesh, shapely
- **Language:** Python

---

## Future Improvements

- Interior room classification (bedroom, kitchen, etc.)
- Realistic textures and materials
- Web-based visualization (Three.js)
- Multi-floor building support

---

## License

This project is open-source and available under the MIT License.
