# 3D-Floorplanner

An automated pipeline that leverages Computer Vision (YOLO) and Generative AI (Gemini) to convert standard 2D floorplan images into dynamic, multi-volume 3D architectural models. 

This project bridges the gap between static 2D layouts and 3D architectural visualization by extracting structural geometry and utilizing a Large Language Model to make contextual architectural design decisions.

---

## Features

- **Automated Feature Extraction:** Utilizes a YOLO object detection model to identify and extract spatial coordinates for walls and doors from 2D floorplan images.

- **AI-Driven Architectural Design:** Integrates Google's Gemini 2.5 Flash to act as an automated architect. The LLM analyzes the floorplan's footprint to assign optimal roof styles (e.g., flat, split-level, mono-pitch, shed) and structural parameters (overhang, parapets, canopies).

- **Procedural 3D Generation:** Employs `trimesh` and `shapely` for robust computational geometry, constructing precise 3D meshes, handling Boolean intersections, and extruding complex polygons.

- **Advanced Architectural Elements:** Automatically generates exterior features including door-aligned entrance canopies, parapet walls, roof railings, and exterior staircases.

- **Standardized Output:** Exports the final geometry as a standard `.obj` file compatible with Blender, Unity, Unreal Engine, and web-based 3D viewers.

---

## Pipeline Overview

1. **Vision Phase**  
   A 2D image (`floorplan.png`) is processed. The YOLO model outputs bounding boxes and coordinates for foundational structures.

2. **Analysis Phase**  
   `ask_gemini.py` passes the layout to the Gemini API, which returns a structured JSON payload defining the architectural style.

3. **Construction Phase**  
   `builder.py` takes the structural coordinates and AI parameters to procedurally generate the 3D meshes.

4. **Export Phase**  
   The distinct meshes (walls, floors, roofs, canopies, stairs) are concatenated and exported as `apartment.obj`.

---

## Prerequisites

- Python 3.9+
- A valid Google Gemini API Key
- A valid Groq API Key 

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Executioner501/3d-Floorplans.git
cd 3d-Floorplans
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Before running the pipeline, you must supply your API keys.

- Open `main.py` and insert your **Groq API Key**
- Open `ask_gemini.py` and insert your **Google Gemini API Key**

> ⚠️ For production use, store API keys as environment variables instead of hardcoding.

---

## Usage

### 1. Add your floorplan

Place your image in the root directory and name it:

```
floorplan.png
```

*(Or update the path in the script)*

### 2. Run the pipeline

```bash
python main.py
```

### 3. Output

After execution, the generated 3D model will be saved as:

```
apartment.obj
```

---

## Output Details

The `.obj` file includes material-based visual separation:

- **Walls:** Warm Cream  
- **Floors:** Dark Charcoal  
- **Roof/Slabs:** Dark Slate  
- **Columns/Stairs:** Light Concrete  
- **Railings:** Steel  

---

## Tech Stack

- **Computer Vision:** YOLO  
- **LLM:** Gemini 2.5 Flash  
- **Geometry:** trimesh, shapely  
- **Language:** Python  

---

## Future Improvements

- Interior room detection (bedrooms, kitchens, etc.)
- Texture mapping and realistic materials
- Web-based 3D preview (Three.js)
- Multi-floor building support

---

## License

This project is open-source and available under the MIT License.
