# CortexView — Comprehensive Project Dependencies & Tech Stack

This document provides a complete, exhaustive disclosure of all frameworks, libraries, tools, binaries, and cloud infrastructure utilized across the **CortexView** project. Prepared for hackathon evaluation, licensing compliance, and technical attribution.

---

## 1. Frontend Architecture & Client-Side Stack (`frontend/`)

Built with React 19, TypeScript, and Vite.

### Production Dependencies (`dependencies`)
| Dependency | Version / Spec | Category | Role in CortexView | License |
|---|---|---|---|---|
| **`react`** | `^19.2.8` | Core UI Framework | Declarative component hierarchy, state management, and DOM reconciliation. | MIT |
| **`react-dom`** | `^19.2.8` | UI Renderer | Renders React components to the browser DOM. | MIT |
| **`lucide-react`** | `^1.55.0` | Iconography | Medical, layer, visibility, and measurement pin interface icons. | ISC |
| **`react-markdown`** | `^10.1.0` | Content Rendering | Renders formatted markdown for cards, tooltips, and educational disclaimers. | MIT |
| **`tailwindcss`** | `^4.3.3` | Styling Engine | Utility-first CSS styling framework (Tailwind CSS v4 engine). | MIT |
| **`@tailwindcss/vite`** | `^4.3.3` | Build Integration | Official Vite plugin for Tailwind CSS v4 compilation. | MIT |

### Development & Build Tooling (`devDependencies`)
| Dependency | Version / Spec | Category | Role in CortexView | License |
|---|---|---|---|---|
| **`typescript`** | `~6.0.2` | Language Compiler | Static type checking and interface validation mirroring backend Pydantic models. | Apache 2.0 |
| **`vite`** | `^8.3.0` | Bundler / Dev Server | Fast development server, HMR, production bundling, and `/api` reverse proxy. | MIT |
| **`@vitejs/plugin-react`** | `^6.1.1` | Build Plugin | Babel/Fast Refresh compiler plugin for Vite. | MIT |
| **`oxlint`** | `^1.81.0` | Static Analysis | Rust-based linter for high-speed TypeScript and JSX verification. | MIT / Apache 2.0 |
| **`@types/react`** | `^19.2.18` | Type Definitions | Static typing headers for React core. | MIT |
| **`@types/react-dom`** | `^19.2.7` | Type Definitions | Static typing headers for React DOM. | MIT |
| **`@types/node`** | `^24.13.3` | Type Definitions | Static typing headers for Node.js build runtime. | MIT |

---

## 2. Backend & Deterministic Engine (`backend/`)

Built with Python 3.10+ and modern scientific computing libraries. Adheres strictly to Contract C4.

### Core Runtime Dependencies (`backend/requirements.txt`)
| Dependency | Category | Role in CortexView | License |
|---|---|---|---|
| **`fastapi`** | Web Framework | High-performance asynchronous REST API (`POST /api/analyze`, `GET /api/render/{job_id}`). | MIT |
| **`uvicorn`** | ASGI Server | Production-ready asynchronous web server running FastAPI. | BSD-3-Clause |
| **`python-multipart`** | Request Streaming | Multipart form-data parser handling user DICOM and ZIP file uploads. | Apache 2.0 |
| **`pydantic`** (v2) | Schema Validation | Data modeling and JSON schema enforcement (`MetricsReport`, `LayerMetrics`, `IntensityMetrics`). | MIT |
| **`numpy`** | Scientific Computing | 3D array manipulation, coordinate indexing, spatial slicing, and single-pass `np.bincount` volume estimation. | BSD-3-Clause |
| **`scipy`** | Scientific Computing | • `scipy.ndimage`: Euclidean Distance Transform (EDT) for anchor pin placement; 8-connectivity blob labeling.<br>• `scipy.spatial`: Convex Hull vertex computation and pairwise caliper distance calculation (`pdist`) for maximum lesion diameter. | BSD-3-Clause |
| **`scikit-image`** | Image Processing | Morphological analysis, region boundary detection, and connected-component property extraction. | BSD-3-Clause |
| **`pydicom`** | Medical Imaging | Reading raw `.dcm` DICOM slices, extracting `PixelSpacing`, `SliceThickness`, `RescaleSlope/Intercept`, and stripping Protected Health Information (PHI). | MIT / BSD |
| **`Pillow` (`PIL`)** | Raster Imaging | Multi-layer RGBA PNG compositing, alpha transparency mask clipping, and color palette mapping (`shared/layers.json`). | HPND |

### Testing & Verification
| Dependency | Category | Role in CortexView | License |
|---|---|---|---|
| **`pytest`** | Test Framework | Unit test execution (`tests/test_metrics.py`), validating 7 analytical geometric test cases. | MIT |
| **`pluggy`** | Plugin Architecture | Plugin management core utilized by pytest. | MIT |

---

## 3. Data Ingestion, Scraping & Analysis Tools (`scripts/analysis/`)

Scripts used for acquiring, verifying, and profiling medical research datasets.

| Dependency / Tool | Script Target | Role in CortexView | License |
|---|---|---|---|
| **`requests`** | `download_remind.py` | Automated, resumable downloading of Preop MRI and SEG series from the public TCIA REST API. | Apache 2.0 |
| **`tifffile`** | `tiffmini.py`, `scan_dataset.py` | Multi-page TIFF decoding and slice extraction for the Kaggle Lower-Grade Glioma (LGG) dataset. | BSD-3-Clause |
| **`matplotlib`** | `contact_sheet.py`, `registration_check.py` | Generating slice contact sheets, co-registration verification figures, and metric distribution histograms. | PSF |

---

## 4. Neuroimaging Validation & Ground-Truth Pipeline (`scripts/synthseg/`)

*Note: Used strictly offline for silver-standard benchmark validation on the ReMIND dataset; never bundled into live application runtime.*

### Python Neuroimaging Libraries
| Dependency | Category | Role in CortexView | License |
|---|---|---|---|
| **`SimpleITK`** | Medical Image I/O & Transform | Resampling 3D NIfTI label volumes onto native subject acquisition grids via nearest-neighbor interpolation. | Apache 2.0 |
| **`nibabel`** | Neuroimaging Format I/O | Reading, writing, and affine matrix management for NIfTI neuroimaging files (`.nii.gz`). | MIT |
| **`highdicom`** | DICOM Standard Extension | Decoding and extracting DICOM-SEG (Segmentation Objects) into discrete categorical label rasters. | MIT |
| **`tensorflow`** | Machine Learning Backend | Neural network runtime required when executing the standalone `BBillot/SynthSeg` repository. | Apache 2.0 |

### Standalone Executables, Containers & External Systems
| Tool / Binary | Execution Environment | Role in CortexView | License / Access |
|---|---|---|---|
| **`dcm2niix`** | Standalone C++ CLI | Converts raw DICOM series into standardized NIfTI files (`image.nii.gz`) while preserving geometry. | BSD-3-Clause |
| **`FreeSurfer`** ($\ge 7.4.1$) | Linux / WSL2 / Docker CLI | Provides `mri_synthseg` (SynthSeg 2.0) with `--robust` to generate automated Gray Matter and White Matter reference masks. | FreeSurfer License (Free registration) |
| **`Docker`** | Container Engine | Runs the official `freesurfer/freesurfer:7.4.1` image on host operating systems. | Apache 2.0 |
| **`dcmqi`** (`segimage2itkimage`) | Standalone CLI | Alternative tool for converting DICOM-SEG objects into ITK/NIfTI format. | BSD-3-Clause |
| **`nnU-Net`** (`nnunetv2`) | Deep Learning Framework | Self-configuring biomedical image segmentation framework used as deep learning baseline and benchmark comparison. | Apache 2.0 |

---

## 5. Generative AI Model & Inference Pipeline (Track FT)

Manages diffusion-based medical illustration rendering and hosted inference.

### LoRA Fine-Tuning & Model Stack (Local Training)
| Framework / Package | Category | Role in CortexView | License |
|---|---|---|---|
| **`FLUX.2 klein 4B`** | Foundation Model | Base diffusion foundation model fine-tuned to render medical illustrations from segmented color guide maps. | Black Forest Labs Model License |
| **`PyTorch` (`torch`)** | Deep Learning Core | Tensor operations, CUDA acceleration, and gradient backpropagation. | Modified BSD |
| **`diffusers`** | Generative Pipeline | Hugging Face diffusion pipeline loading FLUX checkpoints and sampling illustration outputs. | Apache 2.0 |
| **`transformers`** | Neural Architecture | Text/conditioning encoders and tokenizer utilities. | Apache 2.0 |
| **`accelerate`** | Training Optimization | Memory-efficient training, mixed precision (`bfloat16`), and gradient checkpointing on 8 GB VRAM GPUs. | Apache 2.0 |
| **`peft`** | Parameter-Efficient Tuning | Implements Low-Rank Adaptation (LoRA) for fine-tuning FLUX.2 klein without modifying base weights. | Apache 2.0 |
| **`bitsandbytes`** | Quantization | 4-bit / 8-bit weight quantization enabling training within consumer hardware limits (RTX 5060 8 GB). | MIT |

### Cloud Deployment & Serving
| Service / Tool | Architecture | Role in CortexView |
|---|---|---|
| **`Cloudflare Workers AI`** | Serverless Cloud GPU | Hosted inference platform executing the fine-tuned FLUX.2 klein model in response to frontend render polling. |
| **`cloudflared`** (Cloudflare Tunnel) | Reverse Proxy Daemon | Exposes local development endpoints and GPU worker bridges securely without public IPs or port forwarding. |

---

## 6. Python Standard Library Modules (Zero-Install Built-ins)

Standard Python 3.10+ modules actively utilized across scripts, backend services, and test fixtures:
* **`json`**: Parsing and serializing `shared/layers.json`, `shared/text_templates.json`, and API responses.
* **`pathlib` (`Path`)**: Operating-system agnostic file path resolution across Windows, macOS, and Linux.
* **`typing`**: Type hinting (`Dict`, `List`, `Optional`, `Tuple`, `Any`) enforcing strict interface contracts.
* **`os` & `sys`**: Process management, environment configuration, and `sys.path` resolution.
* **`hashlib`**: SHA-256 cryptographic hashing for dataset audit trails and provenance tracking.
* **`shutil`**: File system operations and directory cleanups.
* **`urllib`**: Handling URL encoding and TCIA download queries.

---

## 7. Open Data Sources & Research Benchmarks

* **ReMIND (The Cancer Imaging Archive - TCIA):** Pre-operative multimodal brain MRI with expert tumor, cerebrum, and ventricle segmentations. Distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* **Kaggle Lower-Grade Glioma (LGG-110):** Multimodal brain MRI with FLAIR abnormality contours.

---

## 8. AI Assistance & Developer Tooling Disclosures

In accordance with hackathon transparency guidelines, the following Large Language Model (LLM) agents and AI-assisted development environments were utilized during project ideation, architectural design, and implementation:

| Tool / Provider | Category |
|---|---|---|
| **Anthropic Claude** (Claude 3.5 Sonnet / Claude Code / Artifacts) | AI Pair Programmer & System Architect |
| **Google Gemini** (Gemini 2.5 / Gemini in Task & Agent Mode) | AI Pair Programmer & Technical Co-Developer |
| **GitHub Copilot** | AI Pair Programmer & Technical Co-Developer |
| **Cursor** | AI Pair Programmer & Technical Co-Developer |

*(Note: In accordance with our core principle — "Generative where it helps, deterministic where it matters" — neither Gemini nor Claude generates runtime patient data, coordinates, or diagnostic text during application execution).*

---

## 9. Academic Citations & Paper Attributions

Please cite the following publications when utilizing or referencing the deep learning segmentation architectures, evaluation pipelines, and datasets associated with CortexView:

### nnU-Net (Biomedical Deep Learning Segmentation Baseline)
> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature methods*, 18(2), 203-211. [doi:10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group},
  doi={10.1038/s41592-020-01008-z}
}
```

### SynthSeg (Zero-Retraining Brain MRI Segmentation)
> Billot, B., Greve, D. N., Puonti, O., Thielscher, A., Van Leemput, K., Fischl, B., Dalca, A. V., & Iglesias, J. E. (2023). SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. *Medical Image Analysis*, 86, 102789. [doi:10.1016/j.media.2023.102789](https://doi.org/10.1016/j.media.2023.102789)

```bibtex
@article{billot2023synthseg,
  title={SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining},
  author={Billot, Benjamin and Greve, Douglas N and Puonti, Oula and Thielscher, Axel and Van Leemput, Koen and Fischl, Bruce and Dalca, Adrian V and Iglesias, Juan Eugenio},
  journal={Medical Image Analysis},
  volume={86},
  pages={102789},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.media.2023.102789}
}
```

### SynthSeg Robust & Quality Control
> Billot, B., Magdamo, C., Cheng, Y., Arnold, S. E., Das, S., & Iglesias, J. E. (2023). Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets. *Proceedings of the National Academy of Sciences (PNAS)*, 120(9), e2216399120. [doi:10.1073/pnas.2216399120](https://doi.org/10.1073/pnas.2216399120)

### ReMIND Benchmark Dataset
> Juvekar, P., et al. (2024). ReMIND: The Brain Resection Multimodal Imaging Database. *Scientific Data*, 11, 494. [doi:10.1038/s41597-024-03295-z](https://doi.org/10.1038/s41597-024-03295-z) (TCIA collection: [doi:10.7937/3RAG-D070](https://doi.org/10.7937/3RAG-D070)).

