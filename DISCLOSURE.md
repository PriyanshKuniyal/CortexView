# Medical, Technical, & Regulatory Disclosure

**Project:** CortexView (Plie)  
**Classification:** Educational Visualization & Anatomical Communication Tool  
**Status:** Non-Diagnostic / Proof-of-Concept Research Prototype  

---

## 1. Important Medical & Diagnostic Disclaimer

> **CORTEXVIEW IS NOT A MEDICAL DEVICE AND IS NOT INTENDED FOR CLINICAL DIAGNOSIS, PROGNOSIS, STAGING, OR TREATMENT PLANNING.**

* **No Medical Advice:** The software, visualizations, measurements, and generated annotations provided by CortexView are for **educational, instructional, and communicative purposes only**. They do not constitute medical, radiological, or clinical advice.
* **Not a Certified Diagnostic Tool:** This software has not been evaluated, cleared, or approved by the U.S. Food and Drug Administration (FDA), European Medicines Agency (CE-IVD/MDR), Central Drugs Standard Control Organization (CDSCO), or any other regulatory health authority.
* **Clinical Supervision Required:** Patients and caregivers should never alter, initiate, or discontinue any treatment based on information visualized in this application. Always consult a qualified, licensed physician or radiologist regarding medical conditions, imaging interpretation, or treatment decisions.

---

## 2. Intended Use & Target Scope

CortexView is developed to bridge the communication gap between complex volumetric radiological data and human understanding.

### Intended Use Cases
1. **Doctor-to-Patient Communication:** Assisting clinicians in explaining confirmed anatomical relationships, surgical approaches, and volumetric context to patients in a visual, accessible manner.
2. **Medical Education:** Serving as an interactive pedagogical neuroanatomy atlas for medical students, neurology trainees, and neuroscience education.
3. **Research Figure Generation:** Accelerating the creation of clean, color-coded anatomical figures from de-identified research cohorts for scientific case reports and academic literature.

### Explicitly Excluded Use Cases
* Autonomous computer-aided detection (CADe) or diagnosis (CADx).
* Triage, lesion classification (e.g., distinguishing benign from malignant lesions), or tissue characterization.
* Stereotactic surgical planning or guidance without certified surgical navigation systems.

---

## 3. Technical Architecture & AI Safety Guardrails

CortexView is designed around the core safety thesis:  
> **"Generative where it helps, deterministic where it matters."**

To prevent AI hallucinations, geometric drift, or false anatomical boundaries, the software enforces a strict structural separation between **generative rendering** and **quantitative truth**:

1. **Deterministic Geometry & Numbers (Zero AI):**
   * All anatomical segmentation masks (skull, gray matter, white matter, CSF, and suspected lesion) are isolated using deterministic classical image processing and mathematical algorithms.
   * Every measurement—including surface areas ($\text{mm}^2$), 3D volumes ($\text{mL}$), distance-to-edge anchor pin placements, and bounding boxes—is computed using pure geometry and voxel math. 
   * **No neural network or Large Language Model (LLM) generates or influences any numbers or anatomical coordinates.**
2. **Generative Model Boundary (FLUX.2 klein):**
   * The generative diffusion model is used **strictly as an aesthetic illustrator**. It converts flat color guide maps into clean editorial illustrations.
   * All generative illustrations pass through an automated **fidelity check** and are strictly clipped back to the deterministic masks via alpha transparency masks. The AI cannot redraw, bleed across, or invent anatomical boundaries.
3. **Fixed Language Templates:**
   * Text associated with data pins uses hardcoded sentence templates with strict numerical interpolation. Zero conversational or generative LLM text is applied to patient data.
   * To prevent misinterpretation, diagnostic terms such as *"tumor"*, *"cancer"*, or *"malignancy"* are strictly prohibited in runtime copy; neutral spatial descriptors (*"highlighted area"* or *"suspected lesion"*) are used instead.

---

## 4. Patient Privacy & Data Handling (HIPAA / GDPR Alignment)

* **Immediate In-Memory De-Identification:** When a DICOM file is uploaded, all Protected Health Information (PHI)—including Patient Name, Patient ID, Date of Birth, Institution, and Referring Physician—is sanitized in volatile memory.
* **No Persistent Storage:** Scans, slices, and derived masks are processed in-memory and are not written to persistent public databases. Rendered artifacts are cached transiently in memory for the duration of the user session and discarded.
* **No Remote Telemetry of Patient Data:** No identifiable patient imagery or metadata is transmitted to third-party tracking services or external foundation model APIs without local de-identification.

---

## 5. Dataset Attributions & Licensing

Research cohorts and benchmark segmentations utilized during the development and validation of CortexView include:

* **ReMIND Database (TCIA):** Juvekar, P. et al., *The Brain Resection Multimodal Imaging Database (ReMIND)*, The Cancer Imaging Archive (2023). Available under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
* **Kaggle Brain MRI Segmentation (LGG-110):** Buda, M. et al., *Association of genomic subtypes of lower-grade gliomas with shape features on MRI*, *Computers in Biology and Medicine* (2019).
* **SynthSeg Reference Pipeline (Offline Evaluation):** Billot, B. et al., *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining*, *Medical Image Analysis* (2023). *(Used strictly offline for silver-standard benchmark validation; not bundled into live application runtime).*

---

## 6. Limitation of Liability

The creators, maintainers, and contributors of CortexView provide this repository "as is," without warranty of any kind, express or implied. Under no circumstances shall the authors or copyright holders be liable for any claim, damages, or other liability arising from the use of, or inability to use, this software.