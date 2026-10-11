# Claude Prompt — Create a 3Blue1Brown-Inspired Manim Video for CortexView

## Task

Create a polished, approximately **90-second animated explainer video** about my project, **CortexView**, using **Manim Community Edition**, the Python mathematical animation library.

Take inspiration from the visual storytelling, mathematical clarity, elegant transitions, and intuitive explanations associated with 3Blue1Brown. Do **not** copy any specific video, animation, or script. Use the general philosophy: explain a complex idea through beautiful visuals, simple geometric representations, carefully timed reveals, and a clear narrative.

I want an actual rendered animation—not just a storyboard or a collection of static slides.

The goal is to help viewers understand what the project does, why it exists, how it works conceptually, and what makes its approach interesting, all in approximately 1 minute and 30 seconds.

---

## 1. Inspect the Project First

Before writing the animation, inspect the existing repository and relevant files.

Pay attention to:

- The CortexView application and its interface.
- Its brain MRI visualization pipeline.
- Tissue-layer visualization and segmentation representations.
- Measurement and metrics calculations.
- Its current synthetic MRI implementation.
- Intended future capabilities.
- The local 4B-parameter model and fine-tuning setup, if these are part of the current project.

Use the implementation as the source of truth.

### Accuracy requirements

The existing CortexView preview includes a **synthetic MRI-like brain phantom** and computes measurements from its generated pixel labels. Do not portray this synthetic example as a real patient's MRI.

Do not imply that the application already processes uploaded DICOM files if the current implementation does not support that functionality.

Do not claim that the 4B model is already trained, integrated, or performing segmentation unless you can verify this from the codebase.

Clearly distinguish between:

1. What currently works.
2. What is being developed.
3. What is planned for the future.

If a technical claim cannot be established from the repository, leave it out or flag it for confirmation. Explain the project without exaggerating its capabilities.

---

## 2. Visual Direction

Create a distinctive visual identity inspired by mathematical explainer videos.

### Visual principles

- Dark background with a restrained, elegant color palette.
- Bright, clearly differentiated colors for brain regions and tissue layers.
- Clean geometric shapes and precise diagrams.
- Smooth camera movements and transitions.
- Mathematical notation when it genuinely helps explain a concept.
- Carefully controlled typography.
- Minimal on-screen text.
- Visual explanations instead of text-heavy slides.
- Smooth transformations between related objects.
- A feeling that the viewer is discovering the underlying idea.

Use Manim's strengths: vector graphics, equations, geometric constructions, transformations, and precise timing.

Avoid:

- Generic PowerPoint-style slides.
- Excessive text.
- Unnecessary particle effects.
- Random zooming.
- Overly complicated transitions.
- Constant motion that distracts from the explanation.
- Decorative brain imagery that communicates nothing.
- Fake medical scans presented as authentic patient data.

The aesthetic should feel scientific, modern, thoughtful, and technically sophisticated.

### Render settings

- Aspect ratio: 16:9 landscape.
- Target resolution: 1920 × 1080.
- Target frame rate: 60 FPS if practical; otherwise use a stable supported frame rate.
- Target duration: approximately 90 seconds.
- Use Manim Community Edition.
- Keep typography, spacing, and color assignments consistent.
- Ensure animations remain legible at normal playback speed.

Prioritize a complete, correct, polished video over computationally expensive effects.

---

## 3. Storyboard — Approximately 90 Seconds

Treat the timings as targets rather than rigid constraints. Keep the complete video close to 90 seconds.

### Scene 1 — The Hook (0–10 seconds)

**Narrative idea:** A medical image looks like a collection of pixels. How can those pixels become meaningful, measurable structures?

Begin with a dark screen. Reveal a stylized, clearly illustrative axial brain slice made from simple geometric shapes or a synthetic image derived from the project's existing phantom.

Slowly reveal its underlying pixel grid. Transform the image into a collection of colored regions.

The audience should intuitively understand that an image can be represented as structured data.

On-screen text:

> An image is more than pixels.

Make the opening visually striking without explaining everything immediately.

### Scene 2 — From Pixels to Tissue Layers (10–25 seconds)

Zoom into the brain illustration. Reveal its regions one at a time, using smooth transitions.

Represent the tissue categories implemented by the current project, where applicable:

- Skull and scalp.
- Gray matter.
- White matter.
- CSF and ventricles.
- A highlighted synthetic lesion region.

Use distinct colors and a clear visual legend.

Show how a labeled pixel grid can represent a segmentation mask. Visually demonstrate the difference between:

- The original synthetic MRI-like image.
- The corresponding labeled regions.
- The isolated view of a selected layer.

On-screen text:

> One image. Multiple layers of meaning.

Do not present the synthetic lesion as a real medical finding.

### Scene 3 — From Segmentation to Measurement (25–43 seconds)

This is the conceptual center of the video.

Animate a selected labeled region. Show individual pixels becoming a measurable area.

Introduce the pixel-spacing concept.

For a region containing \(N\) pixels, with pixel spacing \(s\) millimeters per pixel, show:

\[
A = N s^2
\]

Explain visually that each pixel represents an area of \(s^2\) square millimeters. Then show how the pixel count can be converted into an area measurement.

If appropriate, demonstrate a boundary around the highlighted region and explain how a geometric measurement can be derived from its pixels.

Use only formulas that correspond to the actual implementation. If the code computes lesion diameter from boundary-point distances, explain that accurately rather than implying that every possible clinical measurement is supported.

On-screen text:

> Pixels → Geometry → Measurements

The viewer should understand the relationship between image data and numerical measurements.

### Scene 4 — Generative Intelligence and Deterministic Measurement (43–60 seconds)

Introduce the design idea:

> Generative where it helps. Deterministic where it matters.

Use an elegant split-screen or transformation.

On one side, illustrate a hypothetical AI model proposing or identifying a region. On the other, show deterministic code counting labeled pixels and calculating measurements.

Visually distinguish the two processes.

For example:

```text
AI model → Candidate segmentation

Segmentation mask + pixel spacing → Calculated measurement
```

Explain that an AI system may help interpret an image, while explicit algorithms can calculate measurements from a defined mask.

**Accuracy requirement:** Present this as the intended architecture only if the current code confirms the AI is not yet integrated. Clearly distinguish the current synthetic demonstration from planned AI-assisted functionality.

Do not animate an AI model producing clinically valid segmentation unless the implementation actually supports that claim.

### Scene 5 — The Local Model and Project Architecture (60–75 seconds)

Inspect the current repository to determine the actual status of the planned local 4B model.

- If a compatible model has been downloaded or integrated, illustrate the actual architecture.
- If local fine-tuning is still being prepared, show the intended workflow without implying training is complete.

A possible conceptual diagram is:

```text
MRI Data
   ↓
Preprocessing
   ↓
Segmentation Model
   ↓
Labeled Regions
   ↓
Deterministic Measurements
   ↓
Visualization
```

Use only components that correspond to the verified implementation, or clearly label the diagram as the intended architecture.

If the model is not integrated yet, make this scene about the development direction rather than inventing a completed pipeline. Keep the diagram animated and visually connected rather than displaying it as a static flowchart.

### Scene 6 — Closing (75–90 seconds)

Return to the original brain illustration. Transform it into a clean, layered visualization with corresponding measurement labels.

Slowly pull back to reveal the full composition.

End with the project name:

> CortexView

Suggested closing line:

> Turning complex brain images into structured, measurable visual information.

If current functionality supports it, include the subtitle:

> Visualization. Segmentation. Measurement.

Do not describe the application as a diagnostic system or imply clinical validation without evidence.

Finish with a subtle visual resolution rather than an abrupt cut.

---

## 4. Required Deliverable: ElevenLabs Voiceover Script

In addition to the video, generate a complete narration script that I can copy into ElevenLabs to produce the voiceover.

This is a required deliverable.

Write natural spoken English, not an essay.

### Tone

- Curious.
- Calm.
- Intelligent.
- Slightly cinematic.
- Accessible to a technical audience without excluding beginners.
- Confident without making unsupported claims.

Avoid excessive jargon, marketing clichés, and unnatural sentences.

### Script requirements

- Approximately 90 seconds of narration.
- Target approximately 175–205 words, adjusted for natural pauses.
- Divide narration into clearly labeled scene sections in the timed version.
- Include approximate timestamps for each section.
- Keep narration aligned with the animation timeline.
- Include pronunciation guidance for technical terms where helpful.
- Use punctuation to encourage natural delivery.
- Keep pauses and emphasis separate from words intended to be spoken.

Create **two versions**:

#### Version A — Clean ElevenLabs Script

Save as `voiceover_clean.txt`.

It must contain only the words to be spoken. Do not include scene headings, timestamps, sound effects, stage directions, or production notes.

Make it easy to copy directly into ElevenLabs.

#### Version B — Timed Production Script

Save as `voiceover_timed.md`.

For every scene, include:

- Scene number.
- Start and end timestamps.
- Spoken narration.
- Visual action during narration.
- Suggested pauses and emphasis.
- Estimated narration duration.

The clean script must not include phrases like “Scene One” unless they genuinely belong in the story.

The narration must describe what the audience sees at the appropriate time. Do not discuss measurements while the animation is still introducing the title.

Also create `pronunciation_notes.md` with optional guidance for technical terms and voice delivery.

---

## 5. Implement the Animation in Manim

Create a complete, runnable Manim project inside the existing repository. Adapt this example structure to the existing project rather than duplicating directories unnecessarily:

```text
project/
├── animation/
│   ├── __init__.py
│   ├── cortexview_video.py
│   ├── components.py
│   ├── colors.py
│   ├── assets/
│   │   └── README.md
│   ├── scripts/
│   │   ├── voiceover_clean.txt
│   │   ├── voiceover_timed.md
│   │   └── pronunciation_notes.md
│   └── README.md
```

Create separate Manim scene classes or reusable components where appropriate.

Possible components:

- Reusable synthetic brain-slice illustration.
- Pixel-grid visualization.
- Tissue-layer legend.
- Segmentation-mask visualization.
- Animated area calculation.
- Deterministic measurement diagram.
- AI-model workflow illustration.
- Final CortexView composition.

Reuse actual project assets when suitable.

If the current application generates the synthetic brain phantom in JavaScript, inspect whether its representation can be exported or reproduced accurately in Python.

Do not silently change the meaning of the source data just to simplify the animation. If exact reproduction is impractical, create a clearly illustrative representation and document the distinction.

### Code quality

- Keep Python code readable and modular.
- Use reusable helper functions.
- Avoid unnecessary dependencies.
- Keep assets within the project directory.
- Do not require cloud rendering services or external APIs.
- Include comments for complex animations.
- Make scene transitions deliberate and consistent.
- Make timing, colors, and text easy to adjust.

---

## 6. Synchronize Animation and Narration

Design the animation around the narration from the beginning.

For every scene, specify:

- Start time.
- End time.
- Narration duration.
- Animation duration.
- Important visual beats.
- Transition into the next scene.

Allow enough time for viewers to understand each visual transformation. Do not force viewers to read equations or labels while they disappear.

Because ElevenLabs audio timing may vary by voice, organize the animation so timing can be adjusted later. Use explicit timing configuration where practical instead of scattering hard-coded durations throughout the code.

Do not make rendering depend on an ElevenLabs API key.

The initial deliverable must be renderable **without voiceover audio**. I will generate the voiceover separately and combine it with the video afterward.

If practical, provide an optional local workflow for combining the rendered video with a supplied narration file.

---

## 7. Rendering and Preview Commands

Provide exact commands for:

1. Installing the required Manim dependencies.
2. Rendering the complete video.
3. Rendering individual scenes for debugging.
4. Producing a lower-resolution preview.
5. Rendering the final 1080p version.
6. Combining supplied ElevenLabs voiceover audio with the video, if an optional audio assembly script is implemented.

Use Manim Community Edition and verify the installed version before relying on version-specific APIs.

Do not assume a particular operating system without checking the repository environment.

Document required system dependencies, including LaTeX if used. Avoid LaTeX dependencies where a simpler, reliable Manim text-rendering approach is sufficient.

Save output in a predictable location, for example:

```text
output/
├── cortexview_preview.mp4
├── cortexview_final.mp4
└── README.md
```

Do not commit rendered videos or large generated artifacts unless the repository's conventions require it.

---

## 8. Quality Assurance

Before considering the task complete:

- Render every scene.
- Check that the Python code runs.
- Inspect the rendered video for clipping, overlapping labels, unreadable text, and abrupt transitions.
- Verify the total duration is approximately 90 seconds.
- Confirm the narration matches the animation.
- Check scientific formulas against the actual code.
- Ensure the synthetic brain illustration is not presented as real patient data.
- Ensure claims about the AI model reflect its actual development status.
- Verify all narration files have been generated.
- Confirm the rendering commands in the README actually work.

If a full 1080p render is too expensive, first render a low-resolution preview, inspect it, fix problems, and then render the final version.

Do not claim the video has been rendered or visually inspected if you have only written the Python source.

---

## 9. Final Deliverables

I expect these files:

1. **`cortexview_video.py`** — complete Manim animation.
2. **`components.py`** — reusable animation components, if needed.
3. **`voiceover_clean.txt`** — clean narration ready for ElevenLabs.
4. **`voiceover_timed.md`** — scene-by-scene narration and animation timeline.
5. **`pronunciation_notes.md`** — optional guidance for technical terms and delivery.
6. **`README.md`** — installation, rendering, and customization instructions.
7. **`cortexview_preview.mp4`** — preview render, if rendering is possible.
8. **`cortexview_final.mp4`** — final video, if rendering is possible.

Keep narration and animation code synchronized. If the story changes, update both.

If the repository is missing important information, do not invent it. Flag missing details and proceed with what can be completed accurately.

---

## 10. Execution Order

Work in this order:

1. Inspect the CortexView repository and identify what is implemented today.
2. Draft the narration and storyboard based on verified functionality.
3. Check scientific claims and formulas against the code.
4. Implement the Manim scenes and reusable components.
5. Render a low-resolution preview.
6. Inspect the preview and fix visual or timing problems.
7. Render the final video if the environment supports it.
8. Generate and verify the clean ElevenLabs script, timed production script, and pronunciation notes.
9. Document setup, rendering, and optional audio assembly commands.
10. Report what was actually tested and rendered.

**Do not guess the model's status, training objective, dataset format, or implemented capabilities.** Keep the project scientifically accurate and clearly label planned features.

## Final Goal

Create a beautiful, intellectually satisfying 90-second explainer that teaches the idea through visual reasoning—not a promotional slideshow.

Prioritize scientific accuracy, elegant animation, clear narration, and a polished final result.
