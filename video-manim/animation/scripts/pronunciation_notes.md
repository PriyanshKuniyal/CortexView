# Pronunciation and delivery notes (optional)

## Terms

| Term | Say it | Notes |
|---|---|---|
| CortexView | **KOR**-teks-vyoo | One word, stress on the first syllable. |
| MRI | em-ar-**eye** | Spell out the letters. |
| phantom | **FAN**-tum | Here it means a synthetic test image, not anything spooky. |
| s / s squared | "ess", "ess squared" | The letter s, the pixel spacing. If the voice reads "s" oddly, replace it with "ess" in the text. |
| N | "en" | The pixel count. If it is misread, write "en". |
| ReMIND | ree-**MIND** | Like the word "remind". It's the name of a public MRI dataset. |
| segmenter | **SEG**-men-ter | |
| deterministic | dee-TER-mih-**NIS**-tik | |
| millimeters | **MIL**-ih-mee-terz | |

## Delivery

- Tone: calm, curious, quietly confident. A guided look rather than a sales pitch. Smile slightly on "Same input, same answer."
- Pace: about 145–150 words per minute overall. Scene 4 is the densest (43 words in 17 s), so keep it moving. Scene 6 is the loosest, so let the last line land.
- Pauses: a beat after each question ("…and measure?"), and short pauses between the listed layers ("Scalp and skull. Gray matter. White matter. Fluid.") so they line up with the tiles appearing.
- Emphasis: "synthetic" and "simulated" are honesty markers. Say them clearly, not as asides.
- Avoid dramatic or "trailer" delivery; the visuals carry the drama.

## Suggested ElevenLabs settings (a starting point; adjust by ear)

- Model: a multilingual or English v2/v3-class narration model.
- Stability ≈ 0.45–0.55, Similarity ≈ 0.75, Style ≈ 0.1–0.2, Speaker boost on.
- Speed 0.95–1.0. If the result runs longer than ~88 s, raise the speed slightly before cutting words.
- Generate one paragraph at a time (each paragraph of `voiceover_clean.txt` is one scene). That makes it easy to place each clip at its scene start: 0:00, 0:10, 0:25, 0:43, 1:00, 1:15.
- If you prefer one take, ElevenLabs' `<break time="0.6s" />` tags can be added between paragraphs in the ElevenLabs editor. They're left out of the clean script on purpose.

## Syncing the recording to the animation

1. Note where each paragraph actually starts in the recording.
2. If a scene's narration runs longer than its slot, increase that segment's duration in `animation/config.py` (`SEGMENTS`) and push its later `BEATS` back by the same amount, then re-render.
3. Combine the audio and video with `python video-manim/animation/render.py mux narration.mp3` (add `--offset 0.3` to delay the voice).
