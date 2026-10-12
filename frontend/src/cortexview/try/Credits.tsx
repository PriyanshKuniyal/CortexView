/** Final credits: every dataset, model and tool CortexView is built on, with its licence and a link. */

interface Credit {
  name: string
  what: string
  license: string
  url: string
}

const GROUPS: { title: string; items: Credit[] }[] = [
  {
    title: 'Built and tested with',
    items: [
      {
        name: 'ReMIND',
        what: 'Brain MRIs with surgeons’ tumor outlines, Brigham and Women’s Hospital. Juvekar et al., Sci Data 2024.',
        license: 'CC BY 4.0',
        url: 'https://www.cancerimagingarchive.net/collection/remind/',
      },
      {
        name: 'LGG MRI Segmentation (Kaggle)',
        what: 'Lower-grade glioma slices with checked lesion masks, from TCGA-LGG. Buda et al., 2019.',
        license: 'CC BY-NC-SA 4.0',
        url: 'https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation',
      },
      {
        name: 'SynthSeg (FreeSurfer)',
        what: 'Gray/white-matter reference outlines, used offline only to check our segmentation. Billot et al., 2023.',
        license: 'FreeSurfer licence',
        url: 'https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg',
      },
    ],
  },
  {
    title: 'Sample scans on this page (never used for training)',
    items: [
      {
        name: 'UPenn-GBM',
        what: 'Glioblastoma MRIs, University of Pennsylvania. Bakas et al., Sci Data 2022.',
        license: 'CC BY 4.0',
        url: 'https://www.cancerimagingarchive.net/collection/upenn-gbm/',
      },
      {
        name: 'LEMON (OpenNeuro ds000221)',
        what: 'Healthy adult volunteers, MPI Leipzig Mind-Brain-Body study. Babayan et al., Sci Data 2019. Converted from NIfTI to DICOM.',
        license: 'CC0',
        url: 'https://openneuro.org/datasets/ds000221',
      },
      {
        name: 'The Cancer Imaging Archive',
        what: 'Hosts ReMIND and UPenn-GBM.',
        license: 'TCIA data use policy',
        url: 'https://www.cancerimagingarchive.net/',
      },
    ],
  },

]

export function Credits() {
  return (
    <footer className="crd" aria-label="Credits">
      <div className="crd-hd">
        <span className="mono">Credits</span>
        <p>CortexView stands on public data and open tools. Thank you to the patients, volunteers and teams who shared them.</p>
      </div>
      <div className="crd-grid">
        {GROUPS.map((g) => (
          <div key={g.title}>
            <div className="crd-h mono">{g.title}</div>
            <ul>
              {g.items.map((c) => (
                <li key={c.name}>
                  <a href={c.url} target="_blank" rel="noreferrer">
                    {c.name} <span aria-hidden="true">↗</span>
                  </a>
                  <span className="crd-lic mono">{c.license}</span>
                  <p>{c.what}</p>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <p className="crd-foot mono">Educational visualization · not a diagnostic device · no patient data is stored</p>
    </footer>
  )
}
