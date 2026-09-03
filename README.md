# NavOL project website

This repository contains the static project page for **NavOL: Navigation Policy
with Online Imitation Learning**, accepted at ICML 2026.

- Project page: https://logosroboticsgroup.github.io/NavOL/
- Paper: https://arxiv.org/abs/2605.11762
- Code: https://github.com/WAboutMe/NavOL
- Models and data: https://huggingface.co/datasets/WAboutme/NavOL

## Structure

- `index.html`: page content, metadata, and project links
- `static/css/index.css`: NavOL-specific visual design and responsive layout
- `static/js/index.js`: carousel setup, video behavior, citation copying, and
  scroll controls
- `static/images/`: paper figures and real-world photographs
- `static/videos/`: selected real-world demonstrations
- `static/pdfs/`: locally hosted document assets

## Local preview

From the repository root, start a static server:

```bash
python -m http.server 8000
```

Then open http://localhost:8000/ in a browser. A server preview is preferred to
opening `index.html` directly because it matches GitHub Pages URL behavior more
closely.

## Content updates

Keep paper, code, and asset links synchronized with the public releases. Paper
figures should be exported at web resolution before being added to
`static/images/`. Large videos should be compressed for web playback and use
`preload="metadata"` so the landing page does not eagerly download every file.

Before deployment, check:

1. every local `href` and `src` resolves;
2. desktop and mobile layouts have no horizontal overflow;
3. the browser console has no JavaScript errors;
4. paper, code, and Hugging Face links point to the intended public resources;
5. the BibTeX block matches the current paper metadata.

## Deployment

The site is served by GitHub Pages from the `gh-pages` branch. Pushing a tested
commit to that branch publishes the new version.

## Attribution and license

The page builds on the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
which was adapted from the [Nerfies](https://nerfies.github.io/) project page.
Website content is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
