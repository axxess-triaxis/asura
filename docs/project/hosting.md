# Hosting the docs

The docs are a static site (`mkdocs build` → `site/`), so any static host works.

## Options compared

| Host | Cost | Setup | Auto-deploy on push | Custom domain | Notes |
|---|---|---|---|---|---|
| **GitHub Pages** ⭐ | Free | One workflow file | ✓ (GitHub Actions) | ✓ | Lives next to the code, no extra account. **Default choice** |
| **Cloudflare Pages** ⭐ | Free | Connect repo in dashboard | ✓ | ✓ | Fastest global CDN, unlimited bandwidth, preview URL per branch |
| Netlify | Free tier | Connect repo | ✓ | ✓ | Closest to the Vercel experience; build minutes capped |
| Read the Docs | Free (public) | `.readthedocs.yaml` | ✓ | ✓ | Built for docs; versioned docs per release; shows ads |
| Render (static) | Free | Connect repo | ✓ | ✓ | Handy if you later add an API backend |
| Hugging Face Spaces (static) | Free | Push to a Space | ✓ | ✗ | Sits next to a live Asura demo |
| AWS S3 + CloudFront / Amplify | Pay-as-you-go | CDK/console | ✓ (Amplify) | ✓ | Overkill for docs; useful if everything else is on AWS |

## GitHub Pages (configured)
`.github/workflows/docs.yml` builds with `mkdocs build --strict` and deploys on every push to `main`.
One-time setup: *Repo → Settings → Pages → Source: GitHub Actions*.
The site appears at `https://<username>.github.io/asura/`.

## Cloudflare Pages
1. Cloudflare dashboard → *Workers & Pages → Create → Pages → Connect to Git* → pick the repo.
2. Build command: `pip install -r docs/requirements.txt && mkdocs build`
3. Output directory: `site`
4. Environment variable: `PYTHON_VERSION = 3.12`

## Netlify
`netlify.toml` (already in the repo) holds the build settings. Connect the repo in Netlify and it
picks them up.
