# Sticker vectorizer

These scripts turn the photos in `crops/` into the SVGs in `../svg/`.

## Needs

- Python packages: `opencv-python-headless scikit-image scipy pillow vtracer`
- Command-line tools: `potrace` and `rsvg-convert` (package `librsvg2-bin`)

## Files

- `crops.py` holds each piece's source photo and crop box.
- `vec2.py` is the main pipeline. Per-piece settings are in `CFG` at the top.
- `damage.py` is a separate textured trace for the crayon piece.
- `export.py` copies the results into `../svg`, `../photo` and `../manifest.json`.

## Rerun one piece

Put the crop at `work/<name>_crop.png`. You can copy it from `crops/<name>.jpg`. Then run:

```
python3 vec2.py <name>
python3 export.py
```

## SVG layers

Each SVG has a `sticker-border` group, which is the white die-cut shape. Each ink color is a separate group on top of it.
