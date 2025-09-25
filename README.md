# facecrop
Crop faces from image(s) with InsightFace, save as squares. Useful for circular icons or thumbnails.

## Requirements

- [uv](https://docs.astral.sh/uv/)

## Quickstart

1. `uv run python main.py --help`

## Examples

### Custom expand (zooms out more)
uv run python main.py ./input ./output --expand 2.3

#### Process a single image to PNG
uv run python main.py img.jpg out --out-ext .png

#### Process a directory recursively, overwrite existing results
uv run python main.py ./input ./output --overwrite
