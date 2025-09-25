from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image
import typer
from rich.console import Console
from rich.progress import track

# --- InsightFace ---
from insightface.app import FaceAnalysis

app = typer.Typer(help="Crop faces to square thumbnails using InsightFace.")
console = Console()

# -------- Detector (initialized once) --------
_FACE_APP: Optional[FaceAnalysis] = None

def get_face_app() -> FaceAnalysis:
    global _FACE_APP
    if _FACE_APP is None:
        _FACE_APP = FaceAnalysis(name="buffalo_l")
        # ctx_id=0 uses GPU if available, else CPU; det_size is the input size for detector
        _FACE_APP.prepare(ctx_id=0, det_size=(640, 640))
    return _FACE_APP


# -------- Helpers --------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def iter_images(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMG_EXTS:
            yield path
        return
    if recursive:
        for p in path.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                yield p
    else:
        for p in path.glob("*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                yield p


def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def choose_largest_face_bbox(faces: List) -> Optional[Tuple[int, int, int, int]]:
    """
    InsightFace FaceAnalysis returns a list of face objects.
    Each has .bbox = [x1, y1, x2, y2] (floats).
    Return (x, y, w, h) for the largest area face.
    """
    if not faces:
        return None
    best = None
    best_area = -1.0
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        area = w * h
        if area > best_area:
            best_area = area
            best = (int(round(x1)), int(round(y1)), int(round(w)), int(round(h)))
    return best


def expand_to_square(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    expand_factor: float,
) -> Tuple[int, int, int, int]:
    """
    Expand the chosen bounding box by 'expand_factor' (e.g., 1.3 = 30% bigger)
    and make it a square, centered on the original box center. Clamped to image.
    Returns integer (x, y, side, side).
    """
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0

    # Expand
    w2 = w * expand_factor
    h2 = h * expand_factor

    # Make square by using the larger side
    side = max(w2, h2)

    # Top-left of square
    x0 = cx - side / 2.0
    y0 = cy - side / 2.0

    # Clamp to image bounds
    x0 = max(0, min(x0, img_w - side))
    y0 = max(0, min(y0, img_h - side))

    # Final integer box
    x0_i = int(round(x0))
    y0_i = int(round(y0))
    side_i = int(round(side))

    # If the side extends past the image edge due to rounding, clamp again
    side_i = min(side_i, img_w - x0_i, img_h - y0_i)

    return (x0_i, y0_i, side_i, side_i)


def crop_square_with_pad(
    img_bgr: np.ndarray,
    box_sq: Tuple[int, int, int, int],
    min_side: int,
) -> Image.Image:
    """
    Crops the square region. If the square is smaller than min_side, upscale after crop.
    """
    x, y, w, h = box_sq
    crop = img_bgr[y : y + h, x : x + w]

    # Convert to PIL for saving/resizing
    im = bgr_to_pil(crop)

    if min_side > 0 and (im.width < min_side or im.height < min_side):
        # upscale with high-quality resampling
        im = im.resize((min_side, min_side), Image.Resampling.LANCZOS)

    return im


def out_path_for(in_path: Path, input_root: Path, output_root: Path, out_ext: str) -> Path:
    """Mirror input directory structure under output_root, change extension."""
    rel = in_path.relative_to(input_root) if input_root.is_dir() else in_path.name
    rel = Path(rel)
    out_rel = rel.with_suffix(out_ext)
    return output_root / out_rel


# -------- Main processing --------
def process_image(
    in_img_path: Path,
    out_img_path: Path,
    expand_factor: float,
    min_side: int,
    overwrite: bool,
) -> Tuple[bool, Optional[str]]:
    try:
        if out_img_path.exists() and not overwrite:
            return False, "exists"

        img_bgr = load_image_bgr(in_img_path)
        h, w = img_bgr.shape[:2]

        faces = get_face_app().get(img_bgr)
        box = choose_largest_face_bbox(faces)

        if box is None:
            return False, "no-face"

        box_sq = expand_to_square(box, w, h, expand_factor)
        im_out = crop_square_with_pad(img_bgr, box_sq, min_side)

        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as JPEG/PNG based on extension
        ext = out_img_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            im_out.save(out_img_path, format="JPEG", quality=95, optimize=True)
        elif ext == ".png":
            im_out.save(out_img_path, format="PNG", compress_level=6)
        else:
            # default to PNG if unknown
            im_out = im_out.convert("RGB")
            im_out.save(out_img_path.with_suffix(".png"), format="PNG", compress_level=6)
        return True, None
    except Exception as e:
        return False, str(e)


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="An image file or a directory of images."),
    output_dir: Path = typer.Argument(..., help="Where to write cropped images."),
    expand: float = typer.Option(1.3, min=1.0, help="Zoom-out factor for face box (>=1.0)."),
    min_side: int = typer.Option(512, help="Minimum output side length (upscales if smaller)."),
    recursive: bool = typer.Option(True, help="Recurse into subdirectories when input is a directory."),
    out_ext: str = typer.Option(".jpg", help="Output extension: .jpg or .png."),
    overwrite: bool = typer.Option(False, help="Overwrite outputs that already exist."),
    quiet: bool = typer.Option(False, help="Less logging."),
):
    """
    Detects a face using InsightFace, expands the box, crops to a square, and saves it.
    Picks the largest face if multiple are found.
    """
    if out_ext.lower() not in {".jpg", ".jpeg", ".png"}:
        console.print("[yellow]Unsupported out_ext; using .jpg[/yellow]")
        out_ext = ".jpg"

    paths = list(iter_images(input_path, recursive))
    if not paths:
        console.print(f"[red]No images found in {input_path}[/red]")
        raise typer.Exit(code=2)

    successes = 0
    skipped_exists = 0
    skipped_no_face = 0
    failures = 0

    iterator = paths if quiet else track(paths, description="Processing")
    for p in iterator:
        out_p = out_path_for(p, input_path, output_dir, out_ext)
        ok, reason = process_image(
            in_img_path=p,
            out_img_path=out_p,
            expand_factor=expand,
            min_side=min_side,
            overwrite=overwrite,
        )
        if ok:
            successes += 1
            if not quiet:
                console.log(f"[green]Saved[/green] {out_p}")
        else:
            if reason == "exists":
                skipped_exists += 1
            elif reason == "no-face":
                skipped_no_face += 1
                if not quiet:
                    console.log(f"[yellow]No face[/yellow] in {p}")
            else:
                failures += 1
                if not quiet:
                    console.log(f"[red]Failed[/red] {p}: {reason}")

    console.print(
        f"\n[bold]Done[/bold]: {successes} saved, "
        f"{skipped_exists} skipped (exists), "
        f"{skipped_no_face} skipped (no face), "
        f"{failures} failed."
    )


if __name__ == "__main__":
    app()
