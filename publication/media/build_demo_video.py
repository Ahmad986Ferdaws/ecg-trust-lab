"""Build the ECG Trust Lab research-demo video from verified UI captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH = 1920
HEIGHT = 1080
FPS = 30
BACKGROUND = "#061416"
PANEL = "#0b2022"
PANEL_2 = "#0e282b"
TEXT = "#eef8f4"
MUTED = "#a8c1bd"
TEAL = "#54dfd0"
TEAL_DARK = "#287f80"
GOLD = "#f5c856"
TRANSITION_SECONDS = 0.7
SLIDE_DURATIONS = (4.5, 5.0, 6.0, 7.0, 5.0, 5.5)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    candidates = (
        Path("C:/Windows/Fonts") / name,
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def rounded_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    *,
    radius: int = 30,
    fill: str = PANEL,
    outline: str = "#1c4c4c",
    width: int = 2,
) -> None:
    draw = ImageDraw.Draw(canvas)
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shifted = (box[0] + 8, box[1] + 14, box[2] + 8, box[3] + 14)
    shadow_draw.rounded_rectangle(shifted, radius=radius, fill=(0, 0, 0, 115))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(16)))
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def gradient_background() -> Image.Image:
    image = Image.new("RGBA", (WIDTH, HEIGHT), BACKGROUND)
    px = image.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            left_glow = max(0.0, 1.0 - math.hypot(x - 140, y - 160) / 1100)
            right_glow = max(0.0, 1.0 - math.hypot(x - 1810, y - 80) / 1050)
            r = int(6 + 2 * left_glow + 7 * right_glow)
            g = int(20 + 18 * left_glow + 8 * right_glow)
            b = int(22 + 17 * left_glow + 22 * right_glow)
            px[x, y] = (r, g, b, 255)
    return image


def fit_crop(
    source: Image.Image,
    crop: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> Image.Image:
    cut = source.crop(crop)
    tw = target[2] - target[0]
    th = target[3] - target[1]
    ratio = min(tw / cut.width, th / cut.height)
    size = (round(cut.width * ratio), round(cut.height * ratio))
    return cut.resize(size, Image.Resampling.LANCZOS)


def paste_centered(
    canvas: Image.Image,
    image: Image.Image,
    target: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x = target[0] + (target[2] - target[0] - image.width) // 2
    y = target[1] + (target[3] - target[1] - image.height) // 2
    canvas.alpha_composite(image.convert("RGBA"), (x, y))
    return (x, y, x + image.width, y + image.height)


def header(draw: ImageDraw.ImageDraw, eyebrow: str, title: str, subtitle: str) -> None:
    draw.text((90, 66), eyebrow.upper(), fill=TEAL, font=font(23, bold=True))
    draw.text((90, 104), title, fill=TEXT, font=font(55, bold=True))
    draw.text((92, 178), subtitle, fill=MUTED, font=font(27))


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    color: str = TEAL,
    fill: str = "#103335",
) -> None:
    fnt = font(24, bold=True)
    bbox = draw.textbbox((0, 0), text, font=fnt)
    w = bbox[2] - bbox[0] + 48
    h = 54
    draw.rounded_rectangle((xy[0], xy[1], xy[0] + w, xy[1] + h), radius=27, fill=fill)
    draw.text((xy[0] + 24, xy[1] + 11), text, fill=color, font=fnt)


def footer(draw: ImageDraw.ImageDraw, counter: str) -> None:
    draw.line((90, 1010, 1830, 1010), fill="#174043", width=2)
    draw.text(
        (90, 1026),
        "ECG TRUST LAB  •  PTB-XL RESEARCH PROTOTYPE",
        fill="#769792",
        font=font(19, bold=True),
    )
    draw.text((1640, 1026), counter, fill="#769792", font=font(19, bold=True))


def slide_cover() -> Image.Image:
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((90, 80, 340, 92), radius=6, fill=TEAL)
    draw.text((90, 148), "ECG TRUST LAB", fill=TEXT, font=font(82, bold=True))
    draw.text((94, 263), "A trustworthy 12-lead ECG research prototype", fill=MUTED, font=font(39))
    rounded_panel(image, (90, 400, 1830, 790), radius=38, fill="#0b2022")
    draw.text(
        (150, 458), "From waveform to a calibrated decision", fill=TEXT, font=font(49, bold=True)
    )
    draw.text(
        (150, 542),
        "Five diagnostic superclasses  •  confidence calibration  •  uncertainty gating",
        fill=MUTED,
        font=font(29),
    )
    pill(draw, (150, 648), "PTB-XL")
    pill(draw, (355, 648), "ResNet-1D")
    pill(draw, (610, 648), "Grad-CAM")
    pill(draw, (870, 648), "Reproducible audit")
    draw.rounded_rectangle(
        (90, 850, 1830, 947), radius=24, fill="#2a2415", outline="#7a6326", width=2
    )
    draw.text((128, 878), "RESEARCH USE ONLY", fill=GOLD, font=font(29, bold=True))
    draw.text(
        (440, 878),
        "Not a medical device. Not for diagnosis, treatment, triage, or emergencies.",
        fill="#f5daa0",
        font=font(27),
    )
    footer(draw, "01 / 06")
    return image


def slide_ready(initial: Image.Image) -> Image.Image:
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "Step 1",
        "Start with a frozen, ready model",
        "Select a label-free ECG example or upload an exact 12-lead record.",
    )
    rounded_panel(image, (90, 250, 1830, 960))
    capture = fit_crop(initial, (0, 0, initial.width, 1030), (120, 280, 1800, 930))
    placed = paste_centered(image, capture, (120, 280, 1800, 930))
    draw.rounded_rectangle(placed, radius=18, outline="#2c6f6c", width=3)
    pill(draw, (1520, 114), "MODEL READY")
    footer(draw, "02 / 06")
    return image


def slide_decision(gradcam: Image.Image) -> Image.Image:
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "Step 2",
        "Probabilities become a guarded decision",
        "Raw and calibrated scores are compared with frozen thresholds.",
    )
    rounded_panel(image, (90, 250, 1830, 955))
    capture = fit_crop(gradcam, (365, 95, gradcam.width - 18, 625), (120, 285, 1800, 895))
    placed = paste_centered(image, capture, (120, 285, 1800, 895))
    draw.rounded_rectangle(placed, radius=18, outline="#2c6f6c", width=3)
    pill(draw, (128, 866), "ACCEPT / DEFER GATE")
    pill(draw, (520, 866), "5 CALIBRATED PROBABILITIES")
    pill(draw, (1105, 866), "FROZEN THRESHOLDS")
    footer(draw, "03 / 06")
    return image


def slide_waveform(gradcam: Image.Image) -> Image.Image:
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "Step 3",
        "Inspect all 12 ECG leads",
        "Ten seconds of signal are shown together; highlighted bands mark model sensitivity.",
    )
    rounded_panel(image, (90, 250, 1830, 960))
    capture = fit_crop(gradcam, (375, 600, gradcam.width - 18, 1450), (120, 280, 1800, 930))
    placed = paste_centered(image, capture, (120, 280, 1800, 930))
    draw.rounded_rectangle(placed, radius=18, outline="#2c6f6c", width=3)
    pill(draw, (1320, 112), "12 LEADS  •  10 SECONDS")
    footer(draw, "04 / 06")
    return image


def slide_gradcam(gradcam: Image.Image) -> Image.Image:
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "Explanation",
        "Grad-CAM highlights influential segments",
        "The overlay helps inspect model behavior; it does not prove medical causality.",
    )
    rounded_panel(image, (90, 250, 1830, 955))
    capture = fit_crop(gradcam, (375, 590, gradcam.width - 30, 1235), (120, 280, 1800, 885))
    placed = paste_centered(image, capture, (120, 280, 1800, 885))
    draw.rounded_rectangle(placed, radius=18, outline="#2c6f6c", width=3)
    draw.rounded_rectangle(
        (125, 865, 1795, 925), radius=20, fill="#2a2415", outline="#7a6326", width=2
    )
    draw.text((157, 880), "INTERPRET WITH CARE", fill=GOLD, font=font(24, bold=True))
    draw.text(
        (495, 880),
        "Attribution shows sensitivity—not why a heart condition exists.",
        fill="#f5daa0",
        font=font(24),
    )
    footer(draw, "05 / 06")
    return image


def slide_summary() -> Image.Image:
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    draw.text((90, 75), "RESEARCH OUTCOME", fill=TEAL, font=font(24, bold=True))
    draw.text(
        (90, 116), "A complete, auditable ECG-ML workflow", fill=TEXT, font=font(55, bold=True)
    )
    rounded_panel(image, (90, 235, 960, 675), radius=34)
    draw.text((145, 290), "Final held-out comparison", fill=MUTED, font=font(27, bold=True))
    draw.text((145, 360), "0.9219", fill=TEAL, font=font(91, bold=True))
    draw.text((510, 390), "ResNet macro-AUROC", fill=TEXT, font=font(31, bold=True))
    draw.text((145, 500), "0.8974", fill="#81a6c9", font=font(62, bold=True))
    draw.text((510, 520), "Transformer macro-AUROC", fill=MUTED, font=font(27))
    rounded_panel(image, (1000, 235, 1830, 675), radius=34, fill=PANEL_2)
    points = (
        "Calibrated probabilities",
        "Uncertainty-based defer gate",
        "Robustness + subgroup audits",
        "Reproducible frozen artifacts",
    )
    for index, point in enumerate(points):
        y = 292 + index * 83
        draw.ellipse((1055, y + 5, 1080, y + 30), fill=TEAL)
        draw.text((1110, y), point, fill=TEXT, font=font(29, bold=True))
    draw.rounded_rectangle(
        (90, 742, 1830, 927), radius=32, fill="#2a2415", outline="#7a6326", width=2
    )
    draw.text((145, 782), "RESEARCH USE ONLY", fill=GOLD, font=font(31, bold=True))
    draw.text(
        (145, 838),
        (
            "Promising evidence—not clinical approval. "
            "External and prospective validation remain necessary."
        ),
        fill="#f5daa0",
        font=font(28),
    )
    footer(draw, "06 / 06")
    return image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_video(ffmpeg: Path, slides: list[Path], output: Path) -> subprocess.CompletedProcess[str]:
    command = [str(ffmpeg), "-hide_banner", "-y"]
    for slide, duration in zip(slides, SLIDE_DURATIONS, strict=True):
        command.extend(("-loop", "1", "-t", str(duration), "-i", str(slide)))

    chains = [
        f"[{index}:v]fps={FPS},format=yuv420p,settb=AVTB[v{index}]" for index in range(len(slides))
    ]
    accumulated = SLIDE_DURATIONS[0]
    prior = "v0"
    for index in range(1, len(slides)):
        offset = accumulated - TRANSITION_SECONDS
        destination = "final" if index == len(slides) - 1 else f"x{index}"
        chains.append(
            f"[{prior}][v{index}]xfade=transition=fade:duration={TRANSITION_SECONDS}:offset={offset:.3f}[{destination}]"
        )
        accumulated += SLIDE_DURATIONS[index] - TRANSITION_SECONDS
        prior = destination

    command.extend(
        (
            "-filter_complex",
            ";".join(chains),
            "-map",
            "[final]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "19",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-metadata",
            "title=ECG Trust Lab — research demo",
            "-metadata",
            "comment=Research use only; not a medical device",
            "-t",
            f"{accumulated:.3f}",
            str(output),
        )
    )
    return subprocess.run(command, check=True, capture_output=True, text=True)


def verify_video(ffmpeg: Path, output: Path) -> str:
    probe = subprocess.run(
        (str(ffmpeg), "-hide_banner", "-i", str(output), "-f", "null", "-"),
        check=True,
        capture_output=True,
        text=True,
    )
    return probe.stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--gradcam", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.initial, args.gradcam, args.ffmpeg):
        if not required.is_file():
            raise FileNotFoundError(required)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slides_dir = args.output_dir / "slides"
    slides_dir.mkdir(exist_ok=True)
    initial = Image.open(args.initial).convert("RGB")
    gradcam = Image.open(args.gradcam).convert("RGB")
    builders = (
        slide_cover,
        lambda: slide_ready(initial),
        lambda: slide_decision(gradcam),
        lambda: slide_waveform(gradcam),
        lambda: slide_gradcam(gradcam),
        slide_summary,
    )
    slides: list[Path] = []
    for index, builder in enumerate(builders, start=1):
        path = slides_dir / f"{index:02d}.png"
        builder().convert("RGB").save(path, optimize=True)
        slides.append(path)

    poster = args.output_dir / "ecg-trust-lab-demo-poster.png"
    Image.open(slides[2]).save(poster, optimize=True)
    output = args.output_dir / "ecg-trust-lab-research-demo.mp4"
    build_video(args.ffmpeg, slides, output)
    verification = verify_video(args.ffmpeg, output)
    manifest = {
        "artifact": output.name,
        "artifact_sha256": sha256(output),
        "poster": poster.name,
        "poster_sha256": sha256(poster),
        "source_captures": {
            args.initial.name: sha256(args.initial),
            args.gradcam.name: sha256(args.gradcam),
        },
        "video": {
            "duration_seconds": round(
                sum(SLIDE_DURATIONS) - TRANSITION_SECONDS * (len(SLIDE_DURATIONS) - 1), 3
            ),
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "codec": "H.264/AVC",
            "pixel_format": "yuv420p",
            "audio": False,
        },
        "scientific_scope": "Research prototype; not a medical device or clinical validation.",
        "decode_verification": "passed",
        "ffmpeg_verification_excerpt": [
            line.strip()
            for line in verification.splitlines()
            if "Duration:" in line or "Video:" in line or "frame=" in line
        ][-4:],
    }
    (args.output_dir / "demo-video-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
