"""Generate the high-contrast Colosseum letter-tile PNG used in the README."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "colosseum_banner.png"
LETTERS = "COLOSSEUM"
COLORS = ("#172B4D", "#A3432E", "#0B6660")
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/georgiab.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"),
)


def main() -> None:
    font_path = next((path for path in FONT_CANDIDATES if path.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("A bold serif font is required to regenerate the banner")

    canvas_width, canvas_height = 2400, 480
    tile_size, gap = 230, 19
    row_width = len(LETTERS) * tile_size + (len(LETTERS) - 1) * gap
    left = (canvas_width - row_width) // 2
    top = (canvas_height - tile_size) // 2
    font = ImageFont.truetype(str(font_path), 172)
    image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    for index, letter in enumerate(LETTERS):
        x = left + index * (tile_size + gap)
        draw.rectangle(
            (x, top, x + tile_size - 1, top + tile_size - 1),
            fill=COLORS[index % len(COLORS)],
            outline="#F3F5F7",
            width=6,
        )
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_x = x + (tile_size - (bbox[2] - bbox[0])) / 2 - bbox[0]
        text_y = top + (tile_size - (bbox[3] - bbox[1])) / 2 - bbox[1]
        draw.text((round(text_x), round(text_y)), letter, font=font, fill="#FFFFFF")

    image.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
