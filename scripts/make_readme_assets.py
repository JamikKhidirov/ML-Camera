"""Генерация картинок и анимаций для README (docs/assets/).

Всё рисуется локально через Pillow — никаких внешних ссылок и «мёртвых»
картинок. Запуск:

    python scripts/make_readme_assets.py

Создаёт:
    docs/assets/banner.gif    — анимированный баннер проекта
    docs/assets/features.gif  — анимация с возможностями
    docs/assets/pipeline.svg  — схема обработки кадра
    docs/assets/models.svg    — схема нейросетей проекта
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"

FONT_DIR = Path("C:/Windows/Fonts")
BG_TOP = (9, 13, 28)
BG_BOTTOM = (17, 24, 46)
CYAN = (64, 214, 255)
VIOLET = (150, 110, 255)
GREEN = (66, 232, 160)
AMBER = (255, 196, 87)
RED = (255, 92, 110)
TEXT = (238, 244, 255)
MUTED = (150, 165, 195)


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        path = FONT_DIR / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def gradient(width: int, height: int, top: tuple, bottom: tuple) -> Image.Image:
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(1, height - 1)
        draw.line(
            [(0, y), (width, y)],
            fill=tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3)),
        )
    return img


def dot_grid(img: Image.Image, step: int = 34, alpha: int = 26) -> None:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(0, img.width, step):
        for y in range(0, img.height, step):
            draw.ellipse((x, y, x + 2, y + 2), fill=(140, 180, 255, alpha))
    img.alpha_composite(overlay)


def glow_layer(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGBA", size, (0, 0, 0, 0))


def blurred_glow(img: Image.Image, layer: Image.Image, radius: int = 8) -> None:
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(radius)))


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw, xy, text, font, fill):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (right - left) / 2, xy[1] - (bottom - top) / 2), text,
              font=font, fill=fill)


def draw_camera(draw: ImageDraw.ImageDraw, box, frame: int, hot: bool = False) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    rounded(draw, box, int(w * 0.09), fill=(30, 40, 66), outline=(92, 112, 160), width=3)
    bump = (x0 + w * 0.34, y0 - int(h * 0.16), x0 + w * 0.66, y0 + int(h * 0.06))
    rounded(draw, bump, int(w * 0.03), fill=(38, 50, 80), outline=(92, 112, 160), width=2)

    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    r = int(min(w, h) * 0.42)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(8, 12, 24),
                 outline=VIOLET if hot else (84, 104, 150), width=3)
    r2 = int(r * 0.78)
    draw.ellipse((cx - r2, cy - r2, cx + r2, cy + r2), fill=(16, 24, 46),
                 outline=CYAN if hot else (56, 78, 124), width=2)
    iris = int(r2 * 0.62)
    draw.ellipse((cx - iris, cy - iris, cx + iris, cy + iris), fill=(28, 90, 130))
    draw.ellipse((cx - int(iris * 0.55), cy - int(iris * 0.62),
                  cx + int(iris * 0.18), cy - int(iris * 0.08)),
                 fill=(214, 248, 255))

    # блик по линзе, медленно вращается
    angle = frame * 8
    rr = int(r2 * 0.86)
    a0 = math.radians(angle)
    a1 = math.radians(angle + 34)
    draw.arc((cx - rr, cy - rr, cx + rr, cy + rr), start=angle, end=angle + 34,
             fill=(120, 230, 255), width=4)

    # индикатор записи
    if (frame // 10) % 2 == 0:
        draw.ellipse((x1 - 22, y0 + 12, x1 - 10, y0 + 24), fill=RED)


def make_banner() -> None:
    width, height = 1100, 300
    frames = 60
    images: list[Image.Image] = []
    title_font = load_font(82, bold=True)
    sub_font = load_font(30, bold=False)
    pill_font = load_font(22, bold=True)
    live_font = load_font(18, bold=True)

    pills = [("Лица", GREEN), ("Машины", CYAN), ("Звери", AMBER),
             ("Вещи", RED), ("Telegram", VIOLET)]

    for f in range(frames):
        t = f / frames
        img = gradient(width, height, BG_TOP, BG_BOTTOM).convert("RGBA")
        dot_grid(img)

        # мягкие цветные пятна на фоне
        glow = glow_layer((width, height))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((40, 40, 360, 360), fill=(80, 120, 255, 34))
        gd.ellipse((620, -140, 1160, 400), fill=(150, 90, 255, 26))
        blurred_glow(img, glow, 60)

        draw = ImageDraw.Draw(img)

        # сканирующая полоса
        sy = int(-60 + (height + 120) * ((t * 1.6) % 1.0))
        scan = glow_layer((width, height))
        sd = ImageDraw.Draw(scan)
        sd.rectangle((0, sy, width, sy + 26), fill=(64, 214, 255, 16))
        sd.rectangle((0, sy + 24, width, sy + 26), fill=(64, 214, 255, 70))
        blurred_glow(img, scan, 6)
        draw = ImageDraw.Draw(img)

        # камера
        draw_camera(draw, (72, 92, 272, 222), f, hot=True)
        # рамка-видоискатель вокруг камеры
        margin = 16
        corners = [
            (52, 72, 100, 72), (52, 72, 52, 120),
            (292, 72, 244, 72), (292, 72, 292, 120),
            (52, 242, 100, 242), (52, 242, 52, 194),
            (292, 242, 244, 242), (292, 242, 292, 194),
        ]
        for c in corners:
            draw.line(c, fill=CYAN, width=4)

        # заголовок
        draw.text((352, 52), "MLCamera", font=title_font, fill=TEXT)
        # акцентная линия под заголовком
        line_w = int(150 + 60 * (0.5 + 0.5 * math.sin(t * math.tau)))
        for x in range(352, 352 + line_w):
            ratio = (x - 352) / max(1, line_w - 1)
            color = tuple(round(CYAN[i] + (VIOLET[i] - CYAN[i]) * ratio) for i in range(3))
            draw.line((x, 148, x, 151), fill=color)

        draw.text((354, 166), "Умная камера для семьи", font=sub_font, fill=MUTED)

        # плашки-возможности, подсвечиваются по кругу
        active = int(t * len(pills)) % len(pills)
        x = 354
        for idx, (label, color) in enumerate(pills):
            tw = draw.textlength(label, font=pill_font)
            pw = int(tw) + 46
            ph = 38
            is_active = idx == active
            if is_active:
                halo = glow_layer((width, height))
                hd = ImageDraw.Draw(halo)
                hd.rounded_rectangle((x - 4, 218, x + pw + 4, 218 + ph + 4),
                                     radius=21, fill=color + (60,))
                blurred_glow(img, halo, 10)
                draw = ImageDraw.Draw(img)
            rounded(draw, (x, 222, x + pw, 222 + ph), 19,
                    fill=(26, 35, 60) if not is_active else tuple(min(255, c // 4 + 26) for c in color),
                    outline=color if is_active else (58, 72, 108), width=2)
            draw.ellipse((x + 16, 236, x + 26, 246), fill=color)
            draw.text((x + 34, 231), label, font=pill_font,
                      fill=TEXT if is_active else (205, 218, 240))
            x += pw + 12

        # LIVE
        if (f // 10) % 2 == 0:
            draw.ellipse((928, 28, 940, 40), fill=RED)
        draw.text((952, 24), "LIVE", font=live_font, fill=(255, 160, 175))

        images.append(img.convert("RGB"))

    OUT.mkdir(parents=True, exist_ok=True)
    save_gif(images, OUT / "banner.gif", duration=70)


def icon_person(draw, box, color):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    r = int(min(w, h) * 0.2)
    cx = x0 + w // 2
    draw.ellipse((cx - r, y0, cx + r, y0 + 2 * r), fill=color)
    draw.pieslice((x0 + int(w * 0.12), y0 + int(h * 0.5), x1 - int(w * 0.12), y0 + int(h * 1.15)),
                  start=180, end=360, fill=color)


def icon_car(draw, box, color):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    body = (x0, y0 + int(h * 0.42), x1, y1 - int(h * 0.18))
    rounded(draw, body, max(4, int(h * 0.12)), fill=color)
    cabin = (x0 + int(w * 0.2), y0 + int(h * 0.14), x0 + int(w * 0.8), y0 + int(h * 0.52))
    rounded(draw, cabin, max(4, int(h * 0.12)), fill=color)
    for cx in (x0 + int(w * 0.22), x1 - int(w * 0.22)):
        cy = y1 - int(h * 0.12)
        rr = int(h * 0.14)
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(12, 18, 34), outline=color, width=3)


def icon_plane(draw, box, color):
    x0, y0, x1, y1 = box
    pts = [(x0, y0 + int((y1 - y0) * 0.52)), (x1, y0 + int((y1 - y0) * 0.1)),
           (x1, y1 - int((y1 - y0) * 0.1)), (x0 + int((x1 - x0) * 0.28), y0 + int((y1 - y0) * 0.6)),
           (x0, y0 + int((y1 - y0) * 0.52))]
    draw.polygon(pts, fill=color)
    draw.polygon([(x0 + int((x1 - x0) * 0.34), y0 + int((y1 - y0) * 0.62)),
                  (x0 + int((x1 - x0) * 0.62), y0 + int((y1 - y0) * 0.92)),
                  (x0 + int((x1 - x0) * 0.44), y1 - int((y1 - y0) * 0.1))], fill=color)


def icon_paw(draw, box, color):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw.ellipse((x0 + int(w * 0.26), y0 + int(h * 0.42),
                  x0 + int(w * 0.74), y0 + int(h * 0.88)), fill=color)
    for cx, cy, r in ((0.22, 0.30, 0.12), (0.42, 0.18, 0.12),
                      (0.62, 0.18, 0.12), (0.82, 0.30, 0.12)):
        rr = int(w * r)
        draw.ellipse((x0 + int(w * cx) - rr, y0 + int(h * cy) - rr,
                      x0 + int(w * cx) + rr, y0 + int(h * cy) + rr), fill=color)


def icon_box(draw, box, color):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    rounded(draw, (x0, y0 + int(h * 0.18), x1, y1), max(4, int(h * 0.08)),
            fill=color)
    draw.rectangle((x0 + int(w * 0.42), y0 + int(h * 0.18),
                    x0 + int(w * 0.58), y1), fill=(20, 28, 50))
    draw.line((x0, y0 + int(h * 0.30), x1, y0 + int(h * 0.30)),
              fill=(20, 28, 50), width=max(2, int(h * 0.05)))


def icon_web(draw, box, color):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    rounded(draw, (x0, y0 + int(h * 0.08), x1, y1 - int(h * 0.08)),
            max(4, int(h * 0.10)), outline=color, width=3)
    draw.line((x0, y0 + int(h * 0.30), x1, y0 + int(h * 0.30)), fill=color, width=2)
    for i, cx in enumerate((0.14, 0.27, 0.40)):
        r = max(2, int(w * 0.04))
        draw.ellipse((x0 + int(w * cx) - r, y0 + int(h * 0.18) - r,
                      x0 + int(w * cx) + r, y0 + int(h * 0.18) + r), fill=color)
    draw.line((x0 + int(w * 0.22), y0 + int(h * 0.58),
               x0 + int(w * 0.78), y0 + int(h * 0.58)), fill=color, width=2)
    draw.line((x0 + int(w * 0.22), y0 + int(h * 0.72),
               x0 + int(w * 0.62), y0 + int(h * 0.72)), fill=color, width=2)


def icon_moon(draw, box, color):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw.ellipse((x0, y0, x0 + int(w * 0.78), y1), fill=color)
    draw.ellipse((x0 + int(w * 0.28), y0 - int(h * 0.08),
                  x0 + int(w * 0.98), y1 - int(h * 0.2)), fill=(20, 28, 50))
    # звёздочки
    def star(cx, cy, r):
        draw.polygon([(cx, cy - r), (cx + r * 0.35, cy - r * 0.35), (cx + r, cy),
                      (cx + r * 0.35, cy + r * 0.35), (cx, cy + r),
                      (cx - r * 0.35, cy + r * 0.35), (cx - r, cy),
                      (cx - r * 0.35, cy - r * 0.35)], fill=color)

    star(x1 - int(w * 0.08), y0 + int(h * 0.18), max(2, int(w * 0.05)))
    star(x1 - int(w * 0.2), y1 - int(h * 0.16), max(1, int(w * 0.035)))


def make_features() -> None:
    width, height = 900, 566
    frames = 72
    images: list[Image.Image] = []
    title_font = load_font(34, bold=True)
    card_title = load_font(25, bold=True)
    card_sub = load_font(18, bold=False)

    items = [
        ("Люди по имени", "InsightFace узнаёт родных", icon_person, GREEN),
        ("Машины", "номер, цвет и марка", icon_car, CYAN),
        ("Животные", "собака, кошка, птица", icon_paw, AMBER),
        ("Оставленные вещи", "рюкзак, сумка, чемодан", icon_box, RED),
        ("Веб-панель", "история, фильтры, фото", icon_web, CYAN),
        ("Telegram + ntfy", "мгновенные уведомления", icon_plane, VIOLET),
    ]

    for f in range(frames):
        t = f / frames
        img = gradient(width, height, (10, 14, 30), (16, 23, 44)).convert("RGBA")
        dot_grid(img, step=30, alpha=20)

        glow = glow_layer((width, height))
        gd = ImageDraw.Draw(glow)
        gd.ellipse((60, 10, 420, 520), fill=(70, 230, 170, 24))
        gd.ellipse((480, -60, 980, 570), fill=(140, 100, 255, 24))
        blurred_glow(img, glow, 70)

        draw = ImageDraw.Draw(img)

        # заголовок
        centered_text(draw, (width // 2, 40), "Что умеет MLCamera", title_font, TEXT)
        # подчёркивание
        line_w = 260
        for x in range((width - line_w) // 2, (width + line_w) // 2):
            ratio = (x - (width - line_w) // 2) / (line_w - 1)
            color = tuple(round(CYAN[i] + (VIOLET[i] - CYAN[i]) * ratio) for i in range(3))
            draw.line((x, 66, x, 69), fill=color)

        # активная карточка
        active = int(t * len(items)) % len(items)
        gap, margin = 22, 40
        card_w = (width - 2 * margin - gap) // 2
        card_h = 124

        for idx, (title, sub, icon, color) in enumerate(items):
            row, col = divmod(idx, 2)
            x0 = margin + col * (card_w + gap)
            y0 = 92 + row * (card_h + gap)
            x1, y1 = x0 + card_w, y0 + card_h
            is_active = idx == active

            if is_active:
                halo = glow_layer((width, height))
                hd = ImageDraw.Draw(halo)
                hd.rounded_rectangle((x0 - 5, y0 - 5, x1 + 5, y1 + 5),
                                     radius=22, fill=color + (54,))
                blurred_glow(img, halo, 14)
                draw = ImageDraw.Draw(img)

            rounded(draw, (x0, y0, x1, y1), 18,
                    fill=(24, 32, 56) if not is_active else (30, 40, 68),
                    outline=color if is_active else (52, 66, 100), width=2)

            # иконка
            ibox = (x0 + 24, y0 + 32, x0 + 82, y0 + 92)
            icon(draw, ibox, color)

            draw.text((x0 + 100, y0 + 32), title, font=card_title, fill=TEXT)
            draw.text((x0 + 100, y0 + 66), sub, font=card_sub, fill=MUTED)

            # индикатор активности
            if is_active:
                pulse = 4 + int(3 * (0.5 + 0.5 * math.sin(t * math.tau * 4)))
                draw.ellipse((x1 - 28 - pulse, y0 + 20 - pulse,
                              x1 - 16 + pulse, y0 + 32 + pulse),
                             outline=color, width=2)

        # бегущая полоса
        sx = int(-160 + (width + 320) * ((t * 1.4) % 1.0))
        streak = glow_layer((width, height))
        sd = ImageDraw.Draw(streak)
        sd.polygon([(sx, 0), (sx + 70, 0), (sx + 170, height), (sx + 100, height)],
                   fill=(255, 255, 255, 10))
        blurred_glow(img, streak, 18)

        images.append(img.convert("RGB"))

    OUT.mkdir(parents=True, exist_ok=True)
    save_gif(images, OUT / "features.gif", duration=95, colors=96)


def save_gif(images: list[Image.Image], path: Path, duration: int,
             colors: int = 128) -> None:
    paletted = []
    for frame in images:
        paletted.append(frame.quantize(colors=colors, method=Image.Quantize.MEDIANCUT))
    paletted[0].save(
        path,
        save_all=True,
        append_images=paletted[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.0f} КБ")


def make_pipeline_svg() -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="430" viewBox="0 0 1000 430" font-family="Segoe UI, Arial, sans-serif">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#090d1c"/>
      <stop offset="1" stop-color="#11172e"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#40d6ff"/>
      <stop offset="1" stop-color="#966eff"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="6" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#5b7099"/>
    </marker>
  </defs>
  <rect width="1000" height="430" rx="24" fill="url(#bg)"/>
  <g stroke="#1d2742" stroke-width="1">
    <path d="M0 90 H1000 M0 180 H1000 M0 270 H1000 M0 360 H1000" opacity=".45"/>
    <path d="M100 0 V430 M350 0 V430 M600 0 V430 M850 0 V430" opacity=".25"/>
  </g>

  <text x="500" y="42" fill="#eef4ff" font-size="24" font-weight="700" text-anchor="middle">Как работает MLCamera</text>
  <rect x="410" y="54" width="180" height="3" rx="1.5" fill="url(#accent)"/>

  <!-- camera -->
  <g>
    <rect x="42" y="120" width="170" height="120" rx="16" fill="#1e2844" stroke="#5c70a0" stroke-width="2"/>
    <rect x="98" y="104" width="58" height="24" rx="6" fill="#263252" stroke="#5c70a0" stroke-width="2"/>
    <circle cx="127" cy="180" r="42" fill="#0a0f1e" stroke="#40d6ff" stroke-width="3"/>
    <circle cx="127" cy="180" r="28" fill="#123a55" stroke="#966eff" stroke-width="2"/>
    <circle cx="115" cy="168" r="9" fill="#d6f8ff"/>
    <circle cx="188" cy="136" r="5" fill="#ff5c6e"/>
    <text x="127" y="272" fill="#9aa9c3" font-size="15" text-anchor="middle">IP-камера / RTSP</text>
  </g>

  <!-- motion -->
  <g>
    <rect x="292" y="132" width="150" height="96" rx="14" fill="#1a2340" stroke="#3c4c74" stroke-width="1.5"/>
    <text x="367" y="168" fill="#eef4ff" font-size="15" font-weight="600" text-anchor="middle">Движение</text>
    <text x="367" y="192" fill="#9aa9c3" font-size="13" text-anchor="middle">только при событиях</text>
    <text x="367" y="210" fill="#40d6ff" font-size="12" text-anchor="middle">экономия CPU</text>
    <text x="367" y="272" fill="#9aa9c3" font-size="15" text-anchor="middle">умный триггер</text>
  </g>

  <!-- yolo -->
  <g>
    <rect x="522" y="112" width="170" height="136" rx="14" fill="#1a2340" stroke="#40d6ff" stroke-width="2"/>
    <text x="607" y="150" fill="#eef4ff" font-size="16" font-weight="700" text-anchor="middle">YOLO</text>
    <rect x="548" y="166" width="60" height="38" rx="4" fill="none" stroke="#42e8a0" stroke-width="2"/>
    <text x="578" y="190" fill="#42e8a0" font-size="11" text-anchor="middle">car</text>
    <rect x="620" y="172" width="46" height="30" rx="4" fill="none" stroke="#ffc457" stroke-width="2"/>
    <text x="643" y="192" fill="#ffc457" font-size="11" text-anchor="middle">человек</text>
    <text x="607" y="224" fill="#42e8a0" font-size="11" text-anchor="middle">+ dog · cat · bird</text>
    <text x="607" y="241" fill="#ff5c6e" font-size="11" text-anchor="middle">+ backpack · suitcase</text>
    <text x="607" y="272" fill="#9aa9c3" font-size="15" text-anchor="middle">детекция объектов</text>
  </g>

  <!-- recognition -->
  <g>
    <rect x="772" y="86" width="186" height="82" rx="14" fill="#1a2340" stroke="#966eff" stroke-width="2"/>
    <text x="865" y="118" fill="#eef4ff" font-size="15" font-weight="700" text-anchor="middle">InsightFace</text>
    <text x="865" y="141" fill="#9aa9c3" font-size="13" text-anchor="middle">лица по именам</text>

    <rect x="772" y="184" width="186" height="82" rx="14" fill="#1a2340" stroke="#40d6ff" stroke-width="2"/>
    <text x="865" y="216" fill="#eef4ff" font-size="15" font-weight="700" text-anchor="middle">OCR + HSV</text>
    <text x="865" y="239" fill="#9aa9c3" font-size="13" text-anchor="middle">номер и цвет</text>
  </g>

  <!-- arrows -->
  <g stroke="#5b7099" stroke-width="2" fill="none" marker-end="url(#arrow)">
    <path d="M212 180 H 288"/>
    <path d="M442 180 H 518"/>
    <path d="M692 180 H 768"/>
  </g>
  <path d="M865 168 V 182" stroke="#5b7099" stroke-width="2" fill="none"/>
  <path d="M865 266 V 300 H 500" stroke="#5b7099" stroke-width="2" fill="none" marker-end="url(#arrow)"/>

  <!-- db + notify -->
  <g>
    <rect x="372" y="300" width="240" height="82" rx="14" fill="#1a2340" stroke="#42e8a0" stroke-width="2"/>
    <text x="492" y="332" fill="#eef4ff" font-size="15" font-weight="700" text-anchor="middle">SQLite + фото</text>
    <text x="492" y="355" fill="#9aa9c3" font-size="13" text-anchor="middle">события · история · веб-панель</text>
  </g>
  <path d="M612 341 H 700" stroke="#5b7099" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
  <g>
    <rect x="704" y="300" width="254" height="82" rx="14" fill="#1a2340" stroke="#966eff" stroke-width="2"/>
    <text x="831" y="332" fill="#eef4ff" font-size="15" font-weight="700" text-anchor="middle">Telegram + ntfy</text>
    <text x="831" y="355" fill="#9aa9c3" font-size="13" text-anchor="middle">фото и текст владельцам</text>
  </g>

  <text x="500" y="415" fill="#5b7099" font-size="12" text-anchor="middle">MLCamera · всё локально, без облачных подписок</text>
</svg>
"""
    (OUT / "pipeline.svg").write_text(svg, encoding="utf-8")
    print(f"{(OUT / 'pipeline.svg').relative_to(ROOT)}: {len(svg)} символов")


def make_models_svg() -> None:
    cards = [
        ("ДЕТЕКЦИЯ", "YOLOv8m / n", "люди · машины · звери", "6–50 МБ", "AGPL-3.0", "#42e8a0", "Y"),
        ("ЛИЦА", "InsightFace", "детекция + эмбеддинги", "158 МБ", "MIT / research", "#966eff", "F"),
        ("МАРКА", "ConvNeXt-Tiny", "49 марок", "106 МБ", "Apache-2.0", "#40d6ff", "B"),
        ("НОМЕР", "YOLOv9-t", "поиск пластины", "7.4 МБ", "MIT", "#ffc457", "D"),
        ("НОМЕР", "MobileViT v2", "чтение символов", "14.5 МБ", "MIT", "#ff5c6e", "O"),
    ]
    margin, gap, card_w = 30, 14, 176
    card_y, card_h = 104, 232
    parts = []
    for i, (role, name, sub, size, lic, color, glyph) in enumerate(cards):
        x = margin + i * (card_w + gap)
        parts.append(f"""
  <g>
    <rect x="{x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="16" fill="#1a2340" stroke="{color}" stroke-opacity=".55" stroke-width="1.5"/>
    <rect x="{x + 18}" y="{card_y}" width="{card_w - 36}" height="3" rx="1.5" fill="{color}"/>
    <text x="{x + 18}" y="{card_y + 30}" fill="{color}" font-size="11" font-weight="700" letter-spacing="1.2">{role}</text>
    <circle cx="{x + card_w // 2}" cy="{card_y + 82}" r="27" fill="{color}" fill-opacity=".12" stroke="{color}" stroke-opacity=".75" stroke-width="1.5"/>
    <text x="{x + card_w // 2}" y="{card_y + 89}" fill="{color}" font-size="20" font-weight="700" text-anchor="middle">{glyph}</text>
    <text x="{x + card_w // 2}" y="{card_y + 140}" fill="#eef4ff" font-size="16" font-weight="700" text-anchor="middle">{name}</text>
    <text x="{x + card_w // 2}" y="{card_y + 162}" fill="#9aa9c3" font-size="11.5" text-anchor="middle">{sub}</text>
    <line x1="{x + 20}" y1="{card_y + 180}" x2="{x + card_w - 20}" y2="{card_y + 180}" stroke="#2a3658" stroke-width="1"/>
    <text x="{x + 18}" y="{card_y + 205}" fill="#dbe6ff" font-size="13" font-weight="600">{size}</text>
    <text x="{x + card_w - 18}" y="{card_y + 205}" fill="#7f8fb0" font-size="10.5" text-anchor="end">{lic}</text>
    <text x="{x + card_w // 2}" y="{card_y + 224}" fill="{color}" font-size="10" text-anchor="middle">готова · без обучения</text>
  </g>""")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="410" viewBox="0 0 1000 410" font-family="Segoe UI, Arial, sans-serif">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#090d1c"/>
      <stop offset="1" stop-color="#11172e"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#40d6ff"/>
      <stop offset="1" stop-color="#966eff"/>
    </linearGradient>
  </defs>
  <rect width="1000" height="410" rx="24" fill="url(#bg)"/>
  <g stroke="#1d2742" stroke-width="1">
    <path d="M0 90 H1000 M0 180 H1000 M0 270 H1000 M0 360 H1000" opacity=".35"/>
    <path d="M100 0 V410 M350 0 V410 M600 0 V410 M850 0 V410" opacity=".18"/>
  </g>
  <text x="500" y="42" fill="#eef4ff" font-size="24" font-weight="700" text-anchor="middle">Нейросети внутри MLCamera</text>
  <rect x="400" y="54" width="200" height="3" rx="1.5" fill="url(#accent)"/>
  <text x="500" y="78" fill="#9aa9c3" font-size="13" text-anchor="middle">5 готовых моделей · животные и вещи — на той же YOLO</text>
{''.join(parts)}
  <text x="500" y="392" fill="#5b7099" font-size="12" text-anchor="middle">YOLO · InsightFace · ConvNeXt · FastALPR · всё локально на CPU</text>
</svg>
"""
    (OUT / "models.svg").write_text(svg, encoding="utf-8")
    print(f"{(OUT / 'models.svg').relative_to(ROOT)}: {len(svg)} символов")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_banner()
    make_features()
    make_pipeline_svg()
    make_models_svg()
