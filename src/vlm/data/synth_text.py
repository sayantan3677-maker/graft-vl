"""Procedural reading data (exact-transcription targets) for the reading curriculum.

Diagnosis (roadmap §16, 2026-10-06): after caption-only S1 alignment the model passes word
*meaning* but not *spelling* (BANANA → "CEREAL"), although a linear probe recovers letters from
the same frozen SigLIP features at 89%. Exact-text targets make the loss reward reading letters.

Four task families, all with verifiable answers:
  word      a word / string / number / date / price / email / code → "What is written in the image?"
  lines     1–8 lines of real sentences → "Transcribe all the text in the image."
  form      key–value layout → "What is the <key>?" → value
  overlay   text on a real photo (sign/label/caption box) → exact text
"""

import random
import string
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/liberation2",
             "/usr/share/fonts/truetype/noto", "/usr/share/fonts/truetype/crosextra", "/usr/share/fonts/truetype/gentium"]

WORD_PROMPTS = ["What is written in the image?", "What does the text in the image say?", "Read the text in this image.",
                "What word or text is shown?", "Transcribe the text in the image exactly."]
LINES_PROMPTS = ["Transcribe all the text in the image.", "Read all the text in this image, line by line.",
                 "What does this text say? Reproduce it exactly.", "Extract the text from this image."]
FORM_KEYS = ["Invoice No", "Order ID", "Date", "Total", "Customer", "Phone", "Email", "Account", "Reference", "Due Date",
             "Amount Paid", "Ship To", "Patient ID", "Policy Number", "Tracking Number", "Subtotal", "Tax", "Name", "City", "Code"]


COVERAGE_TEST = string.ascii_letters + string.digits + "@$£.,:/-+%()"  # ₹/€ are drawn only when covered (see token())


def _render(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
    img = Image.new("L", (48, 48), 0)
    ImageDraw.Draw(img).text((4, 4), ch, fill=255, font=font)
    return img.tobytes()


def _covers_latin(path: str) -> bool:
    """True if every test character renders as a real glyph (not the font's missing-glyph box)."""
    try:
        font = ImageFont.truetype(path, 32)
        missing = _render(font, "\U0010fffd")
        return all((g := _render(font, c)) != missing and any(g) for c in COVERAGE_TEST)
    except OSError:
        return False


_FONT_CACHE: list[str] | None = None


def list_fonts() -> list[str]:
    """Fonts that fully cover Latin letters, digits and common symbols (others render tofu boxes)."""
    global _FONT_CACHE
    if _FONT_CACHE is None:
        candidates = []
        for d in FONT_DIRS:
            p = Path(d)
            if p.exists():
                candidates += [str(f) for f in p.rglob("*.ttf") if not any(x in f.name for x in ("Math", "Symbol", "Emoji"))]
        _FONT_CACHE = sorted(f for f in set(candidates) if _covers_latin(f))
    return _FONT_CACHE


@dataclass
class Sample:
    image: Image.Image
    question: str
    answer: str
    task: str


class SynthText:
    def __init__(self, words: list[str], sentences: list[str], backgrounds: list[Image.Image], seed: int):
        self.rng = random.Random(seed)
        self.words, self.sentences, self.backgrounds = words, sentences, backgrounds
        self.fonts = list_fonts()

    # ------------------------------------------------------------------ text sources

    def token(self) -> str:
        r, rng = self.rng.random(), self.rng
        if r < 0.40:
            w = rng.choice(self.words)
            return rng.choice([w, w.upper(), w.capitalize()])
        if r < 0.55:
            return " ".join(rng.choice(self.words) for _ in range(rng.randint(2, 4)))
        if r < 0.65:
            return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(rng.randint(4, 10)))
        if r < 0.72:
            return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2030)}"
        if r < 0.80:
            return rng.choice(["$", "£", "Rs. ", "EUR ", ""]) + f"{rng.randint(0, 99999):,}.{rng.randint(0, 99):02d}"
        if r < 0.86:
            return f"{rng.choice(self.words).lower()}.{rng.choice(self.words).lower()}@{rng.choice(['gmail.com', 'example.org', 'mail.in'])}"
        if r < 0.92:
            return f"www.{rng.choice(self.words).lower()}{rng.choice(['.com', '.org', '.in', '.io'])}"
        return f"+{rng.randint(1, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"

    # ------------------------------------------------------------------ drawing helpers

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.rng.choice(self.fonts), size)

    def colors(self) -> tuple[tuple, tuple]:
        rng = self.rng
        light = tuple(rng.randint(200, 255) for _ in range(3))
        dark = tuple(rng.randint(0, 70) for _ in range(3))
        return (light, dark) if rng.random() < 0.8 else (dark, light)

    def canvas(self, w: int, h: int, bg: tuple) -> Image.Image:
        img = Image.new("RGB", (w, h), bg)
        if self.rng.random() < 0.25:  # subtle gradient / noise texture
            d = ImageDraw.Draw(img)
            for y in range(0, h, 4):
                shade = int(12 * (y / h) * self.rng.choice([-1, 1]))
                d.rectangle([0, y, w, y + 4], fill=tuple(max(0, min(255, c + shade)) for c in bg))
        return img

    def finish(self, img: Image.Image) -> Image.Image:
        rng = self.rng
        if rng.random() < 0.2:
            img = img.rotate(rng.uniform(-4, 4), expand=True, fillcolor=img.getpixel((0, 0)))
        if rng.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
        return img

    # ------------------------------------------------------------------ tasks

    def word(self) -> Sample:
        rng, text = self.rng, self.token()
        size = rng.choice([18, 24, 32, 44, 60, 80, 110])
        font = self.font(size)
        bg, fg = self.colors()
        tw = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(text, font=font)
        w = int(max(tw + rng.randint(40, 400), rng.randint(256, 1024)))
        h = int(max(size * 2 + rng.randint(20, 300), rng.randint(128, 768)))
        img = self.canvas(w, h, bg)
        ImageDraw.Draw(img).text((rng.randint(10, max(11, int(w - tw - 10))), rng.randint(5, max(6, h - size - 10))), text, fill=fg, font=font)
        return Sample(self.finish(img), rng.choice(WORD_PROMPTS), text, "word")

    def lines(self) -> Sample:
        rng = self.rng
        n = rng.randint(1, 8)
        size = rng.choice([16, 20, 24, 28, 36])
        font = self.font(size)
        width_chars = rng.randint(30, 70)
        lines = []
        for s in rng.sample(self.sentences, min(n, len(self.sentences))):
            s = s[: width_chars * 2]
            while s and len(lines) < n:
                cut = s[:width_chars].rsplit(" ", 1)[0] if len(s) > width_chars else s
                lines.append(cut.strip())
                s = s[len(cut):].strip()
        lines = [x for x in lines if x][:n]
        bg, fg = self.colors()
        draw0 = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        tw = max(draw0.textlength(x, font=font) for x in lines)
        line_h = int(size * rng.uniform(1.25, 1.6))
        w, h = int(tw + rng.randint(40, 200)), int(line_h * len(lines) + rng.randint(30, 200))
        img = self.canvas(w, h, bg)
        d = ImageDraw.Draw(img)
        x0, y0 = rng.randint(15, 40), rng.randint(10, 40)
        for i, x in enumerate(lines):
            d.text((x0, y0 + i * line_h), x, fill=fg, font=font)
        return Sample(self.finish(img), rng.choice(LINES_PROMPTS), "\n".join(lines), "lines")

    def form(self) -> Sample:
        rng = self.rng
        keys = rng.sample(FORM_KEYS, rng.randint(3, 8))
        vals = [self.token() for _ in keys]
        size = rng.choice([16, 20, 24, 28])
        font, bold = self.font(size), self.font(size)
        bg, fg = self.colors()
        line_h = int(size * 1.8)
        w, h = rng.randint(500, 1100), line_h * len(keys) + rng.randint(80, 240)
        img = self.canvas(w, h, bg)
        d = ImageDraw.Draw(img)
        if rng.random() < 0.6:
            d.text((25, 15), rng.choice(["INVOICE", "Receipt", "Order Summary", "Registration Form", "Statement"]), fill=fg, font=self.font(size + 10))
        y0 = 15 + size + 30
        col = rng.randint(180, 280)
        for i, (k, v) in enumerate(zip(keys, vals)):
            d.text((25, y0 + i * line_h), f"{k}:", fill=fg, font=bold)
            d.text((col, y0 + i * line_h), v, fill=fg, font=font)
        j = rng.randrange(len(keys))
        q = rng.choice([f"What is the {keys[j]}?", f"What is written next to '{keys[j]}'?", f"Find the {keys[j]} in this document."])
        return Sample(self.finish(img), q, vals[j], "form")

    def overlay(self) -> Sample:
        rng = self.rng
        bg_img = rng.choice(self.backgrounds).convert("RGB").copy()
        bg_img.thumbnail((rng.randint(512, 1280),) * 2)
        text = self.token()
        size = max(18, int(bg_img.width / rng.uniform(10, 25)))
        font = self.font(size)
        d = ImageDraw.Draw(bg_img)
        tw = d.textlength(text, font=font)
        x = rng.randint(5, max(6, int(bg_img.width - tw - 5)))
        y = rng.randint(5, max(6, bg_img.height - size - 15))
        box_bg, fg = self.colors()
        if rng.random() < 0.8:  # sign / label box for contrast
            d.rectangle([x - 8, y - 6, x + tw + 8, y + size * 1.3], fill=box_bg)
        d.text((x, y), text, fill=fg, font=font)
        q = rng.choice(["What is written on the sign?", "What text appears in the image?", "Read the label in the image."] + WORD_PROMPTS)
        return Sample(self.finish(bg_img), q, text, "overlay")

    def sample(self) -> Sample:
        r = self.rng.random()
        if r < 0.35:
            return self.word()
        if r < 0.65:
            return self.lines()
        if r < 0.85:
            return self.form()
        return self.overlay() if self.backgrounds else self.word()
