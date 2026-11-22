"""Render the demo's example images from scratch (no third-party images, so they can ship with the repo/Space).

Usage: python demo/make_examples.py   → demo/examples/{invoice,chart,settings}.png
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "examples"


def font(size: int, bold: bool = False):
    for name in (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def invoice():
    im = Image.new("RGB", (1240, 1600), "white")
    d = ImageDraw.Draw(im)
    d.text((80, 80), "HARBORLINE PRINT SUPPLY", font=font(44, True), fill="#1f3a5f")
    d.text((80, 140), "42 Harbor Lane, Springfield · billing@harborline.example", font=font(22), fill="#555")
    d.text((860, 80), "INVOICE", font=font(48, True), fill="#111")
    for i, (k, v) in enumerate([("Invoice #", "INV-20481"), ("Issue date", "03 Nov 2026"), ("Due date", "03 Dec 2026")]):
        d.text((860, 150 + 36 * i), k, font=font(22), fill="#555")
        d.text((1010, 150 + 36 * i), v, font=font(22, True), fill="#111")
    d.text((80, 300), "Bill to", font=font(24, True), fill="#111")
    d.text((80, 336), "Lakeside Design Studio\n17 Elm Street, Riverton", font=font(22), fill="#333", spacing=8)
    y = 470
    d.rectangle((80, y, 1160, y + 50), fill="#1f3a5f")
    for x, h in [(95, "Description"), (700, "Qty"), (820, "Unit price"), (1010, "Amount")]:
        d.text((x, y + 12), h, font=font(22, True), fill="white")
    rows = [("A3 poster print, matte", 120, 2.50), ("Business cards (box of 500)", 4, 38.00),
            ("Vinyl banner 2 m × 1 m", 2, 64.00), ("Design proof revisions", 3, 25.00)]
    sub = 0.0
    for i, (desc, q, p) in enumerate(rows):
        yy = y + 60 + i * 56
        if i % 2:
            d.rectangle((80, yy - 6, 1160, yy + 44), fill="#f2f5f9")
        amt = q * p
        sub += amt
        for x, t in [(95, desc), (700, str(q)), (820, f"${p:,.2f}"), (1010, f"${amt:,.2f}")]:
            d.text((x, yy + 6), t, font=font(22), fill="#222")
    tax = round(sub * 0.08, 2)
    yy = y + 60 + len(rows) * 56 + 40
    for i, (k, v, b) in enumerate([("Subtotal", sub, False), ("Tax (8%)", tax, False), ("Total due", sub + tax, True)]):
        d.text((820, yy + 44 * i), k, font=font(24, b), fill="#111")
        d.text((1010, yy + 44 * i), f"${v:,.2f}", font=font(24, b), fill="#111")
    d.text((80, 1450), "Payment terms: net 30. Please include the invoice number with your payment.", font=font(20), fill="#555")
    im.save(OUT / "invoice.png")


def chart():
    im = Image.new("RGB", (1200, 800), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 40), "Quarterly revenue, 2026 (USD thousands)", font=font(34, True), fill="#111")
    vals = [("Q1", 412), ("Q2", 486), ("Q3", 531), ("Q4", 468)]
    x0, y0, h, top = 140, 700, 540, 600
    d.line((x0, y0, 1140, y0), fill="#333", width=2)
    for t in range(0, top + 1, 100):
        yy = y0 - h * t / top
        d.line((x0 - 8, yy, 1140, yy), fill="#e5e5e5")
        d.text((60, yy - 12), str(t), font=font(20), fill="#555")
    for i, (q, v) in enumerate(vals):
        bx = x0 + 60 + i * 240
        by = y0 - h * v / top
        d.rectangle((bx, by, bx + 140, y0), fill="#3b6ea8")
        d.text((bx + 40, by - 34), str(v), font=font(24, True), fill="#111")
        d.text((bx + 50, y0 + 14), q, font=font(24), fill="#111")
    im.save(OUT / "chart.png")


def settings():
    im = Image.new("RGB", (1000, 760), "#f4f5f7")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1000, 80), fill="#24292f")
    d.text((32, 24), "Settings › Notifications", font=font(28, True), fill="white")
    items = [("Email notifications", True), ("Push notifications", False), ("Weekly summary", True),
             ("Product updates", False), ("Security alerts", True), ("Marketing emails", False)]
    for i, (name, on) in enumerate(items):
        y = 120 + i * 100
        d.rounded_rectangle((32, y, 968, y + 80), 12, fill="white", outline="#dde1e6")
        d.text((60, y + 24), name, font=font(26), fill="#111")
        d.rounded_rectangle((840, y + 22, 920, y + 58), 18, fill="#2da44e" if on else "#c9ced6")
        cx = 902 if on else 858
        d.ellipse((cx - 16, y + 24, cx + 16, y + 56), fill="white")
        d.text((760, y + 26), "On" if on else "Off", font=font(22), fill="#555")
    im.save(OUT / "settings.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    invoice(), chart(), settings()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
