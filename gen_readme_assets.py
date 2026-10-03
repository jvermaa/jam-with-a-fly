import random, os
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FONT = TTFont("tools/BagelFatOne-Regular.ttf")
GS = FONT.getGlyphSet()
CMAP = FONT.getBestCmap()
UPM = FONT["head"].unitsPerEm
HMTX = FONT["hmtx"]

def text_width(s, size):
    return sum(HMTX[CMAP[ord(c)]][0] for c in s) * size / UPM

def text_path(s, x, y, size, anchor="start", fill="#000", extra=""):
    w = text_width(s, size)
    if anchor == "middle": x -= w / 2
    elif anchor == "end": x -= w
    sc = size / UPM
    pen = SVGPathPen(GS)
    cx = x
    for c in s:
        g = CMAP[ord(c)]
        tp = TransformPen(pen, (sc, 0, 0, -sc, cx, y))
        GS[g].draw(tp)
        cx += HMTX[g][0] * sc
    return f'<path d="{pen.getCommands()}" fill="{fill}" {extra}/>'

PAL = {
    "light": dict(bg="#F6F4FF", ink="#1C1846", faint="#DCD8F3", soft="#8C86B8",
                  you="#FF4D8D", fly="#3D5AFE", spark="#FFB800", wing="#3D5AFE"),
    "dark":  dict(bg="#17133A", ink="#F4F2FF", faint="#342F68", soft="#8F89C9",
                  you="#FF5C97", fly="#7184FF", spark="#FFD04D", wing="#7184FF"),
}

def fly(x, y, s, p):
    # tiny cartoon fruit fly, centered at x,y
    return f'''<g transform="translate({x} {y}) scale({s})">
  <ellipse cx="-14" cy="-10" rx="20" ry="9" fill="{p['wing']}" opacity=".28" transform="rotate(-28 -14 -10)"/>
  <ellipse cx="14" cy="-10" rx="20" ry="9" fill="{p['wing']}" opacity=".28" transform="rotate(28 14 -10)"/>
  <path d="M-6 6 L-16 16 M6 6 L16 16 M-7 0 L-19 2 M7 0 L19 2 M-6 -4 L-15 -10 M6 -4 L15 -10" stroke="{p['ink']}" stroke-width="2.4" stroke-linecap="round"/>
  <ellipse cx="0" cy="6" rx="8" ry="13" fill="{p['ink']}"/>
  <circle cx="0" cy="-10" r="8" fill="{p['ink']}"/>
  <circle cx="-5" cy="-12" r="4.2" fill="{p['you']}"/>
  <circle cx="5" cy="-12" r="4.2" fill="{p['you']}"/>
</g>'''

# ---------- HERO ----------
CALL = {0: [0, 6, 8, 11], 1: [4, 12], 2: [0, 2, 4, 6, 8, 10, 12, 14]}
ANS  = {0: [0, 6, 8, 10], 1: [4, 12, 13], 2: [0, 2, 4, 6, 8, 10, 12]}

def col_x(i, x0=72, x1=1208):
    # 32 steps, small gap every 4, big gap between call and answer
    small, big = 14, 56
    gaps = sum(small for k in range(1, 32) if k % 4 == 0 and k != 16) + big
    step = (x1 - x0 - gaps) / 31
    x = x0 + i * step
    for k in range(1, i + 1):
        if k == 16: x += big
        elif k % 4 == 0: x += small
    return x

def hero(mode):
    p = PAL[mode]; W, H = 1280, 560; LOOP = 4.0
    rows_y = [318, 382, 446]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="jam with a fly: you play a beat, a simulated fruit fly plays it back">',
           f'''<style>
.hit{{transform-box:fill-box;transform-origin:center;animation:pop {LOOP}s linear infinite}}
@keyframes pop{{0%{{transform:scale(1.35)}}6%{{transform:scale(1)}}100%{{transform:scale(1)}}}}
.head{{animation:sweep {LOOP}s linear infinite}}
@keyframes sweep{{from{{transform:translateX(0)}}to{{transform:translateX({col_x(31)-col_x(0)}px)}}}}
@media (prefers-reduced-motion:reduce){{.hit,.head{{animation:none}}}}
</style>''',
           f'<rect width="{W}" height="{H}" rx="36" fill="{p["bg"]}"/>',
           text_path("jam with a fly", 72, 168, 112, fill=p["ink"]),
           fly(1150, 112, 1.9, p),
           text_path("you", col_x(0) - 2, 268, 30, fill=p["you"]),
           text_path("fly", col_x(16) - 2, 268, 30, fill=p["fly"])]
    sz = 24
    for r, y in enumerate(rows_y):
        for i in range(32):
            x = col_x(i)
            half = i // 16; st = i % 16
            pat = CALL if half == 0 else ANS
            if st in pat[r]:
                c = p["you"] if half == 0 else p["fly"]
                delay = i / 32 * LOOP
                out.append(f'<rect class="hit" style="animation-delay:{delay - LOOP:.3f}s" x="{x-sz/2:.1f}" y="{y-sz/2}" width="{sz}" height="{sz}" rx="8" fill="{c}"/>')
            else:
                out.append(f'<circle cx="{x:.1f}" cy="{y}" r="4" fill="{p["faint"]}"/>')
    x0 = col_x(0)
    out.append(f'<g class="head"><rect x="{x0-2:.1f}" y="{rows_y[0]-34}" width="4" height="{rows_y[-1]-rows_y[0]+68}" rx="2" fill="{p["spark"]}"/>'
               f'<circle cx="{x0:.1f}" cy="{rows_y[0]-40}" r="6" fill="{p["spark"]}"/></g>')
    out.append('</svg>')
    return "\n".join(out)

# ---------- HOW IT WORKS ----------
def mini_grid(cx, cy, color, p, seed):
    rnd = random.Random(seed); s = []
    pat = [[1,0,0,1,1,0,0,0],[0,0,1,0,0,0,1,0],[1,1,1,1,1,1,1,1]]
    for r in range(3):
        for c in range(8):
            x = cx - 70 + c * 20; y = cy - 22 + r * 22
            if pat[r][c]: s.append(f'<rect x="{x-7}" y="{y-7}" width="14" height="14" rx="4" fill="{color}"/>')
            else: s.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{p["faint"]}"/>')
    return "".join(s)

def how(mode):
    p = PAL[mode]; W, H = 1280, 330
    xs = [150, 395, 640, 885, 1130]; cy = 140
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="your beat goes into the fly\'s ears, through 165,122 simulated neurons, out its legs and wings, and comes back as its beat">',
         f'<rect width="{W}" height="{H}" rx="32" fill="{p["bg"]}"/>']
    for a, b in zip(xs, xs[1:]):
        o.append(f'<path d="M{a+92} {cy} H{b-92}" stroke="{p["soft"]}" stroke-width="3" stroke-dasharray="2 10" stroke-linecap="round"/>')
        o.append(f'<path d="M{b-100} {cy-7} L{b-90} {cy} L{b-100} {cy+7}" stroke="{p["soft"]}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    o.append(mini_grid(xs[0], cy, p["you"], p, 1))
    # ears: sound arcs into a dot
    e = xs[1]
    o.append(f'<circle cx="{e+28}" cy="{cy}" r="12" fill="{p["you"]}"/>')
    for i, r in enumerate([26, 44, 62]):
        o.append(f'<path d="M{e+28-r*0.7:.1f} {cy-r*0.7:.1f} A{r} {r} 0 0 0 {e+28-r*0.7:.1f} {cy+r*0.7:.1f}" stroke="{p["you"]}" stroke-width="5" fill="none" stroke-linecap="round" opacity="{1-i*0.28:.2f}"/>')
    # brain: dot cloud
    rnd = random.Random(7); b = xs[2]
    for k in range(170):
        while True:
            dx, dy = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
            if dx*dx + dy*dy <= 1: break
        x, y = b + dx * 82, cy + dy * 52
        col = p["spark"] if k % 13 == 0 else p["soft"]
        r = 4.2 if col == p["spark"] else 2.6
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}" opacity="{1 if col==p["spark"] else .7}"/>')
    # legs + wings
    l = xs[3]
    o.append(f'<ellipse cx="{l-22}" cy="{cy-26}" rx="34" ry="14" fill="{p["fly"]}" opacity=".3" transform="rotate(-22 {l-22} {cy-26})"/>')
    o.append(f'<ellipse cx="{l+22}" cy="{cy-26}" rx="34" ry="14" fill="{p["fly"]}" opacity=".3" transform="rotate(22 {l+22} {cy-26})"/>')
    for dx, dy in [(-34, 10), (-40, 30), (-30, 52), (34, 10), (40, 30), (30, 52)]:
        sgn = 1 if dx > 0 else -1
        o.append(f'<path d="M{l} {cy+20} L{l+dx} {cy+dy} L{l+dx+sgn*14} {cy+dy+18}" stroke="{p["fly"]}" stroke-width="5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    o.append(f'<ellipse cx="{l}" cy="{cy+18}" rx="12" ry="22" fill="{p["fly"]}"/>')
    o.append(mini_grid(xs[4], cy, p["fly"], p, 2))
    labels = ["your beat", "its ears", "165,122 neurons", "legs + wings", "its beat"]
    for x, t in zip(xs, labels):
        o.append(text_path(t, x, 262, 27, anchor="middle", fill=p["ink"]))
    o.append('</svg>')
    return "\n".join(o)

# ---------- DEMO PLACEHOLDER ----------
def demo(mode):
    p = PAL[mode]; W, H = 1280, 640
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="demo video coming soon">',
        f'<rect width="{W}" height="{H}" rx="36" fill="{p["bg"]}"/>',
        f'<rect x="24" y="24" width="{W-48}" height="{H-48}" rx="26" fill="none" stroke="{p["faint"]}" stroke-width="3" stroke-dasharray="3 14" stroke-linecap="round"/>',
        f'<circle cx="640" cy="290" r="64" fill="{p["you"]}"/>',
        f'<path d="M620 258 L670 290 L620 322 Z" fill="{p["bg"]}" stroke="{p["bg"]}" stroke-width="8" stroke-linejoin="round"/>',
        text_path("demo drops soon", 640, 430, 40, anchor="middle", fill=p["ink"]),
        '</svg>'])

# ---------- FOOTER ----------
def footer(mode):
    p = PAL[mode]; W, H = 1280, 140
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="a fruit fly between your beat and its beat">']
    for i in range(16):
        o.append(f'<circle cx="{150 + i*28}" cy="70" r="{5 if i%4==0 else 3.5}" fill="{p["you"]}" opacity="{0.25 + 0.75*(i/15):.2f}"/>')
        o.append(f'<circle cx="{W-150 - i*28}" cy="70" r="{5 if i%4==0 else 3.5}" fill="{p["fly"]}" opacity="{0.25 + 0.75*(i/15):.2f}"/>')
    o.append(fly(640, 76, 1.5, p))
    o.append('</svg>')
    return "\n".join(o)

os.makedirs("assets", exist_ok=True)
for mode in ["light", "dark"]:
    for name, fn in [("hero", hero), ("how", how), ("demo", demo), ("footer", footer)]:
        with open(f"assets/{name}-{mode}.svg", "w") as f:
            f.write(fn(mode))
print("done")
