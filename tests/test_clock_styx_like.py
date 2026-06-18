"""
Simulates the actual Styx CAPTCHA modal:
  - dark background
  - clock is LEFT-of-center in a wider-than-tall image
  - small numbers, dual-tone tick marks
This is the layout the bot actually sees.
"""
import os, sys, math, cv2, numpy as np
sys.path.insert(0, "/app")
from styx_register import solve_clock

OUT = "/app/tests/synth_styx"
os.makedirs(OUT, exist_ok=True)

def render_styx_clock(hour, minute, W=1024, H=640):
    img = np.full((H, W, 3), 12, np.uint8)  # near-black bg
    # clock dial sits in the LEFT third of the modal (like the screenshot)
    cx, cy = int(W * 0.30), int(H * 0.55)
    r = int(min(H, W) * 0.32)
    cv2.circle(img, (cx, cy), r, (245, 245, 245), -1)
    cv2.circle(img, (cx, cy), r, (200, 200, 200), 2)
    # tick marks
    for i in range(60):
        a = math.radians(i * 6 - 90)
        if i % 5 == 0:
            p1 = (int(cx + math.cos(a) * (r - 22)), int(cy + math.sin(a) * (r - 22)))
            p2 = (int(cx + math.cos(a) * (r - 4)),  int(cy + math.sin(a) * (r - 4)))
            cv2.line(img, p1, p2, (0, 0, 0), 3)
        else:
            p1 = (int(cx + math.cos(a) * (r - 10)), int(cy + math.sin(a) * (r - 10)))
            p2 = (int(cx + math.cos(a) * (r - 4)),  int(cy + math.sin(a) * (r - 4)))
            cv2.line(img, p1, p2, (90, 90, 90), 1)
    # numbers
    for n in range(1, 13):
        a = math.radians(n * 30 - 90)
        tx = int(cx + math.cos(a) * (r - 48))
        ty = int(cy + math.sin(a) * (r - 48))
        cv2.putText(img, str(n), (tx - 12, ty + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    # minute hand (thinner & longer)
    a_min = math.radians(minute * 6 - 90)
    mx = int(cx + math.cos(a_min) * (r * 0.88))
    my = int(cy + math.sin(a_min) * (r * 0.88))
    cv2.line(img, (cx, cy), (mx, my), (0, 0, 0), 4)
    # hour hand (thicker & shorter)
    a_hr = math.radians((hour % 12) * 30 + minute * 0.5 - 90)
    hx = int(cx + math.cos(a_hr) * (r * 0.55))
    hy = int(cy + math.sin(a_hr) * (r * 0.55))
    cv2.line(img, (cx, cy), (hx, hy), (0, 0, 0), 7)
    # hub
    cv2.circle(img, (cx, cy), 7, (0, 0, 0), -1)
    # add a fake "00 : 00" text on the right (modal noise)
    cv2.putText(img, "00 : 00", (int(W * 0.62), int(H * 0.50)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (180, 180, 180), 3)
    return img

cases = [(10, 23), (3, 0), (7, 45), (12, 5), (6, 30), (1, 50), (9, 15),
         (4, 17), (11, 42), (8, 8)]
results = []
for h, m in cases:
    p = os.path.join(OUT, f"styx_{h:02d}_{m:02d}.png")
    cv2.imwrite(p, render_styx_clock(h, m))
    # Crop the clock region the way the bot would (bounding-box of clock element).
    # Here we just pass the full image to test the dial-detector + solver together.
    got = solve_clock(p, debug_dir=OUT)
    want = f"{h:02d}:{m:02d}"
    gh, gm = [int(x) for x in got.split(":")]
    minute_err = min(abs(gm - m), 60 - abs(gm - m))
    hour_ok = (gh == h) or (minute_err <= 2 and (gh == h - 1 or gh == h + 1 or (h == 12 and gh == 1) or (h == 1 and gh == 12)))
    ok = (minute_err <= 2) and hour_ok
    results.append((want, got, ok))
    print(f"  want={want}  got={got}  ok={ok}")

passed = sum(1 for *_, ok in results if ok)
print(f"\n{passed}/{len(results)} cases passed")
sys.exit(0 if passed == len(results) else 1)
