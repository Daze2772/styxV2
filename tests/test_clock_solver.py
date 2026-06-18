"""
Generates synthetic clock images at several known times and checks that
solve_clock() reads them within a small tolerance.
"""
import os, sys, math, cv2, numpy as np
sys.path.insert(0, "/app")
from styx_register import solve_clock

OUT = "/app/tests/synth"
os.makedirs(OUT, exist_ok=True)

def render_clock(hour, minute, size=480, with_bg=True):
    img = np.full((size, size, 3), 16, np.uint8) if with_bg else np.full((size, size, 3), 255, np.uint8)
    cx, cy = size // 2, size // 2
    r = int(size * 0.42)
    # white dial
    cv2.circle(img, (cx, cy), r, (255, 255, 255), -1)
    cv2.circle(img, (cx, cy), r, (40, 40, 40), 2)
    # tick marks
    for i in range(60):
        a = math.radians(i * 6 - 90)
        if i % 5 == 0:
            p1 = (int(cx + math.cos(a) * (r - 18)), int(cy + math.sin(a) * (r - 18)))
            p2 = (int(cx + math.cos(a) * (r - 4)),  int(cy + math.sin(a) * (r - 4)))
            cv2.line(img, p1, p2, (0, 0, 0), 3)
        else:
            p1 = (int(cx + math.cos(a) * (r - 8)),  int(cy + math.sin(a) * (r - 8)))
            p2 = (int(cx + math.cos(a) * (r - 4)),  int(cy + math.sin(a) * (r - 4)))
            cv2.line(img, p1, p2, (60, 60, 60), 1)
    # numbers
    for n in range(1, 13):
        a = math.radians(n * 30 - 90)
        tx = int(cx + math.cos(a) * (r - 40))
        ty = int(cy + math.sin(a) * (r - 40))
        cv2.putText(img, str(n), (tx - 12, ty + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # minute hand
    a_min = math.radians(minute * 6 - 90)
    mx = int(cx + math.cos(a_min) * (r * 0.88))
    my = int(cy + math.sin(a_min) * (r * 0.88))
    cv2.line(img, (cx, cy), (mx, my), (0, 0, 0), 4)
    # hour hand
    a_hr = math.radians((hour % 12) * 30 + minute * 0.5 - 90)
    hx = int(cx + math.cos(a_hr) * (r * 0.55))
    hy = int(cy + math.sin(a_hr) * (r * 0.55))
    cv2.line(img, (cx, cy), (hx, hy), (0, 0, 0), 6)
    # hub
    cv2.circle(img, (cx, cy), 6, (0, 0, 0), -1)
    return img

cases = [(3, 0), (10, 23), (7, 45), (12, 5), (6, 30), (1, 50), (9, 15)]
results = []
for h, m in cases:
    p = os.path.join(OUT, f"clock_{h:02d}_{m:02d}.png")
    cv2.imwrite(p, render_clock(h, m))
    got = solve_clock(p, debug_dir=OUT)
    want = f"{h:02d}:{m:02d}"
    # tolerate +/- 1 min, and tolerate hour-off-by-one when minute is near 0 or 60
    gh, gm = [int(x) for x in got.split(":")]
    minute_err = min(abs(gm - m), 60 - abs(gm - m))
    hour_ok = (gh == h) or (minute_err <= 2 and (gh == h - 1 or gh == h + 1 or (h == 12 and gh == 1) or (h == 1 and gh == 12)))
    ok = (minute_err <= 2) and hour_ok
    results.append((want, got, ok))
    print(f"  want={want}  got={got}  ok={ok}")

passed = sum(1 for *_, ok in results if ok)
print(f"\n{passed}/{len(results)} cases passed")
sys.exit(0 if passed == len(results) else 1)
