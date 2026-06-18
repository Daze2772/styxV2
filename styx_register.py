"""
Styx Market registration automation.

Key insight from debugging:
  The previous failure ("Timeout waiting for input[name='password']") was NOT a
  selector problem. After clicking the "Quick Verification" Continue button,
  Cloudflare's deeper bot-check classifies playwright-stealth as a bot and
  silently navigates to /verify which returns an "Attention Required" page
  with ZERO inputs. The form simply never loads.

Mitigations applied:
  1. Use `patchright` (stealth-patched fork) instead of vanilla playwright.
  2. Use `channel='chrome'` (real Chrome, not bundled Chromium) when available.
  3. Use `launch_persistent_context` (real on-disk profile) - far more human.
  4. Headed mode by default (headless is the #1 detection signal).
  5. Explicit Cloudflare-block detection so failures are reported correctly.
  6. JS-based form filling fallback that dispatches proper input/change events
     so any reactive framework (React/Vue) registers the values.
"""
import argparse
import random
import string
import csv
import os
import sys
import math
import tempfile
import time
import cv2
import numpy as np
from loguru import logger

# patchright is a hardened-stealth fork of playwright. We REQUIRE it - falling
# back to vanilla playwright silently makes the script useless against
# Cloudflare's bot detection. Fail loud instead.
try:
    from patchright.sync_api import (
        sync_playwright as _pr_sync_playwright,
        TimeoutError as PlaywrightTimeoutError,
    )
except ImportError as _e:
    sys.stderr.write(
        "\n[FATAL] 'patchright' is not installed.\n"
        "Install it with:\n"
        "    pip install -r requirements.txt\n"
        "    patchright install chrome\n\n"
        "Running vanilla playwright will be instantly detected by Cloudflare.\n"
    )
    raise

# camoufox is an optional alternative engine - a Firefox build with C++-level
# fingerprint injection. Currently considered the strongest open-source option
# against Cloudflare Bot Management.
try:
    from camoufox.sync_api import Camoufox as _Camoufox
    HAS_CAMOUFOX = True
except ImportError:
    HAS_CAMOUFOX = False


# ---------- credential generation ----------------------------------------------
def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_password(length=14):
    upper   = random.choice(string.ascii_uppercase)
    lower   = random.choice(string.ascii_lowercase)
    digit   = random.choice(string.digits)
    special = random.choice("!@#$%^&*")
    rest    = ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%^&*", k=length - 4))
    pwd     = list(upper + lower + digit + special + rest)
    random.shuffle(pwd)
    return ''.join(pwd)


# ---------- clock CAPTCHA solver -----------------------------------------------
def _detect_dial(gray):
    """
    Locate the white clock dial inside the larger CAPTCHA image and return
    (cx, cy, r). Falls back to the image center if Hough fails.
    """
    h, w = gray.shape
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w),
        param1=80, param2=30,
        minRadius=int(min(h, w) * 0.20),
        maxRadius=int(min(h, w) * 0.55),
    )
    if circles is not None:
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)
    return w // 2, h // 2, int(min(h, w) * 0.45)


def _angle_from_12(dx, dy):
    """
    Clock-face angle in degrees [0, 360) where 0 = 12 o'clock position,
    increasing clockwise. dy points down in image coords.
    """
    # atan2(dx, -dy):  dx>0,dy<0 (up-right) => positive small => OK.
    deg = math.degrees(math.atan2(dx, -dy))
    return deg % 360


def solve_clock(image_path, debug_dir=None):
    """
    Robust analog-clock reader for the Styx CAPTCHA.

    Algorithm:
      1. Find the actual dial (Hough circle).
      2. Mask everything outside ~0.92*radius (drops the bezel ticks & numbers).
      3. Threshold (Otsu) to isolate dark hands on white dial.
      4. Find connected components touching the center (real hands always do).
      5. For each hand component, find its farthest point from center
         (= tip), measure its length, and compute its clock angle.
      6. Sort by length: longest = minute hand, shorter = hour hand.
      7. Translate angles -> hh:mm with proper hour-hand correction
         (a real hour hand advances 0.5deg per minute, so we use
          the *minute* reading to refine which hour we're closest to).
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Could not read clock image at {image_path}")
        return "12:00"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1) locate the dial
    cx, cy, r = _detect_dial(gray)
    logger.debug(f"clock dial @ ({cx},{cy}) r={r}")

    # 2) mask outside the dial (drop tick marks + numbers)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), int(r * 0.92), 255, -1)
    dial = cv2.bitwise_and(gray, gray, mask=mask)
    # Pixels outside the mask become 0 (black). Push them to white so they
    # don't get picked up as "hand" during thresholding.
    dial[mask == 0] = 255

    # 3) threshold: hands are dark on a bright dial -> invert.
    _, hands_bin = cv2.threshold(
        dial, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    # Light cleanup only - we MUST NOT close the gap between the two hands.
    kernel = np.ones((3, 3), np.uint8)
    hands_bin = cv2.morphologyEx(hands_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3b) Punch a hole at the hub. This separates the two hands (which
    # otherwise meet at the center pivot) into two distinct components.
    hub_radius = max(8, int(r * 0.10))
    cv2.circle(hands_bin, (cx, cy), hub_radius, 0, -1)

    # 4) connected components - a real hand has its CLOSEST pixel near the hub
    num, labels, stats, _ = cv2.connectedComponentsWithStats(hands_bin, connectivity=8)
    candidates = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 25:
            continue
        ys, xs = np.where(labels == i)
        dists = np.hypot(xs - cx, ys - cy)
        min_d = float(dists.min())
        max_d = float(dists.max())
        # The closest pixel of a hand to the center sits right at the
        # hub-mask boundary (~r*0.10). Numbers/ticks are far from center.
        if min_d > r * 0.22:
            continue
        # Must extend out far enough to be a real hand.
        if max_d < r * 0.30:
            continue
        idx_tip = int(np.argmax(dists))
        tip_x, tip_y = int(xs[idx_tip]), int(ys[idx_tip])
        candidates.append({
            "label": int(i),
            "length": max_d,
            "tip": (tip_x, tip_y),
            "angle": _angle_from_12(tip_x - cx, tip_y - cy),
            "pixels": area,
        })

    if not candidates:
        logger.warning("solve_clock: no qualifying hands found.")
        return "12:00"

    # 5) If two hands overlap visually they become 1 component - split via Hough.
    if len(candidates) == 1:
        logger.debug("solve_clock: only 1 hand component - attempting Hough split.")
        lines = cv2.HoughLinesP(
            hands_bin, rho=1, theta=np.pi / 180,
            threshold=20, minLineLength=int(r * 0.30), maxLineGap=6,
        )
        if lines is not None:
            clusters = {}
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                d1 = math.hypot(x1 - cx, y1 - cy)
                d2 = math.hypot(x2 - cx, y2 - cy)
                near = min(d1, d2)
                if near > r * 0.30:
                    continue
                if d1 > d2:
                    tx, ty, tl = x1, y1, d1
                else:
                    tx, ty, tl = x2, y2, d2
                ang = _angle_from_12(tx - cx, ty - cy)
                key = int(ang // 12)            # 12-deg buckets
                if key not in clusters or tl > clusters[key]["length"]:
                    clusters[key] = {"length": tl, "tip": (tx, ty), "angle": ang}
            if len(clusters) >= 2:
                candidates = sorted(clusters.values(),
                                    key=lambda x: x["length"], reverse=True)[:2]

    # 6) sort: longest is the minute hand
    candidates.sort(key=lambda x: x["length"], reverse=True)
    minute_hand = candidates[0]
    hour_hand = candidates[1] if len(candidates) > 1 else candidates[0]

    logger.debug(f"minute hand: len={minute_hand['length']:.1f} "
                 f"angle={minute_hand['angle']:.1f}")
    logger.debug(f"hour hand:   len={hour_hand['length']:.1f} "
                 f"angle={hour_hand['angle']:.1f}")

    # 7) angles -> time
    minute = int(round(minute_hand["angle"] / 6.0)) % 60   # 6 deg per minute
    # The hour hand advances 0.5deg per minute, so we expect:
    #   hour_angle == integer_hour*30 + minute*0.5
    # Pick the integer_hour [0..11] that best fits the observed minute.
    expected_offset = minute / 60.0
    best_diff = 9999
    best_hour = 0
    for h_int in range(12):
        expected_angle = (h_int + expected_offset) * 30.0
        diff = abs((hour_hand["angle"] - expected_angle + 180) % 360 - 180)
        if diff < best_diff:
            best_diff = diff
            best_hour = h_int
    hour = best_hour
    if hour == 0:
        hour = 12

    naive_hour = int(round(hour_hand["angle"] / 30.0)) % 12
    if naive_hour == 0:
        naive_hour = 12

    logger.info(f"solve_clock -> hour={hour} (naive={naive_hour})  minute={minute}")

    # Optional debug overlay
    if debug_dir:
        try:
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), r, (0, 255, 255), 1)
            for hand, color in ((minute_hand, (0, 255, 0)), (hour_hand, (0, 0, 255))):
                cv2.line(overlay, (cx, cy), hand["tip"], color, 2)
                cv2.circle(overlay, hand["tip"], 4, color, -1)
            cv2.putText(overlay, f"{hour:02d}:{minute:02d}",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            out_path = os.path.join(
                debug_dir,
                f"clock_solved_{os.path.basename(image_path)}",
            )
            cv2.imwrite(out_path, overlay)
            logger.debug(f"debug overlay -> {out_path}")
        except Exception as e:
            logger.debug(f"debug overlay failed: {e}")

    return f"{hour:02d}:{minute:02d}"


# ---------- robust form-fill helpers -------------------------------------------
JS_SET_NATIVE_VALUE = r"""
([selector, value]) => {
    // Resolve - try multiple strategies including shadow DOM piercing.
    function deepQuery(root, sel) {
        const found = root.querySelector(sel);
        if (found) return found;
        const all = root.querySelectorAll('*');
        for (const el of all) {
            if (el.shadowRoot) {
                const r = deepQuery(el.shadowRoot, sel);
                if (r) return r;
            }
        }
        return null;
    }
    const el = deepQuery(document, selector);
    if (!el) return { ok: false, reason: 'not_found' };

    // Use the native setter so frameworks (React/Vue) detect the change.
    const proto = el.tagName === 'TEXTAREA'
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, value);

    el.dispatchEvent(new Event('input',  { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    el.dispatchEvent(new Event('blur',   { bubbles: true }));
    return { ok: true, name: el.name, type: el.type, cls: el.className };
}
"""


def js_fill(page, selector, value):
    """Fill via native JS setter + proper events so reactive frameworks update."""
    return page.evaluate(JS_SET_NATIVE_VALUE, [selector, value])


def smart_fill(page, candidate_selectors, value, label=""):
    """
    Try several strategies in order until one succeeds:
      1) Playwright locator.fill on each candidate
      2) JS native-setter fill (handles framework state + shadow DOM)
    """
    # Strategy 1 - Playwright fill
    for sel in candidate_selectors:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="attached", timeout=3000)
            loc.scroll_into_view_if_needed(timeout=2000)
            loc.fill(value, timeout=3000)
            actual = loc.input_value(timeout=2000)
            if actual == value:
                logger.info(f"  [{label}] filled via PW selector: {sel}")
                return True
        except Exception as e:
            logger.debug(f"  [{label}] PW fill failed on {sel}: {e}")

    # Strategy 2 - JS native setter
    for sel in candidate_selectors:
        try:
            res = js_fill(page, sel, value)
            if res and res.get("ok"):
                logger.info(f"  [{label}] filled via JS native setter: {sel} -> {res}")
                return True
        except Exception as e:
            logger.debug(f"  [{label}] JS fill failed on {sel}: {e}")

    logger.error(f"  [{label}] ALL fill strategies failed.")
    return False


# ---------- Cloudflare block detection -----------------------------------------
def is_cloudflare_blocked(page):
    """Return True if the current page is the Cloudflare 'Sorry, you have been blocked' page."""
    try:
        title = page.title() or ""
        if "Attention Required" in title or "Cloudflare" in title:
            html = page.content().lower()
            if "sorry, you have been blocked" in html or "cf-error-details" in html:
                return True
        # Also check URL pattern
        if "/verify" in page.url and "cloudflare" in (page.title() or "").lower():
            return True
    except Exception:
        pass
    return False


# ---------- session reset ------------------------------------------------------
def _reset_site_session(page, url):
    """
    Clear cookies + localStorage/sessionStorage for the target site so each run
    looks like a fresh first-time visitor to Styx, while keeping the underlying
    Chrome profile (history, plugins, accumulated TLS reputation) intact for
    Cloudflare trust.
    """
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    ctx = page.context

    # 1) Cookies - drop anything whose domain matches our target host.
    try:
        all_cookies = ctx.cookies()
        keep, drop = [], []
        for c in all_cookies:
            dom = (c.get("domain") or "").lstrip(".")
            if dom and (dom == host or host.endswith("." + dom) or dom.endswith(host)):
                drop.append(c)
            else:
                keep.append(c)
        ctx.clear_cookies()
        if keep:
            ctx.add_cookies(keep)
        logger.info(f"Session reset: dropped {len(drop)} cookie(s) for {host}, "
                    f"kept {len(keep)}.")
    except Exception as e:
        logger.debug(f"cookie reset failed: {e}")

    # 2) localStorage / sessionStorage - needs an active page on the origin.
    # Best-effort: visit the site root first, clear, then continue.
    try:
        origin = f"{urlparse(url).scheme}://{host}"
        page.goto(origin, wait_until="domcontentloaded", timeout=30000)
        page.evaluate("""() => {
            try { localStorage.clear(); } catch(e) {}
            try { sessionStorage.clear(); } catch(e) {}
            try { indexedDB.databases && indexedDB.databases()
                  .then(dbs => dbs.forEach(db => indexedDB.deleteDatabase(db.name))); } catch(e) {}
        }""")
        logger.info(f"Cleared localStorage/sessionStorage for {origin}.")
    except Exception as e:
        logger.debug(f"storage reset failed (non-fatal): {e}")


# ---------- post-registration: wallet top-up ----------------------------------
def do_topup(page, amount, currency_label="BNB (BEP20)",
             base_url="https://styxmarket.si", debug_dir=None):
    """
    Navigate to /wallet/top-up/, click the requested crypto tile, fill the
    payment amount, click TOP UP BALANCE. Does NOT close the page.

    Args:
        page: Playwright Page (already logged in after registration).
        amount: numeric payment amount (string or int).
        currency_label: visible label on the crypto tile, e.g. "BNB (BEP20)".
    """
    topup_url = f"{base_url.rstrip('/')}/wallet/top-up/"
    logger.info(f"Navigating to top-up page: {topup_url}")
    page.goto(topup_url, wait_until="domcontentloaded", timeout=45000)
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass
    time.sleep(random.uniform(1.5, 2.5))

    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "10_topup_landing.png"),
                        full_page=True)

    # 1) Click the crypto tile (e.g. "BNB (BEP20)")
    logger.info(f"Selecting crypto: {currency_label}")
    clicked = False
    # Try a wide variety of selectors, role-based first (most robust).
    strategies = [
        lambda: page.get_by_role("button", name=currency_label, exact=True).first.click(timeout=4000),
        lambda: page.get_by_text(currency_label, exact=True).first.click(timeout=4000),
        lambda: page.locator(f"button:has-text('{currency_label}')").first.click(timeout=4000),
        lambda: page.locator(f"[class*='currency']:has-text('{currency_label}')").first.click(timeout=4000),
        lambda: page.locator(f"div:has-text('{currency_label}')").first.click(timeout=4000),
        lambda: page.locator(f"a:has-text('{currency_label}')").first.click(timeout=4000),
    ]
    for i, strat in enumerate(strategies, 1):
        try:
            strat()
            clicked = True
            logger.info(f"  -> clicked '{currency_label}' (strategy {i})")
            break
        except Exception as e:
            logger.debug(f"  topup tile strategy {i} failed: {e}")
    if not clicked:
        logger.error(f"Could not click currency tile '{currency_label}'.")
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "11_topup_no_tile.png"),
                            full_page=True)
        return False

    time.sleep(random.uniform(0.8, 1.5))
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "12_topup_tile_selected.png"),
                        full_page=True)

    # 2) Fill the Payment Amount input
    logger.info(f"Entering payment amount: {amount}")
    amount_str = str(amount)
    ok_amt = smart_fill(page,
        ["input[name='amount']",
         "input[name='payment_amount']",
         "input[type='number']",
         "[class*='payment'] input",
         "[class*='amount'] input",
         "input[placeholder*='mount']",
         # last-resort: the only visible input on the top-up page
         "input.input__input:visible",
         "input:visible"],
        amount_str, label="payment_amount")
    if not ok_amt:
        logger.error("Could not fill payment amount input.")
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "13_topup_no_amount.png"),
                            full_page=True)
        return False

    time.sleep(random.uniform(0.5, 1.0))
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "14_topup_amount_filled.png"),
                        full_page=True)

    # 3) Click TOP UP BALANCE
    logger.info("Clicking TOP UP BALANCE...")
    btn_strategies = [
        lambda: page.get_by_role("button", name="TOP UP BALANCE").first.click(timeout=5000),
        lambda: page.get_by_role("button", name="Top up balance").first.click(timeout=5000),
        lambda: page.locator("button:has-text('TOP UP BALANCE')").first.click(timeout=5000),
        lambda: page.locator("button:has-text('Top up balance')").first.click(timeout=5000),
        lambda: page.locator("button:has-text('Top up')").last.click(timeout=5000),
        lambda: page.locator("button[type='submit']").last.click(timeout=5000),
    ]
    btn_clicked = False
    for i, strat in enumerate(btn_strategies, 1):
        try:
            strat()
            btn_clicked = True
            logger.info(f"  -> clicked TOP UP BALANCE (strategy {i})")
            break
        except Exception as e:
            logger.debug(f"  topup-button strategy {i} failed: {e}")
    if not btn_clicked:
        logger.error("Could not click TOP UP BALANCE.")
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "15_topup_no_button.png"),
                            full_page=True)
        return False

    # Give the next page (deposit address / QR) a moment to render
    time.sleep(3)
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "16_topup_submitted.png"),
                        full_page=True)
    logger.success(f"Top-up submitted: {currency_label} amount={amount}")
    return True


# ---------- registration flow --------------------------------------------------
def process_registration(page, url, max_captcha_retries=3, debug_dir=None,
                         topup_amount=0, topup_currency="BNB (BEP20)",
                         keep_open=False, keep_session=False):
    if not keep_session:
        _reset_site_session(page, url)

    logger.info(f"Navigating to {url}")
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    time.sleep(2)  # let any client-side JS settle

    # ---- fingerprint self-check ---------------------------------------------
    # If any of these come back "wrong", Cloudflare WILL flag us.
    try:
        fp = page.evaluate("""() => ({
            webdriver: navigator.webdriver,
            userAgent: navigator.userAgent,
            platform:  navigator.platform,
            languages: navigator.languages,
            vendor:    navigator.vendor,
            plugins:   navigator.plugins.length,
            chrome:    typeof window.chrome,
            permissions: typeof navigator.permissions,
            webglVendor: (() => {
                try {
                    const c = document.createElement('canvas');
                    const gl = c.getContext('webgl');
                    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
                    return gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL);
                } catch(e) { return 'err:' + e.message; }
            })(),
        })""")
        logger.info("--- browser fingerprint ---")
        for k, v in fp.items():
            tag = ""
            if k == "webdriver" and v:
                tag = "  <-- !! LEAK: should be false/undefined"
            if k == "plugins" and v == 0:
                tag = "  <-- !! LEAK: real Chrome has plugins"
            if k == "chrome" and v == "undefined":
                tag = "  <-- !! LEAK: real Chrome has window.chrome"
            logger.info(f"  {k}: {v}{tag}")
        logger.info("---------------------------")
    except Exception as e:
        logger.warning(f"fingerprint check failed: {e}")

    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "01_landing.png"))

    # 1) Quick Verification ----------------------------------------------------
    try:
        logger.info("Looking for Quick Verification 'Continue' button...")
        cont = page.locator("button:has-text('Continue'), .continue-btn, #continue").first
        if cont.is_visible(timeout=8000):
            # Small human-like delay before clicking
            time.sleep(random.uniform(1.2, 2.5))
            cont.click()
            logger.info("Clicked 'Continue'.")
            # Loading-bar animation is ~5-6 seconds, then navigation
            time.sleep(7)
    except PlaywrightTimeoutError:
        logger.debug("No Quick Verification step found, proceeding.")

    # Wait for navigation/render to settle
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass

    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "02_post_verify.png"))

    # 2) Detect Cloudflare block ----------------------------------------------
    if is_cloudflare_blocked(page):
        logger.error("=" * 60)
        logger.error("CLOUDFLARE HARD-BLOCK DETECTED.")
        logger.error(f"Current URL: {page.url}")
        logger.error("The form was never rendered. This is NOT a selector issue.")
        logger.error("Try: (a) run headed, (b) channel='chrome', (c) different IP/proxy,")
        logger.error("     (d) clear /tmp/styx_profile and let a real user solve")
        logger.error("     the challenge once so the cookie is persisted.")
        logger.error("=" * 60)
        if debug_dir:
            with open(os.path.join(debug_dir, "blocked.html"), "w") as f:
                f.write(page.content())
        return None

    # 3) Registration form ----------------------------------------------------
    logger.info("Waiting for registration form to render...")
    try:
        page.wait_for_selector("input.input__input, input[name='password']",
                               state="attached", timeout=20000)
    except PlaywrightTimeoutError:
        logger.error("Form never appeared. Dumping page state for inspection.")
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "03_no_form.png"))
            with open(os.path.join(debug_dir, "no_form.html"), "w") as f:
                f.write(page.content())
        return None

    # Confirm count of inputs we can actually see
    try:
        info = page.evaluate("""() => {
            const inputs = Array.from(document.querySelectorAll('input.input__input, input'));
            return inputs.map(el => ({
                name: el.name, type: el.type, cls: el.className,
                visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length)
            }));
        }""")
        logger.info(f"Detected {len(info)} input(s) on the page:")
        for i in info:
            logger.info(f"  - {i}")
    except Exception:
        pass

    # Username: must start with a letter (some sites reject leading digits),
    # then 9-13 letters/digits. No underscores/dashes - Styx rejects them.
    username = (random.choice(string.ascii_lowercase) +
                ''.join(random.choices(string.ascii_lowercase + string.digits,
                                       k=random.randint(9, 13))))
    password = generate_password()
    secret   = generate_random_string(12)
    logger.info(f"Generated credentials -> user={username}")

    # Multi-strategy fills. Order: most specific to most generic.
    ok_u = smart_fill(page,
        ["input[name='username']",
         "form input.input__input:nth-of-type(1)",
         "div.sign-up-form__body div:nth-child(1) input.input__input",
         "input.input__input >> nth=0"],
        username, label="username")

    ok_p = smart_fill(page,
        ["input[name='password']",
         "input[type='password']",
         "div.sign-up-form__body div:nth-child(2) input.input__input",
         "input.input__input >> nth=1"],
        password, label="password")

    ok_s = smart_fill(page,
        ["input[name='secret_code']", "input[name='secret']", "input[name='pin']",
         "div.sign-up-form__body div:nth-child(3) input.input__input",
         "input.input__input >> nth=2"],
        secret, label="secret_code")

    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "04_filled.png"))

    if not (ok_u and ok_p and ok_s):
        logger.error(f"Failed to fill all fields (u={ok_u} p={ok_p} s={ok_s})")
        return None

    # Submit
    try:
        time.sleep(random.uniform(0.5, 1.2))
        page.locator("button:has-text('Sign up'), button[type='submit'], "
                     ".sign-up__form button").first.click(timeout=8000)
        logger.info("Clicked Sign up.")
    except Exception as e:
        logger.error(f"Failed to click Sign up: {e}")
        return None

    # 4) Clock CAPTCHA --------------------------------------------------------
    # The modal animates in - give it a beat.
    time.sleep(1.5)

    # Wait for ANY reliable CAPTCHA indicator: a 00:00-placeholder input,
    # an "OK" button, or the prompt text. Whichever appears first wins.
    captcha_present = False
    try:
        page.wait_for_function(
            """() => {
                const hasInput = !!document.querySelector(
                    "input[placeholder='00:00'], input[placeholder*=':']");
                const hasOK = Array.from(document.querySelectorAll('button'))
                    .some(b => /^\\s*OK\\s*$/i.test(b.textContent || ''));
                const txt = document.body.innerText || '';
                const hasTxt = /not a robot/i.test(txt)
                            || /time shown in the picture/i.test(txt);
                return hasInput || hasOK || hasTxt;
            }""",
            timeout=30000,
        )
        captcha_present = True
        logger.info("CAPTCHA modal detected.")
    except PlaywrightTimeoutError:
        logger.info("No CAPTCHA modal within 30s.")

    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "05_after_signup.png"),
                        full_page=True)

    # If no CAPTCHA modal showed up, check for form validation errors
    # (these would explain why the form didn't submit).
    if not captcha_present:
        # Separate locators per engine - DO NOT mix CSS and text= in one selector.
        for sel in (".error-message", ".invalid-feedback", ".input__error",
                    ".form-error", "text=This field is required",
                    "text=already exists", "text=already taken"):
            try:
                loc = page.locator(sel).first
                if loc.count() and loc.is_visible():
                    logger.error(f"Form validation error ({sel}): "
                                 f"{loc.inner_text()[:200]}")
                    return None
            except Exception:
                continue
        logger.info("No CAPTCHA and no validation error - account may already "
                    "have been created (page navigated).")
        # Some sites redirect to dashboard on success without a CAPTCHA.
        # Treat that as success.
        logger.success(f"Registered: {username}")
        return {"url": url, "username": username, "password": password, "secret": secret}

    # Solve loop
    success = False
    for attempt in range(1, max_captcha_retries + 1):
        logger.info(f"Solving clock CAPTCHA (attempt {attempt}/{max_captcha_retries})...")
        try:
            # Find the clock image. The modal contains an <img>, <canvas>, or
            # background-image of the analog clock - try multiple strategies.
            clock_el = None
            for sel in (
                "img[src*='clock']", "img[src*='captcha']",
                ".captcha img", ".captcha canvas", ".captcha svg",
                "[class*='captcha'] img", "[class*='captcha'] canvas",
                "[class*='clock'] img", "[class*='clock'] canvas",
                "canvas", "svg",
            ):
                try:
                    cand = page.locator(sel).first
                    if cand.count() and cand.is_visible(timeout=1000):
                        bbox = cand.bounding_box()
                        # Only accept big square-ish elements (the clock dial)
                        if bbox and bbox["width"] >= 120 and bbox["height"] >= 120 \
                                and 0.6 <= bbox["width"] / bbox["height"] <= 1.7:
                            clock_el = cand
                            logger.debug(f"clock element via: {sel}  bbox={bbox}")
                            break
                except Exception:
                    continue

            clock_path = os.path.join(debug_dir or ".", f"clock_{attempt}.png")
            if clock_el is not None:
                clock_el.screenshot(path=clock_path)
            else:
                # Fallback: crop the left half of the modal from a full-page shot.
                logger.warning("Could not isolate clock element - cropping page.")
                full_path = os.path.join(debug_dir or ".",
                                         f"clock_fallback_{attempt}_full.png")
                page.screenshot(path=full_path, full_page=False)
                img = cv2.imread(full_path)
                if img is None:
                    logger.error("Fallback page screenshot unreadable.")
                    break
                h, w = img.shape[:2]
                # Heuristic: clock is roughly centered vertically, left of center
                crop = img[int(h * 0.20):int(h * 0.80),
                           int(w * 0.20):int(w * 0.55)]
                cv2.imwrite(clock_path, crop)
            logger.info(f"Clock screenshot -> {clock_path}")

            time_str = solve_clock(clock_path, debug_dir=debug_dir)
            logger.info(f"Solved time: {time_str}")

            ok_t = smart_fill(page,
                ["input[placeholder='00:00']",
                 "input[placeholder*=':']",
                 "input[name='captcha_time']",
                 "input[name='time']",
                 ".captcha input",
                 "[class*='captcha'] input"],
                time_str, label="captcha_time")
            if not ok_t:
                logger.error("Could not fill captcha time field.")
                break

            # Click OK (case-insensitive)
            try:
                page.get_by_role("button", name="OK").first.click(timeout=5000)
            except Exception:
                page.locator("button:has-text('OK'), button:has-text('Ok')").first.click(timeout=5000)
            page.wait_for_timeout(2500)

            # Check for an error indicator (separate selectors per engine).
            wrong = False
            for sel in ("text=Incorrect", "text=incorrect", "text=Wrong time",
                        "text=Try again", ".captcha-error", ".error"):
                try:
                    loc = page.locator(sel).first
                    if loc.count() and loc.is_visible():
                        wrong = True
                        break
                except Exception:
                    continue
            if wrong:
                logger.warning(f"Clock time '{time_str}' rejected, retrying...")
                continue

            logger.info("CAPTCHA accepted.")
            success = True
            break

        except PlaywrightTimeoutError as e:
            logger.warning(f"CAPTCHA attempt {attempt} timed out: {e}")
            continue
        except Exception as e:
            logger.error(f"CAPTCHA attempt {attempt} error: {e}")
            continue

    if success:
        logger.success(f"Registered: {username}")
        result = {"url": url, "username": username,
                  "password": password, "secret": secret}

        # After successful registration the user is auto-logged-in. Optionally
        # navigate to the wallet top-up page and submit a payment request.
        if topup_amount and float(topup_amount) > 0:
            logger.info("Waiting a few seconds before navigating to top-up...")
            time.sleep(random.uniform(3.0, 5.0))
            try:
                # base_url derived from registration URL so it works on
                # mirrors / staging hosts too.
                from urllib.parse import urlparse
                _u = urlparse(url)
                base = f"{_u.scheme}://{_u.netloc}"
                do_topup(page, amount=topup_amount, base_url=base,
                         currency_label=topup_currency,
                         debug_dir=debug_dir)
            except Exception as e:
                logger.error(f"Top-up flow failed: {e}")

        # Keep the browser session open until the user closes the tab.
        if keep_open:
            logger.info("=" * 64)
            logger.info("Session is now YOURS. Close the browser window when done.")
            logger.info("(The script will exit automatically when you close the tab.)")
            logger.info("=" * 64)
            try:
                page.wait_for_event("close", timeout=0)
            except Exception as e:
                logger.debug(f"wait_for_event('close') ended: {e}")
            logger.info("Browser tab closed by user - exiting.")
            # Persist the result BEFORE we exit, so users don't lose creds.
            try:
                _persist_results([result], "accounts.csv")
            except Exception as e:
                logger.debug(f"persist on exit failed: {e}")
            sys.exit(0)

        return result
    logger.error("Registration flow failed at CAPTCHA stage.")
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "99_captcha_failed.png"),
                        full_page=True)
    return None


# ---------- engine launchers ---------------------------------------------------
def _parse_proxy(proxy_url):
    """
    Convert a proxy URL like 'http://user:pass@host:port' into Playwright's
    proxy dict {server, username, password}.
    """
    if not proxy_url:
        return None
    from urllib.parse import urlparse
    u = urlparse(proxy_url)
    if not u.scheme or not u.hostname:
        logger.error(f"Invalid proxy URL: {proxy_url}")
        return None
    server = f"{u.scheme}://{u.hostname}"
    if u.port:
        server += f":{u.port}"
    out = {"server": server}
    if u.username:
        out["username"] = u.username
    if u.password:
        out["password"] = u.password
    return out


def _warn_if_datacenter_ip():
    """
    Best-effort detection of well-known cloud/datacenter IP ranges.
    Cloudflare assigns very low trust to Azure/AWS/GCP egress IPs - even a
    perfect fingerprint may still be blocked.
    """
    try:
        import urllib.request
        import json
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        ip   = data.get("ip", "?")
        org  = (data.get("org") or "").lower()
        host = (data.get("hostname") or "").lower()
        flag_keywords = ("microsoft", "azure", "amazon", "aws", "google",
                         "gcp", "digitalocean", "linode", "ovh", "hetzner",
                         "vultr", "oracle")
        is_dc = any(k in org for k in flag_keywords) or \
                any(k in host for k in flag_keywords)
        if is_dc:
            logger.warning("=" * 64)
            logger.warning(f"DATACENTER IP DETECTED: {ip}  ({data.get('org')})")
            logger.warning("Cloudflare assigns near-zero trust to datacenter IPs.")
            logger.warning("Even a perfect browser fingerprint may be blocked.")
            logger.warning("Strongly recommended: pass --proxy http://user:pass@host:port")
            logger.warning("using a *residential* proxy (Bright Data, IPRoyal, etc.).")
            logger.warning("=" * 64)
        else:
            logger.info(f"egress IP looks residential/clean: {ip} ({data.get('org')})")
    except Exception as e:
        logger.debug(f"IP-trust pre-flight check failed (non-fatal): {e}")


def _run_with_patchright(args, urls, debug_dir, results):
    """Patchright (Chrome + stealth patches) launcher."""
    # CRITICAL: with channel="chrome" we MUST NOT override user_agent / viewport /
    # timezone_id / locale, because real Chrome already has perfectly consistent
    # values for all of those. Overriding even one creates a mismatch between
    # the JS-reported value and the build/OS/binary, which Cloudflare's worker
    # cross-checks. Let Chrome be itself.
    launch_kwargs = dict(
        user_data_dir=args.profile,
        headless=args.headless,
        no_viewport=True,
        viewport=None,
        ignore_default_args=["--enable-automation"],
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            "--no-default-browser-check",
            "--no-first-run",
        ],
    )
    if args.channel and args.channel != "chromium":
        launch_kwargs["channel"] = args.channel

    proxy = _parse_proxy(args.proxy)
    if proxy:
        launch_kwargs["proxy"] = proxy

    with _pr_sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(**launch_kwargs)
        except Exception as e:
            logger.warning(f"channel='{args.channel}' failed ({e}); "
                           f"falling back to bundled chromium.")
            launch_kwargs.pop("channel", None)
            context = p.chromium.launch_persistent_context(**launch_kwargs)

        for url in urls:
            for i in range(args.count):
                if args.count > 1:
                    logger.info(f"=== {url} attempt {i + 1}/{args.count} ===")
                page = context.new_page()
                try:
                    result = process_registration(
                        page, url, debug_dir=debug_dir,
                        topup_amount=args.topup,
                        topup_currency=args.topup_currency,
                        keep_open=args.keep_open,
                        keep_session=args.keep_session,
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Critical error on {url}: {e}")
                finally:
                    if not args.keep_open:
                        try:
                            page.close()
                        except Exception:
                            pass
                # Human-like pause between registrations
                if args.count > 1 and i < args.count - 1:
                    time.sleep(random.uniform(4.0, 8.0))

        if not args.keep_open:
            context.close()
        else:
            try:
                context.close()
            except Exception:
                pass


def _run_with_camoufox(args, urls, debug_dir, results):
    """Camoufox (Firefox + C++ fingerprint injection) launcher."""
    if not HAS_CAMOUFOX:
        logger.error("--engine=camoufox requested but 'camoufox' is not installed.")
        logger.error("Install with:  pip install camoufox[geoip]  &&  camoufox fetch")
        sys.exit(1)

    proxy = _parse_proxy(args.proxy)
    cf_kwargs = dict(
        headless=args.headless,
        humanize=True,           # human-like cursor movement
        os=("windows", "macos"), # rotate among realistic OS fingerprints
        persistent_context=True,
        user_data_dir=args.profile,
    )
    if proxy:
        cf_kwargs["proxy"] = proxy
        cf_kwargs["geoip"] = True    # auto-match locale/timezone to proxy IP

    with _Camoufox(**cf_kwargs) as browser:
        # When persistent_context=True, the context lives on `browser` directly.
        context = browser
        for url in urls:
            for i in range(args.count):
                if args.count > 1:
                    logger.info(f"=== {url} attempt {i + 1}/{args.count} ===")
                page = context.new_page()
                try:
                    result = process_registration(
                        page, url, debug_dir=debug_dir,
                        topup_amount=args.topup,
                        topup_currency=args.topup_currency,
                        keep_open=args.keep_open,
                        keep_session=args.keep_session,
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Critical error on {url}: {e}")
                finally:
                    if not args.keep_open:
                        try:
                            page.close()
                        except Exception:
                            pass
                if args.count > 1 and i < args.count - 1:
                    time.sleep(random.uniform(4.0, 8.0))


# ---------- main ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Automate Styx Market registration.")
    parser.add_argument("--url",      help="Single registration URL.")
    parser.add_argument("--file",     help="Text file of URLs (one per line).")
    parser.add_argument("--output",   default="accounts.csv", help="CSV output path.")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless (NOT recommended - easily detected).")
    parser.add_argument("--channel",  default="chrome",
                        help="Browser channel: 'chrome' (recommended), "
                             "'chrome-beta', 'msedge' or 'chromium'.")
    parser.add_argument("--profile",  default=os.path.join(tempfile.gettempdir(), "styx_profile"),
                        help="Persistent user-data dir (keeps cookies between runs). "
                             "Defaults to OS temp dir for cross-platform support.")
    parser.add_argument("--fresh-profile", action="store_true",
                        help="Wipe the profile dir before starting (use if CF "
                             "has already flagged a previous session).")
    parser.add_argument("--use-real-chrome-profile", action="store_true",
                        help="Use your REAL system Chrome profile (auto-detected). "
                             "Cloudflare cannot distinguish this from you browsing "
                             "manually. WARNING: close your Chrome before running.")
    parser.add_argument("--debug",    action="store_true",
                        help="Save screenshots + HTML dumps to ./debug_out/")
    parser.add_argument("--proxy",    default=os.environ.get("STYX_PROXY"),
                        help="Proxy URL, e.g. http://user:pass@host:port or "
                             "socks5://host:port. Strongly recommended on "
                             "datacenter IPs (Azure/AWS/GCP). Can also be set "
                             "via the STYX_PROXY env var.")
    parser.add_argument("--engine",   default="patchright",
                        choices=["patchright", "camoufox"],
                        help="Browser engine. 'patchright' = Chrome with stealth "
                             "patches (default, fast). 'camoufox' = Firefox with "
                             "C++-level fingerprint injection (slower but the "
                             "strongest open-source Cloudflare bypass).")
    parser.add_argument("--count",    type=int, default=1,
                        help="How many accounts to register per URL.")
    parser.add_argument("--topup",    type=float, default=0,
                        help="If > 0, after registration: navigate to the wallet "
                             "top-up page, select the crypto from --topup-currency, "
                             "enter this amount and click TOP UP BALANCE. "
                             "Default: 0 (skip top-up).")
    parser.add_argument("--topup-currency", default="BNB (BEP20)",
                        help="Crypto tile label to click on the top-up page. "
                             "Examples: 'BNB (BEP20)', 'USDT (TRC20)', 'Bitcoin', "
                             "'Ethereum (ERC20)'.")
    parser.add_argument("--keep-open", action="store_true",
                        help="After all actions complete, leave the browser open "
                             "until you manually close the tab. The script blocks "
                             "until the page is closed.")
    parser.add_argument("--keep-session", action="store_true",
                        help="Skip the per-run cookie/localStorage reset for the "
                             "target site. By default the script clears Styx "
                             "cookies + storage on every run so each registration "
                             "sees a fresh visitor (the Chrome profile itself is "
                             "still reused to keep Cloudflare trust). Use this "
                             "flag if you instead want to resume the previous "
                             "logged-in session.")
    args = parser.parse_args()

    # Auto-detect real Chrome profile path if requested
    if args.use_real_chrome_profile:
        candidates = [
            os.path.expanduser("~/Library/Application Support/Google/Chrome"),  # macOS
            os.path.expanduser("~/.config/google-chrome"),                       # Linux
            os.path.expanduser("~/AppData/Local/Google/Chrome/User Data"),       # Windows
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                args.profile = cand
                logger.info(f"Using REAL Chrome profile: {cand}")
                logger.warning("Make sure ALL Chrome windows are closed before continuing.")
                break
        else:
            logger.error("Could not find a real Chrome profile on this system.")
            sys.exit(1)

    urls = []
    if args.url:
        urls.append(args.url)
    if args.file and os.path.exists(args.file):
        with open(args.file) as f:
            urls.extend([ln.strip() for ln in f if ln.strip()])
    if not urls:
        urls.append("https://styxmarket.si/accounts/register/?ref=7QXIWQR1")

    debug_dir = None
    if args.debug:
        debug_dir = os.path.abspath("./debug_out")
        os.makedirs(debug_dir, exist_ok=True)
        logger.info(f"Debug artifacts -> {debug_dir}")

    logger.info(f"engine={args.engine}  channel={args.channel}  "
                f"headless={args.headless}  profile={args.profile}")
    if args.proxy:
        # Mask password in logs.
        masked = args.proxy
        if "@" in masked and "://" in masked:
            scheme, rest = masked.split("://", 1)
            creds, host = rest.rsplit("@", 1)
            user_part = creds.split(":", 1)[0]
            masked = f"{scheme}://{user_part}:***@{host}"
        logger.info(f"proxy={masked}")
    else:
        # Pre-flight: warn loudly if running on a datacenter IP (Azure/AWS/GCP).
        _warn_if_datacenter_ip()

    # Optionally wipe the profile (if a previous run got CF-flagged, its
    # fingerprint cookies will still be there and re-flag us instantly).
    if args.fresh_profile and os.path.isdir(args.profile) and not args.use_real_chrome_profile:
        import shutil
        logger.warning(f"Wiping profile dir: {args.profile}")
        shutil.rmtree(args.profile, ignore_errors=True)

    results = []
    if args.engine == "camoufox":
        _run_with_camoufox(args, urls, debug_dir, results)
    else:
        _run_with_patchright(args, urls, debug_dir, results)

    if results:
        _persist_results(results, args.output)
    else:
        logger.warning("No accounts saved.")


def _persist_results(results, output_path):
    """Append accounts to the CSV (creates with header if needed)."""
    if not results:
        return
    new_file = not os.path.isfile(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "username", "password", "secret"])
        if new_file:
            w.writeheader()
        w.writerows(results)
    logger.info(f"Saved {len(results)} account(s) -> {output_path}")


if __name__ == "__main__":
    main()
