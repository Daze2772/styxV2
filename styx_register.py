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
import re
import sys
import math
import tempfile
import threading
import time
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from loguru import logger

# Load .env once at import time so that BSC_PRIVATE_KEY / BSC_RPC_URL are
# available to send_bnb_deposit() without each caller having to remember.
# python-dotenv is non-fatal on missing file (it just no-ops), which is what
# we want when running without web3-send.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

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
    #
    # IMPORTANT: We MUST use trusted events here. Modern frontends
    # (React/Vue/etc.) gate state changes on `event.isTrusted === true`, so a
    # JS `dispatchEvent(new MouseEvent('click', ...))` is silently ignored and
    # the default tile (TRX) stays selected. That's the exact bug we are
    # fixing in this iteration: every prior synthetic-click strategy did
    # nothing visible to the framework, the user kept getting TRX deposit
    # addresses, and the script falsely reported success.
    #
    # Trusted events come from:
    #   - Playwright's native locator `.click()` (drives a real input pipeline)
    #   - `page.mouse.click(x, y)` (real CDP "Input.dispatchMouseEvent" - OS-level)
    # So this section is structured: native locator click -> mouse.click at
    # the tile's bounding-box center -> only then falls back to other selectors.
    logger.info(f"Selecting crypto: {currency_label}")
    clicked = False
    sel = None  # result of the selection check (used in retry/abort logic)

    # ---- Helper: locate the tile element, return its bounding box (in CSS px)
    # We resolve via the SAME DOM walk we used before (find the .wct-coin-name
    # with exact text, walk up to the .wallet-currency-toggler ancestor) but
    # this time we ONLY use JS to compute the bounding box. The actual click
    # is done from Python via page.mouse, which is a trusted event.
    #
    # IMPORTANT (lesson from the 2026-06-18 user log):
    #   The outer `.wallet-currency-toggler` is sometimes a zero-size wrapper
    #   (e.g. `display: contents`). Its child `.wallet-currency-toggler__title
    #    wct-with-icon` is the actual visible/clickable layer (and the one
    #   Playwright reports as "intercepts pointer events"). We MUST therefore
    #   pick the smallest VISIBLE ancestor of the name span, not just the
    #   first toggler ancestor.
    def _find_tile_box():
        try:
            return page.evaluate(
                """(label) => {
                    const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
                    const want = norm(label);

                    // 1) Styx-specific: find by .wct-coin-name
                    const nameEls = Array.from(document.querySelectorAll('.wct-coin-name'));
                    let matched = nameEls.find(el => norm(el.textContent) === want);

                    // 2) Fallback: any element whose OWN direct text node matches
                    if (!matched) {
                        const all = Array.from(document.querySelectorAll('div, span, button, a, p, label, li'));
                        matched = all.find(el => {
                            const own = norm(Array.from(el.childNodes)
                                .filter(n => n.nodeType === 3)
                                .map(n => n.textContent).join(''));
                            return own === want;
                        });
                    }
                    if (!matched) return { ok: false, reason: 'no text match' };

                    const sizeOf = (el) => {
                        try {
                            const r = el.getBoundingClientRect();
                            return { w: r.width, h: r.height, x: r.left, y: r.top };
                        } catch (e) { return { w: 0, h: 0, x: 0, y: 0 }; }
                    };
                    const isVisible = (el) => {
                        if (!el) return false;
                        const sz = sizeOf(el);
                        if (sz.w < 2 || sz.h < 2) return false;
                        try {
                            const cs = getComputedStyle(el);
                            if (cs.visibility === 'hidden' || cs.display === 'none'
                                || parseFloat(cs.opacity) === 0) return false;
                        } catch (e) {}
                        return true;
                    };

                    // Walk up: prefer the inner clickable surface
                    // .wallet-currency-toggler__title (the one that actually
                    // intercepts pointer events on this site). Then any
                    // visible interactive ancestor.
                    const isStyxTitle = (el) =>
                        el && el.className && el.className.toString
                        && /wallet-currency-toggler__title/.test(el.className.toString());
                    const isStyxOuter = (el) =>
                        el && el.className && el.className.toString
                        && /wallet-currency-toggler(?!__)/.test(el.className.toString());
                    const isInteractive = (el) => {
                        if (!el || !el.tagName) return false;
                        const t = el.tagName.toLowerCase();
                        if (t === 'button' || t === 'a' || t === 'label') return true;
                        const role = el.getAttribute && el.getAttribute('role');
                        if (role === 'button' || role === 'link' || role === 'tab' || role === 'radio') return true;
                        if (el.onclick != null) return true;
                        try { if (getComputedStyle(el).cursor === 'pointer') return true; } catch (e) {}
                        return false;
                    };

                    let target = null;
                    let cur = matched;

                    // Pass 1: nearest visible __title (preferred, that's the
                    // element which intercepts pointer events).
                    for (let i = 0; i < 10 && cur; i++) {
                        if (isStyxTitle(cur) && isVisible(cur)) { target = cur; break; }
                        cur = cur.parentElement;
                    }
                    // Pass 2: nearest visible outer toggler.
                    if (!target) {
                        cur = matched;
                        for (let i = 0; i < 10 && cur; i++) {
                            if (isStyxOuter(cur) && isVisible(cur)) { target = cur; break; }
                            cur = cur.parentElement;
                        }
                    }
                    // Pass 3: nearest visible interactive ancestor.
                    if (!target) {
                        cur = matched;
                        for (let i = 0; i < 10 && cur; i++) {
                            if (isInteractive(cur) && isVisible(cur)) { target = cur; break; }
                            cur = cur.parentElement;
                        }
                    }
                    // Pass 4: smallest visible ancestor (last-resort - just
                    // anything we can click that has a real bounding box).
                    if (!target) {
                        cur = matched;
                        for (let i = 0; i < 12 && cur; i++) {
                            if (isVisible(cur)) { target = cur; break; }
                            cur = cur.parentElement;
                        }
                    }
                    if (!target) {
                        return { ok: false, reason: 'no visible ancestor',
                                 text: norm(matched.textContent) };
                    }

                    try { target.scrollIntoView({block: 'center', inline: 'center'}); } catch (e) {}
                    const rect = target.getBoundingClientRect();
                    if (!rect || rect.width < 1 || rect.height < 1) {
                        return { ok: false, reason: 'zero-size rect',
                                 rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height } };
                    }
                    return {
                        ok: true,
                        x: rect.left + rect.width / 2,
                        y: rect.top + rect.height / 2,
                        w: rect.width,
                        h: rect.height,
                        tag: target.tagName,
                        cls: (target.className && target.className.toString)
                                ? target.className.toString() : '',
                    };
                }""",
                currency_label,
            )
        except Exception as e:
            logger.debug(f"  bounding-box lookup failed: {e}")
            return None

    # ---- Helper: is the tile selected? Returns dict {ok, selected, cls, ...}
    #
    # We can't rely on /active|selected/i alone - Styx's actual selected-state
    # class isn't standardized. Instead we DIFF the candidate tile's classes
    # against the other sibling tiles: the one with the unique class is the
    # selected one. We also check the usual a11y / form-state signals.
    def _is_selected():
        try:
            return page.evaluate(
                """(label) => {
                    const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
                    const want = norm(label);
                    const nameEls = Array.from(document.querySelectorAll('.wct-coin-name'));

                    // Helper: walk up and collect BOTH the inner __title and
                    // the outer .wallet-currency-toggler ancestor. Either (or
                    // both) may carry the "selected" state class - on Styx the
                    // outer is often display:contents so the visible state is
                    // on the inner __title.
                    const ancestorsOf = (el) => {
                        let inner = null, outer = null;
                        let cur = el;
                        for (let i = 0; i < 12 && cur; i++) {
                            const cls = (cur.className && cur.className.toString)
                                           ? cur.className.toString() : '';
                            if (!inner && /wallet-currency-toggler__title/.test(cls)) inner = cur;
                            if (!outer && /wallet-currency-toggler(?!__)/.test(cls)) outer = cur;
                            if (inner && outer) break;
                            cur = cur.parentElement;
                        }
                        return { inner, outer };
                    };
                    const stateClassRx = /\\bactive\\b|\\bselected\\b|\\bchecked\\b|\\bchosen\\b|\\bcurrent\\b|--is-active|--is-selected|is-active|is-selected/i;
                    const isStateOn = (el) => {
                        if (!el) return false;
                        const cls = (el.className && el.className.toString)
                                       ? el.className.toString() : '';
                        if (stateClassRx.test(cls)) return true;
                        if (el.getAttribute && (
                               el.getAttribute('aria-selected') === 'true'
                            || el.getAttribute('aria-pressed') === 'true'
                            || el.getAttribute('aria-checked') === 'true'
                            || el.getAttribute('data-active') === 'true'
                            || el.getAttribute('data-selected') === 'true')) return true;
                        return false;
                    };

                    const targetName = nameEls.find(el => norm(el.textContent) === want);
                    if (!targetName) {
                        const tiles = nameEls.map(n => norm(n.textContent));
                        return { ok: false, reason: 'tile not in DOM', tilesFound: tiles };
                    }
                    const { inner: tInner, outer: tOuter } = ancestorsOf(targetName);
                    if (!tInner && !tOuter) return { ok: false, reason: 'no toggler ancestor' };

                    // All sibling tiles. Use outer if present, else __title.
                    let allOuter = Array.from(document.querySelectorAll('.wallet-currency-toggler'))
                        .filter(el => !/__/.test(el.className.toString()));
                    let allInner = Array.from(document.querySelectorAll('.wallet-currency-toggler__title'));
                    const tilesFound = (allOuter.length ? allOuter : allInner).map(t => {
                        const n = t.querySelector('.wct-coin-name')
                              || (t.parentElement && t.parentElement.querySelector('.wct-coin-name'));
                        return n ? norm(n.textContent) : '';
                    });

                    // (1) Direct signals on inner OR outer.
                    const directSelected = isStateOn(tInner) || isStateOn(tOuter);

                    // (2) Hidden input / radio inside either ancestor.
                    let inputChecked = false;
                    const checkInputs = (el) => {
                        if (!el) return;
                        el.querySelectorAll('input').forEach(inp => {
                            if (inp.checked) inputChecked = true;
                        });
                    };
                    checkInputs(tInner);
                    checkInputs(tOuter);

                    // (3) Class-diff vs siblings - run on whichever ancestor
                    // level we're using to identify tiles (prefer outer, fall
                    // back to inner).
                    const tilesForDiff = (allOuter.length ? allOuter : allInner);
                    let diffTarget;
                    if (allOuter.length) {
                        diffTarget = tOuter || tInner;
                    } else {
                        diffTarget = tInner || tOuter;
                    }
                    let diffSelected = false;
                    let uniqueToTarget = [];
                    let uniqueToOthers = [];
                    if (diffTarget && tilesForDiff.length > 1) {
                        const tClsSet = new Set(diffTarget.className.toString().split(/\\s+/).filter(Boolean));
                        const sibSets = tilesForDiff
                            .filter(t => t !== diffTarget)
                            .map(t => new Set(t.className.toString().split(/\\s+/).filter(Boolean)));
                        for (const c of tClsSet) {
                            if (sibSets.every(s => !s.has(c))) uniqueToTarget.push(c);
                        }
                        if (sibSets.length > 0) {
                            const inter = new Set(sibSets[0]);
                            for (let i = 1; i < sibSets.length; i++) {
                                for (const c of Array.from(inter)) {
                                    if (!sibSets[i].has(c)) inter.delete(c);
                                }
                            }
                            for (const c of inter) {
                                if (!tClsSet.has(c)) uniqueToOthers.push(c);
                            }
                        }
                        diffSelected = uniqueToTarget.length > 0 || uniqueToOthers.length > 0;
                    }

                    // Which tile does the page currently appear to highlight?
                    let highlighted = null;
                    const tilesForHL = allOuter.length ? allOuter : allInner;
                    for (const t of tilesForHL) {
                        let on = isStateOn(t);
                        if (!on) {
                            // Also check the matching inner/outer counterpart.
                            if (allOuter.length) {
                                const innerOf = t.querySelector('.wallet-currency-toggler__title');
                                if (isStateOn(innerOf)) on = true;
                            } else {
                                const outerOf = t.closest('.wallet-currency-toggler');
                                if (isStateOn(outerOf)) on = true;
                            }
                        }
                        if (!on) {
                            const ins = t.querySelectorAll('input');
                            for (const i of ins) if (i.checked) { on = true; break; }
                        }
                        if (on) {
                            const n = t.querySelector('.wct-coin-name')
                                  || (t.parentElement && t.parentElement.querySelector('.wct-coin-name'));
                            highlighted = n ? norm(n.textContent) : '';
                            break;
                        }
                    }

                    return {
                        ok: true,
                        selected: directSelected || inputChecked || diffSelected,
                        directSelected: directSelected,
                        inputChecked: inputChecked,
                        diffSelected: diffSelected,
                        uniqueToTarget: uniqueToTarget,
                        uniqueToOthers: uniqueToOthers,
                        highlighted: highlighted,
                        innerCls: tInner ? tInner.className.toString() : null,
                        outerCls: tOuter ? tOuter.className.toString() : null,
                        tilesFound: tilesFound,
                    };
                }""",
                currency_label,
            )
        except Exception as e:
            logger.debug(f"selection check failed: {e}")
            return None

    # ----- Strategy A: NATIVE Playwright locator click on the INNER __title -----
    # On Styx the outer .wallet-currency-toggler is often display:contents (zero
    # bounding box, Playwright reports "element is not visible"). The actual
    # clickable layer that intercepts pointer events is the inner
    # .wallet-currency-toggler__title. We target THAT directly.
    inner_loc = page.locator(
        ".wallet-currency-toggler__title",
        has=page.locator(".wct-coin-name", has_text=currency_label),
    ).first
    try:
        try:
            inner_loc.scroll_into_view_if_needed(timeout=4000)
        except Exception as e:
            logger.debug(f"  scroll_into_view on inner __title failed (non-fatal): {e}")
        inner_loc.click(timeout=5000)
        clicked = True
        logger.info(f"  -> clicked '{currency_label}' via inner __title locator (trusted)")
    except Exception as e:
        logger.debug(f"  inner __title locator click failed: {e}")

    # ----- Strategy B: page.mouse.click at the tile's bounding box (trusted) -----
    if not clicked:
        box = _find_tile_box()
        if box and box.get("ok"):
            try:
                x, y = float(box["x"]), float(box["y"])
                # tiny jitter so we don't always click pixel-perfect center
                jx = x + random.uniform(-2.0, 2.0)
                jy = y + random.uniform(-2.0, 2.0)
                page.mouse.move(jx, jy, steps=random.randint(8, 14))
                time.sleep(random.uniform(0.05, 0.15))
                page.mouse.click(jx, jy, delay=random.randint(40, 90))
                clicked = True
                logger.info(
                    f"  -> mouse-clicked '{currency_label}' at ({jx:.1f},{jy:.1f}) "
                    f"size={box['w']:.0f}x{box['h']:.0f} cls='{box.get('cls','')[:60]}' (trusted)"
                )
            except Exception as e:
                logger.debug(f"  mouse.click at box failed: {e}")
        else:
            logger.debug(f"  could not resolve tile box: {box}")

    # ----- Strategy C: force-click on the name span itself -----
    # Clicking the .wct-coin-name with force=True bypasses Playwright's
    # "element intercepts pointer events" check. Since the interceptor (the
    # __title div) is precisely the layer with the click handler, clicking
    # at the name-span's position effectively clicks the title.
    if not clicked:
        try:
            page.locator(".wct-coin-name", has_text=currency_label).first.click(
                timeout=4000, force=True
            )
            clicked = True
            logger.info(f"  -> force-clicked .wct-coin-name '{currency_label}' (trusted)")
        except Exception as e:
            logger.debug(f"  force-click on .wct-coin-name failed: {e}")

    # ----- Strategy D: legacy text/role fallbacks with force=True -----
    if not clicked:
        strategies = [
            ("role=button exact",
             lambda: page.get_by_role("button", name=currency_label, exact=True)
                          .first.click(timeout=4000, force=True)),
            ("text exact (force)",
             lambda: page.get_by_text(currency_label, exact=True).first
                          .click(timeout=4000, force=True)),
            ("text contains (force)",
             lambda: page.get_by_text(currency_label, exact=False).first
                          .click(timeout=4000, force=True)),
        ]
        for name, strat in strategies:
            try:
                strat()
                clicked = True
                logger.info(f"  -> clicked '{currency_label}' (fallback: {name})")
                break
            except Exception as e:
                logger.debug(f"  topup tile fallback '{name}' failed: {e}")

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

    # 1b) VERIFY the right tile is actually selected, and RETRY via mouse.click
    sel = _is_selected()
    logger.info(f"  selection check: {sel}")

    def _retry_click():
        # Use the bounding-box mouse click on retry too - it's the most
        # reliable trusted-event path we have.
        box = _find_tile_box()
        if box and box.get("ok"):
            try:
                x = float(box["x"]) + random.uniform(-2.0, 2.0)
                y = float(box["y"]) + random.uniform(-2.0, 2.0)
                page.mouse.move(x, y, steps=random.randint(8, 14))
                time.sleep(random.uniform(0.05, 0.15))
                page.mouse.click(x, y, delay=random.randint(40, 90))
                return True
            except Exception as e:
                logger.debug(f"  retry mouse.click failed: {e}")
        # Fallback: inner __title click (force-click bypasses overlay checks).
        try:
            page.locator(
                ".wallet-currency-toggler__title",
                has=page.locator(".wct-coin-name", has_text=currency_label),
            ).first.click(timeout=4000, force=True)
            return True
        except Exception as e:
            logger.debug(f"  retry __title click failed: {e}")
        # Last resort: force-click the name span.
        try:
            page.locator(".wct-coin-name", has_text=currency_label).first.click(
                timeout=4000, force=True
            )
            return True
        except Exception as e:
            logger.debug(f"  retry name-span force-click failed: {e}")
            return False

    retries = 0
    while sel and sel.get("ok") and not sel.get("selected") and retries < 2:
        retries += 1
        logger.warning(
            f"'{currency_label}' click didn't latch (highlighted='{sel.get('highlighted')}') "
            f"- retry {retries}/2 via trusted mouse click."
        )
        if _retry_click():
            time.sleep(random.uniform(0.6, 1.0))
            sel = _is_selected()
            logger.info(f"  selection check (retry {retries}): {sel}")
        else:
            break

    if sel and sel.get("ok") and not sel.get("selected"):
        logger.error(
            f"'{currency_label}' still not selected after {retries} retries "
            f"(page currently highlights '{sel.get('highlighted')}'). "
            f"Aborting top-up to avoid sending the wrong currency."
        )
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "12b_topup_wrong_tile.png"),
                            full_page=True)
        return False

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
    #
    # IMPORTANT: this section used to false-fail in a specific way - the click
    # would go through, the Vue app would route away from the top-up form
    # MID-CLICK, and Playwright would then report "Timeout: element is not
    # visible/stable" even though the click was already registered on the
    # server. Worse, the previous final fallback was `button[type='submit'].last`
    # which on Styx picks up a HIDDEN <button class="refund-modal__button">
    # (the refund-policy modal that lives in the DOM but never renders), so
    # strategy 6 would always "fail" and the whole function returned False.
    #
    # Fix: every strategy is wrapped in a "did the page already move past the
    # form?" check (we look at URL change or appearance of the deposit
    # address / QR / 'protected by Styx' block). If yes -> treat as success
    # regardless of what Playwright reported. We also pre-check BEFORE
    # clicking in case the form already submitted.
    logger.info("Clicking TOP UP BALANCE...")

    def _post_submit_state():
        """Return a dict describing whether the page has navigated to the
        post-submit deposit-address state. Used to detect successful click
        even when Playwright timed out mid-navigation."""
        try:
            return page.evaluate(
                """() => {
                    const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
                    const text = norm(document.body ? document.body.innerText : '');

                    // (1) The post-submit page shows a wallet address (long
                    //     alphanumeric string), a QR canvas/img, and copy
                    //     buttons. The pre-submit form does not.
                    const hasQR = !!document.querySelector(
                        'canvas, img[src*="qr"], img[alt*="QR" i], svg[class*="qr" i], [class*="qrcode"], [class*="qr-code"]');
                    const addrRx = /\\b[a-zA-Z0-9]{25,}\\b/;
                    const hasAddress = addrRx.test(text);
                    const hasProtectedBanner = /all orders are protected|deposit address|send (the )?(exact|payment|amount)/i.test(text);
                    const hasSuccessCheck = /(deposit (was )?(received|confirmed|successful)|payment (was )?(received|confirmed|successful)|thank you|completed)/i.test(text);

                    // (2) The top-up form is GONE (no amount input, no
                    //     wallet-currency-toggler tiles, no TOP UP BALANCE
                    //     button text).
                    const stillHasAmount = !!document.querySelector(
                        "input[name='amount'], input[name='payment_amount']");
                    const stillHasTiles = !!document.querySelector('.wallet-currency-toggler');
                    const stillHasTopupBtn = /\\bTOP UP BALANCE\\b/i.test(text);
                    const formGone = !stillHasAmount && !stillHasTopupBtn && !stillHasTiles;

                    return {
                        hasQR, hasAddress, hasProtectedBanner, hasSuccessCheck,
                        stillHasAmount, stillHasTiles, stillHasTopupBtn, formGone,
                        urlPath: location.pathname + location.search,
                    };
                }"""
            )
        except Exception as e:
            logger.debug(f"  post-submit state probe failed: {e}")
            return None

    def _looks_submitted(state):
        if not state:
            return False
        # Strong signal: deposit-address / QR / "protected by Styx" present.
        if state.get("hasAddress") or state.get("hasQR") or state.get("hasProtectedBanner"):
            return True
        # Strong signal: already showing the confirmation/success state.
        if state.get("hasSuccessCheck"):
            return True
        # Fallback: the entire top-up form is gone.
        if state.get("formGone"):
            return True
        return False

    pre_state = _post_submit_state()
    if _looks_submitted(pre_state):
        logger.info(f"  page is already past the top-up form ({pre_state}); skipping click.")
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "16_topup_submitted.png"),
                            full_page=True)
        logger.success(f"Top-up submitted: {currency_label} amount={amount}")
        return True
    start_url = (pre_state or {}).get("urlPath")

    # Pre-flight: scroll the TOP UP BALANCE button into view BEFORE any
    # click strategy runs. The user reported the button is off-screen on
    # their viewport - locator.click() auto-scrolls, but the diagnostic
    # log + a manual screenshot are taken before that, so this scroll keeps
    # the debug artifacts useful and reduces the chance of any strategy
    # missing because of viewport math.
    try:
        page.evaluate(
            """() => {
                const tgt = document.querySelector('[data-i18n="wallet.top_up_btn"]')
                          || document.querySelector('form > div > button')
                          || document.querySelector('form button');
                if (tgt) tgt.scrollIntoView({behavior: 'instant', block: 'center'});
            }"""
        )
        time.sleep(0.5)
    except Exception as e:
        logger.debug(f"  pre-flight scroll-into-view failed: {e}")

    # Strategies, in order. We deliberately do NOT use
    # `button[type='submit'].last` because Styx has a hidden refund-modal
    # submit button that gets picked first and causes a false "could not
    # click" failure. We constrain to visible buttons only.
    #
    # FIELD-VERIFIED HTML (from user 2026-06-19):
    #   <span data-i18n="wallet.top_up_btn">Top up balance</span>
    #   inside  form > div > button > span
    # The button is below the fold on shorter viewports, so EVERY strategy
    # must scroll into view first (Playwright's locator.click() does this
    # automatically, but page.mouse.click() does NOT - it clicks at
    # viewport coords, and if the button is below the viewport, it misses).
    def _scroll_then_click(selector, force=False):
        """Return a thunk that scrolls the selector into view then clicks."""
        def _do():
            loc = page.locator(selector).first
            try:
                loc.scroll_into_view_if_needed(timeout=3000)
                time.sleep(0.3)
            except Exception:
                pass
            loc.click(timeout=5000, force=force)
        return _do

    btn_strategies = [
        # Strategy 0: locale-stable. data-i18n is Vue-i18n's identifier and
        # does NOT change with language; it survives any future locale
        # changes / wording tweaks.
        ("data-i18n=wallet.top_up_btn (button parent)",
         _scroll_then_click('button:has([data-i18n="wallet.top_up_btn"])')),
        ("data-i18n=wallet.top_up_btn (span itself)",
         _scroll_then_click('[data-i18n="wallet.top_up_btn"]')),
        # Strategy 1: literal CSS path from the user's element inspector.
        # On /wallet/top-up/ this form contains EXACTLY ONE button.
        ("form > div > button (CSS path)",
         _scroll_then_click('form > div > button')),
        ("form button:visible",
         _scroll_then_click('form button:visible')),
        # Strategies 2-7: text/role matchers (kept as fallback in case the
        # data-i18n attr ever disappears).
        ("role=TOP UP BALANCE",
         lambda: page.get_by_role("button", name="TOP UP BALANCE").first.click(timeout=5000)),
        ("role=Top up balance",
         lambda: page.get_by_role("button", name="Top up balance").first.click(timeout=5000)),
        ("text=TOP UP BALANCE",
         _scroll_then_click("button:has-text('TOP UP BALANCE'):visible")),
        ("text=Top up balance",
         _scroll_then_click("button:has-text('Top up balance'):visible")),
        ("text=Top up (visible)",
         _scroll_then_click("button:has-text('Top up'):visible")),
        # Force-click variants - if Playwright thinks the element isn't
        # actionable but we know it's the one we want, force the click.
        ("data-i18n (force)",
         _scroll_then_click('button:has([data-i18n="wallet.top_up_btn"])', force=True)),
        ("text=TOP UP BALANCE (force)",
         _scroll_then_click("button:has-text('TOP UP BALANCE')", force=True)),
        ("text=Top up (force)",
         _scroll_then_click("button:has-text('Top up')", force=True)),
    ]

    # ---- Diagnostic: enumerate every visible button-like element so the
    # debug log tells us exactly what's on the page if all strategies fail.
    try:
        candidates_dump = page.evaluate(
            """() => {
                const out = [];
                const els = document.querySelectorAll(
                    'button, [role="button"], input[type="submit"], a.button, a[class*="button" i], a.btn, a[class*="btn" i]');
                for (const el of els) {
                    try {
                        const r = el.getBoundingClientRect();
                        const cs = getComputedStyle(el);
                        if (r.width < 4 || r.height < 4) continue;
                        if (cs.visibility === 'hidden' || cs.display === 'none'
                            || parseFloat(cs.opacity) === 0) continue;
                        if (r.top < -10 || r.left < -10
                            || r.top > window.innerHeight + 100) continue;
                        const txt = (el.innerText || el.textContent || '').trim().slice(0, 80);
                        out.push({
                            tag: el.tagName.toLowerCase(),
                            type: el.getAttribute('type') || '',
                            cls: (el.className && el.className.toString
                                  ? el.className.toString() : '').slice(0, 80),
                            txt: txt,
                            disabled: el.disabled === true
                                     || el.getAttribute('aria-disabled') === 'true',
                            x: Math.round(r.left + r.width / 2),
                            y: Math.round(r.top + r.height / 2),
                        });
                    } catch (e) {}
                }
                return out.slice(0, 40);
            }"""
        )
        if candidates_dump:
            logger.info("  visible clickable candidates on top-up page:")
            for c in candidates_dump:
                logger.info(
                    f"    <{c['tag']} type={c['type']!r} disabled={c['disabled']} "
                    f"@({c['x']},{c['y']}) cls={c['cls']!r} txt={c['txt']!r}")
    except Exception as e:
        logger.debug(f"  diagnostic enumeration failed: {e}")

    btn_clicked = False
    for i, (name, strat) in enumerate(btn_strategies, 1):
        try:
            strat()
            btn_clicked = True
            logger.info(f"  -> clicked TOP UP BALANCE (strategy {i}: {name})")
            break
        except Exception as e:
            # The click MAY have actually gone through but Playwright bailed
            # out of its actionability wait because the page is navigating.
            # Check the page state - if it has moved past the form, treat
            # this as a successful click.
            err_msg = str(e).splitlines()[0] if str(e) else 'unknown'
            time.sleep(0.5)  # small grace for the navigation to settle
            post_state = _post_submit_state()
            if _looks_submitted(post_state):
                logger.info(
                    f"  -> strategy {i} ({name}) reported '{err_msg[:80]}' but the "
                    f"page has moved on ({post_state}). Treating as successful click."
                )
                btn_clicked = True
                break
            # URL changed even though no deposit signals yet? Probably still
            # success (e.g. POST -> GET redirect in flight).
            if (start_url and post_state
                and post_state.get("urlPath") and post_state.get("urlPath") != start_url):
                logger.info(
                    f"  -> strategy {i} ({name}) reported timeout but URL "
                    f"changed ('{start_url}' -> '{post_state.get('urlPath')}'). "
                    f"Treating as successful click."
                )
                btn_clicked = True
                break
            logger.debug(f"  topup-button strategy {i} ({name}) failed: {err_msg[:200]}")

    # ---- Strategy 8: trusted mouse-click on the best heuristic candidate.
    # Finds the visible+enabled button-like element whose text best matches
    # /top.?up|deposit|submit|pay/i, scores by position (bottom-of-form is
    # better), and clicks via page.mouse.click (real CDP event).
    if not btn_clicked:
        logger.info("  strategy 8: heuristic mouse-click on best topup-button candidate.")
        try:
            best = page.evaluate(
                """() => {
                    const rx = /(top.?up|deposit|pay|submit|continue|confirm)/i;
                    const allow = ['button', 'a', 'input', 'div', 'span'];
                    const out = [];
                    const els = document.querySelectorAll('*');
                    for (const el of els) {
                        try {
                            const tag = el.tagName ? el.tagName.toLowerCase() : '';
                            if (!allow.includes(tag)) continue;
                            const r = el.getBoundingClientRect();
                            if (r.width < 30 || r.height < 18) continue;
                            const cs = getComputedStyle(el);
                            if (cs.visibility === 'hidden' || cs.display === 'none'
                                || parseFloat(cs.opacity) === 0) continue;
                            if (el.disabled === true
                                || el.getAttribute('aria-disabled') === 'true') continue;
                            const txt = (el.innerText || el.textContent || '').trim();
                            if (!txt || txt.length > 60) continue;
                            if (!rx.test(txt)) continue;
                            // skip the refund-modal hidden submit etc.
                            const cls = (el.className && el.className.toString
                                          ? el.className.toString() : '').toLowerCase();
                            if (/refund|modal/.test(cls)) continue;
                            let score = 0;
                            if (/top.?up.?balance/i.test(txt)) score += 80;
                            else if (/top.?up/i.test(txt)) score += 60;
                            else if (/deposit|pay/i.test(txt)) score += 40;
                            else if (/submit|confirm/i.test(txt)) score += 20;
                            // prefer real buttons / submits
                            if (tag === 'button') score += 20;
                            if (el.getAttribute('type') === 'submit') score += 20;
                            if (el.getAttribute('role') === 'button') score += 10;
                            // prefer elements lower in the viewport (form CTA usually is)
                            score += Math.min(40, r.top / window.innerHeight * 40);
                            // prefer larger buttons
                            score += Math.min(15, (r.width * r.height) / 4000);
                            out.push({
                                tag, txt: txt.slice(0, 80), cls: cls.slice(0, 80),
                                x: r.left + r.width / 2, y: r.top + r.height / 2,
                                w: r.width, h: r.height, score,
                            });
                        } catch (e) {}
                    }
                    out.sort((a, b) => b.score - a.score);
                    return out.slice(0, 5);
                }"""
            )
            if best:
                logger.info(f"  candidates ranked: {best}")
                top = best[0]
                logger.info(
                    f"  clicking best candidate <{top['tag']}> "
                    f"'{top['txt']}' at ({top['x']:.0f},{top['y']:.0f}) "
                    f"score={top['score']:.0f}")
                # Scroll the best candidate into view BEFORE mouse.click so
                # we hit the right viewport coordinates. mouse.click does NOT
                # auto-scroll the way Playwright's locator.click does.
                try:
                    rescroll = page.evaluate(
                        """(txt) => {
                            const rx = /(top.?up|deposit|pay|submit|continue|confirm)/i;
                            const els = document.querySelectorAll('button, [role="button"], input[type="submit"], a, div, span');
                            for (const el of els) {
                                try {
                                    const t = (el.innerText || el.textContent || '').trim();
                                    if (t === txt || (t && txt.includes(t.slice(0,20)))) {
                                        el.scrollIntoView({behavior: 'instant', block: 'center'});
                                        const r = el.getBoundingClientRect();
                                        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
                                    }
                                } catch (e) {}
                            }
                            return null;
                        }""", top["txt"]
                    )
                    if rescroll and rescroll.get("x") is not None:
                        time.sleep(0.4)
                        top["x"], top["y"] = rescroll["x"], rescroll["y"]
                        logger.info(
                            f"  scrolled into view; new coords ({top['x']:.0f},{top['y']:.0f})")
                except Exception as e:
                    logger.debug(f"  scroll-before-click failed: {e}")
                try:
                    page.mouse.click(float(top["x"]), float(top["y"]),
                                     delay=random.randint(40, 90))
                except Exception as e:
                    logger.debug(f"  heuristic mouse.click failed: {e}")
                time.sleep(1.0)
                post_state = _post_submit_state()
                if _looks_submitted(post_state):
                    logger.info(f"  -> heuristic click worked: {post_state}")
                    btn_clicked = True
        except Exception as e:
            logger.debug(f"  heuristic-candidate evaluation failed: {e}")

    # ---- Strategy 9: form.requestSubmit() - native submit event so Vue's
    # @submit handler runs. Targets the form that contains the amount input.
    if not btn_clicked:
        logger.info("  strategy 9: form.requestSubmit() on the amount-input's form.")
        try:
            submitted = page.evaluate(
                """() => {
                    const inp = document.querySelector(
                        "input[name='amount'], input[name='payment_amount'], input[type='number'], input.input__input");
                    if (!inp) return { ok: false, reason: 'no amount input' };
                    let f = inp.form;
                    if (!f) {
                        let p = inp.parentElement;
                        while (p && p.tagName !== 'FORM') p = p.parentElement;
                        f = p;
                    }
                    if (!f) return { ok: false, reason: 'no form ancestor' };
                    inp.dispatchEvent(new Event('blur', { bubbles: true }));
                    if (typeof f.requestSubmit === 'function') {
                        f.requestSubmit();
                        return { ok: true, used: 'requestSubmit' };
                    }
                    f.submit();
                    return { ok: true, used: 'submit' };
                }"""
            )
            logger.info(f"  form-submit result: {submitted}")
            time.sleep(1.5)
            post_state = _post_submit_state()
            if _looks_submitted(post_state):
                logger.info("  -> form.requestSubmit worked.")
                btn_clicked = True
        except Exception as e:
            logger.debug(f"  form.requestSubmit failed: {e}")

    # ---- Strategy 10: press Enter on the amount input (some Vue forms
    # listen for keydown.enter to submit).
    if not btn_clicked:
        logger.info("  strategy 10: press Enter on the amount input.")
        try:
            for sel in ("input[name='amount']", "input[name='payment_amount']",
                        "input[type='number']", "input.input__input:visible"):
                try:
                    loc = page.locator(sel).first
                    loc.focus(timeout=2000)
                    loc.press("Enter", timeout=3000)
                    time.sleep(1.5)
                    if _looks_submitted(_post_submit_state()):
                        logger.info(f"  -> Enter-key on {sel} worked.")
                        btn_clicked = True
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"  Enter-key strategy failed: {e}")

    if not btn_clicked:
        # Final safety net: maybe the click DID go through earlier in this
        # loop but every strategy raised before we noticed. Re-probe.
        final_state = _post_submit_state()
        if _looks_submitted(final_state):
            logger.warning(
                "  All button strategies raised, but the page has moved past "
                f"the top-up form anyway ({final_state}). Treating as success."
            )
            btn_clicked = True
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


# ---------- web3 BNB deposit ---------------------------------------------------
# A valid EVM address is "0x" + 40 hex chars. BSC native BNB has no prefix
# (unlike TRX which uses "T...") so this regex is intentionally strict.
_EVM_ADDR_RE = re.compile(r"0x[a-fA-F0-9]{40}")


def extract_deposit_address(page, timeout_seconds=30, debug_dir=None):
    """Scrape the on-page deposit address from Styx's post-submit top-up view.

    After do_topup() submits "TOP UP BALANCE", Styx renders a panel with a QR
    code, the deposit address (an EVM 0x... string for BNB/BEP20), and the
    requested amount. We poll the DOM for that address until we find one or
    the timeout expires.

    Strategy (in priority order, all best-effort):
      1. Look at inputs/textareas (Styx renders the address in a copy-input).
      2. Look at any element whose data-* attrs contain the address.
      3. Fall back to regex over the entire body innerText.

    Returns the EVM checksum-normalised lower-case address as a string, or
    None on timeout / error / wrong-coin (TRX prefix etc.).
    """
    logger.info("Extracting deposit address from top-up page...")
    deadline = time.time() + max(5, int(timeout_seconds))
    last_text_len = 0
    while time.time() < deadline:
        try:
            data = page.evaluate(
                """() => {
                    const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
                    // (1) inputs/textareas
                    const inputs = Array.from(document.querySelectorAll(
                        'input, textarea'));
                    const fromInputs = inputs
                        .map(el => (el.value || el.getAttribute('value') || '').toString())
                        .filter(v => /^0x[a-fA-F0-9]{40}$/.test(v.trim()));
                    // (2) data-* attrs on any element
                    const fromAttrs = [];
                    Array.from(document.querySelectorAll('*')).slice(0, 5000).forEach(el => {
                        try {
                            for (const a of el.attributes) {
                                if (a.name.startsWith('data-') && /^0x[a-fA-F0-9]{40}$/.test(a.value.trim())) {
                                    fromAttrs.push(a.value.trim());
                                }
                            }
                        } catch (e) {}
                    });
                    // (3) body text
                    const bodyText = norm(document.body ? document.body.innerText : '');
                    const matches = bodyText.match(/0x[a-fA-F0-9]{40}/g) || [];
                    return {
                        fromInputs: fromInputs,
                        fromAttrs: fromAttrs,
                        fromText: matches,
                        textLen: bodyText.length,
                    };
                }"""
            )
        except Exception as e:
            logger.debug(f"  address-probe failed: {e}")
            time.sleep(1)
            continue

        # priority: inputs > data-attrs > body text
        candidates = list(data.get("fromInputs") or [])
        candidates += list(data.get("fromAttrs") or [])
        candidates += list(data.get("fromText") or [])
        # de-dupe preserving order, only valid 0x...40hex
        seen = set()
        clean = []
        for c in candidates:
            c = (c or "").strip()
            if _EVM_ADDR_RE.fullmatch(c) and c.lower() not in seen:
                seen.add(c.lower())
                clean.append(c)
        if clean:
            addr = clean[0]
            logger.success(f"Found deposit address: {addr}")
            if debug_dir:
                try:
                    with open(os.path.join(debug_dir, "deposit_address.txt"),
                              "w") as f:
                        f.write(addr + "\n")
                except Exception:
                    pass
            return addr

        if data.get("textLen", 0) != last_text_len:
            last_text_len = data.get("textLen", 0)
            logger.debug(f"  no 0x address yet (textLen={last_text_len})")
        time.sleep(1)

    logger.error("Timed out extracting deposit address from top-up page.")
    if debug_dir:
        try:
            page.screenshot(path=os.path.join(debug_dir,
                            "15b_no_deposit_address.png"), full_page=True)
        except Exception:
            pass
    return None


def send_bnb_deposit(to_address, amount_bnb=0.001, max_retries=3,
                     debug_dir=None):
    """Send a native BNB transfer on BSC mainnet (chain id 56).

    Reads BSC_PRIVATE_KEY / BSC_RPC_URL from environment (already loaded from
    /app/.env at module import time). Retries up to `max_retries` times on
    RPC/timeout/value errors with exponential backoff (5s, 15s, 30s, ...).

    On total failure raises RuntimeError - caller MUST treat this as fatal
    per user spec ("halt the entire run"). On success returns the tx hash
    (0x-prefixed hex string).

    NOTE: web3.py is only imported inside this function so that the rest of
    the script (CAPTCHA solving, registration, ...) keeps running for users
    who never enable the web3-send feature.
    """
    # ----- preflight env / lib checks ----------------------------------------
    pk = os.environ.get("BSC_PRIVATE_KEY")
    if not pk:
        raise RuntimeError(
            "BSC_PRIVATE_KEY is not set in /app/.env - cannot send BNB. "
            "Either populate the env or pass --no-web3-send.")
    rpc_url = os.environ.get("BSC_RPC_URL",
                             "https://bsc-dataseed.bnbchain.org")
    rpc_url_fb = os.environ.get("BSC_RPC_URL_FALLBACK",
                                "https://bsc-dataseed.binance.org")
    try:
        from web3 import Web3
        from web3.exceptions import TimeExhausted, Web3RPCError
        from eth_account import Account
    except ImportError as e:
        raise RuntimeError(
            f"web3.py is not installed ({e}). Run "
            f"`pip install -r /app/requirements.txt`.") from e

    if not _EVM_ADDR_RE.fullmatch((to_address or "").strip()):
        raise RuntimeError(
            f"Refusing to send to invalid EVM address: {to_address!r}")

    # ----- build w3 (with fallback RPC) --------------------------------------
    def _connect(url):
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 20}))
        if not w3.is_connected():
            raise RuntimeError(f"RPC not reachable: {url}")
        cid = int(w3.eth.chain_id)
        if cid != 56:
            raise RuntimeError(
                f"Wrong chain id on {url}: got {cid}, expected 56 (BSC mainnet)")
        return w3

    try:
        w3 = _connect(rpc_url)
    except Exception as e:
        logger.warning(f"Primary BSC RPC failed ({rpc_url}): {e}; "
                       f"falling back to {rpc_url_fb}")
        w3 = _connect(rpc_url_fb)

    acct = Account.from_key(pk)
    recipient = Web3.to_checksum_address(to_address)
    value_wei = w3.to_wei(Decimal(str(amount_bnb)), "ether")

    # Balance sanity check (only logged - actual send will fail anyway if low)
    try:
        bal_wei = w3.eth.get_balance(acct.address)
        gas_price_est = w3.eth.gas_price
        needed = value_wei + 21000 * gas_price_est
        logger.info(
            f"Sender {acct.address} balance={w3.from_wei(bal_wei, 'ether')} "
            f"BNB, need ~{w3.from_wei(needed, 'ether')} BNB "
            f"(value+gas), gasPrice={w3.from_wei(gas_price_est, 'gwei')} gwei")
        if bal_wei < needed:
            raise RuntimeError(
                f"Insufficient BNB balance: have "
                f"{w3.from_wei(bal_wei, 'ether')}, need ~"
                f"{w3.from_wei(needed, 'ether')}. Top up the wallet at "
                f"{acct.address} on BSC.")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Balance check failed ({e}); proceeding anyway.")

    last_error = None
    backoff = [5, 15, 30]
    for attempt in range(1, max_retries + 1):
        try:
            nonce = w3.eth.get_transaction_count(acct.address,
                                                 block_identifier="pending")
            gas_price = w3.eth.gas_price
            tx = {
                "chainId": 56,
                "from": acct.address,
                "to": recipient,
                "value": value_wei,
                "nonce": nonce,
                "gas": 21000,
                "gasPrice": gas_price,
            }
            signed = w3.eth.account.sign_transaction(tx, private_key=acct.key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            tx_hex = tx_hash.hex()
            if not tx_hex.startswith("0x"):
                tx_hex = "0x" + tx_hex
            logger.success(
                f"[attempt {attempt}/{max_retries}] BNB transfer broadcast: "
                f"{tx_hex} - waiting for receipt...")
            receipt = w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=120, poll_latency=3)
            if receipt.status != 1:
                raise RuntimeError(
                    f"Tx mined but reverted: status={receipt.status} "
                    f"hash={tx_hex}")
            logger.success(
                f"BNB deposit confirmed on-chain: {tx_hex} "
                f"(block {receipt.blockNumber}, gasUsed={receipt.gasUsed})")
            if debug_dir:
                try:
                    with open(os.path.join(debug_dir, "tx_hash.txt"),
                              "a") as f:
                        f.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
                            f"{recipient}\t{amount_bnb}\t{tx_hex}\n")
                except Exception:
                    pass
            return tx_hex
        except (TimeExhausted, Web3RPCError, ValueError, OSError,
                RuntimeError) as e:
            last_error = e
            logger.error(
                f"[attempt {attempt}/{max_retries}] BNB send failed: "
                f"{type(e).__name__}: {e}")
            if attempt < max_retries:
                wait = backoff[min(attempt - 1, len(backoff) - 1)]
                logger.warning(f"  retrying in {wait}s...")
                time.sleep(wait)
                continue
            break

    raise RuntimeError(
        f"BNB deposit failed after {max_retries} attempts. "
        f"Last error: {last_error}")


def wait_for_deposit_confirmed(page, timeout_seconds=900, poll_interval=3,
                               debug_dir=None):
    """Block until Styx's top-up page shows a "deposit received / confirmed"
    state.

    On Styx the post-submit page shows a deposit-address block (QR code +
    address + amount). When the on-chain transaction is detected, this whole
    block is replaced by a green checkmark / success message. We poll the DOM
    for either signal:

       (1) the deposit-address block / QR / "send X to this address" text
           HAS DISAPPEARED, AND
       (2) a positive indicator appears: success-class element, a checkmark
           icon, or text like 'received', 'confirmed', 'success', 'thank you',
           'completed', 'deposited'.

    Either signal is enough to consider deposit confirmed (we OR them, then
    AND with a sanity-check that we're not looking at an error page).

    Returns True on confirmation, False on timeout / error page.
    """
    logger.info(f"Waiting for deposit confirmation (timeout={timeout_seconds}s)...")
    deadline = time.time() + max(30, int(timeout_seconds))
    last_state = None
    poll = 0
    while time.time() < deadline:
        poll += 1
        try:
            state = page.evaluate(
                """() => {
                    const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
                    const bodyText = norm(document.body ? document.body.innerText : '');
                    const lo = bodyText.toLowerCase();

                    // (a) deposit-block visible? Look for QR canvas/img,
                    //     wallet address (long hex/base58 string), or a
                    //     "pending" / "waiting" / "send to this address" hint.
                    const hasQR = !!document.querySelector(
                        'canvas, img[src*="qr"], img[alt*="QR" i], svg[class*="qr" i], [class*="qr-code"], [class*="qrcode"]');
                    // crude wallet-address regex: 20+ alphanumeric chars with
                    // at least one digit and one letter, on its own (no spaces)
                    const addrRx = /\\b[a-zA-Z0-9]{25,}\\b/;
                    const hasAddress = addrRx.test(bodyText);
                    const hasPending = /(awaiting|pending|waiting for (the )?(payment|transaction|deposit)|send (the )?(exact|payment|amount)|deposit address|send to (this|the following))/i.test(bodyText);
                    const hasCopyBtn = !!document.querySelector(
                        '[class*="copy"], button[title*="copy" i], [aria-label*="copy" i]');

                    const depositBlockVisible = hasQR || hasAddress || hasPending || hasCopyBtn;

                    // (b) positive success indicator?
                    const successText = /(deposit (was )?(received|confirmed|successful|completed)|payment (was )?(received|confirmed|successful|completed)|transaction (was )?(received|confirmed|successful|completed)|thank you|balance (has been )?(topped up|credited|updated)|funds received)/i.test(bodyText);
                    const successClass = !!document.querySelector(
                        '.success, .is-success, [class*="success"], [class*="confirmed"], [class*="completed"], [class*="received"], [class*="paid"]');
                    // big green checkmark - look for SVG paths drawing a check,
                    // or a font-awesome / lucide check icon, with green color
                    // applied via CSS.
                    const checkmark = (() => {
                        const candidates = Array.from(document.querySelectorAll(
                            'svg, i.fa-check, i.fa-circle-check, i[class*="check"], [class*="check"][class*="mark"], [class*="checkmark"]'
                        ));
                        for (const el of candidates) {
                            try {
                                const cs = getComputedStyle(el);
                                const color = (cs.color || '') + ' ' + (cs.fill || '') + ' ' + (cs.stroke || '');
                                // crude green check: any rgb(.., >=128, ..) where green > red and green > blue.
                                const m = color.match(/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                                if (m) {
                                    const r = +m[1], g = +m[2], b = +m[3];
                                    if (g >= 120 && g > r && g > b) return true;
                                }
                                // class-based green hint
                                const cls = (el.className && el.className.toString)
                                              ? el.className.toString() : '';
                                if (/(green|success|emerald)/i.test(cls)) return true;
                            } catch (e) {}
                        }
                        return false;
                    })();

                    // Hard "error" signals - if the page shows an explicit
                    // error/expired/cancelled state, we should NOT report
                    // confirmed.
                    const hasError = /(expired|cancell?ed|failed|rejected|not received yet|insufficient|under-paid|underpayment)/i.test(bodyText);

                    return {
                        depositBlockVisible: depositBlockVisible,
                        successText: successText,
                        successClass: successClass,
                        checkmark: checkmark,
                        hasError: hasError,
                        urlPath: location.pathname,
                        textLen: bodyText.length,
                    };
                }"""
            )
        except Exception as e:
            logger.debug(f"  deposit-state probe failed: {e}")
            time.sleep(poll_interval)
            continue

        if state != last_state:
            logger.info(f"  deposit state poll #{poll}: {state}")
            last_state = state

        if state.get("hasError"):
            logger.error(f"  Deposit page shows an ERROR state: {state}")
            if debug_dir:
                page.screenshot(path=os.path.join(debug_dir, "17_deposit_error.png"),
                                full_page=True)
            return False

        # Confirmed = success indicator present AND (deposit block gone OR
        # explicit success text). We're strict here so we don't false-positive
        # on the initial page render (which has a green padlock somewhere).
        confirmed = (
            state.get("successText")
            or (state.get("checkmark") and not state.get("depositBlockVisible"))
            or (state.get("successClass") and not state.get("depositBlockVisible") and state.get("checkmark"))
        )
        if confirmed:
            logger.success(f"  Deposit CONFIRMED at poll #{poll}: {state}")
            if debug_dir:
                page.screenshot(path=os.path.join(debug_dir, "17_deposit_confirmed.png"),
                                full_page=True)
            return True

        time.sleep(poll_interval)

    logger.warning(f"Deposit confirmation timed out after {timeout_seconds}s.")
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "17_deposit_timeout.png"),
                        full_page=True)
    return False


def do_quick_verification(page, timeout_ms=4000, wait_after=7.0, debug_dir=None,
                          screenshot_prefix=None):
    """Detect & dismiss Styx's 'Quick verification' challenge (the credit-card
    slider + Continue button) if it's currently shown.

    Styx fires this challenge in two situations:
      1. First visit to the site (handled inline in process_registration).
      2. Subsequent navigations to a page after a long idle period (e.g.
         after we've been polling /wallet/deposit/... for several minutes
         waiting for blockchain confirmation). The user hit this when
         landing on the seller URL: the page shows "Quick verification /
         Please complete a short check to continue to the site / Continue".
         Previously we mis-classified this as a login page.

    Returns True if the challenge was dismissed (or wasn't present);
    False only if it WAS present but we failed to click Continue.
    """
    try:
        # Multi-signal: the title says "Quick verification" / "Please
        # complete a short check" AND there's a Continue button.
        sig = page.evaluate(
            """() => {
                const text = (document.body ? document.body.innerText : '').toLowerCase();
                const hasTitle = /quick verification|short check to continue|please complete a short check/i.test(text);
                const hasContinueText = /your connection will be encrypted/i.test(text);
                return { hasTitle, hasContinueText, len: text.length };
            }"""
        )
    except Exception:
        sig = None

    looks_like_quickverify = bool(sig) and (sig.get("hasTitle") or sig.get("hasContinueText"))
    if not looks_like_quickverify:
        return True  # nothing to dismiss

    logger.info(f"Quick verification challenge detected: {sig}")
    if debug_dir and screenshot_prefix:
        try:
            page.screenshot(path=os.path.join(debug_dir, f"{screenshot_prefix}_quickverify.png"),
                            full_page=True)
        except Exception:
            pass
    try:
        cont = page.locator(
            "button:has-text('Continue'), .continue-btn, #continue"
        ).first
        if cont.is_visible(timeout=timeout_ms):
            # Small human-like delay before clicking.
            time.sleep(random.uniform(1.0, 2.0))
            cont.click()
            logger.info("Clicked Quick verification 'Continue'.")
            # The loading-bar animation is ~5-7s, then the page navigates.
            time.sleep(wait_after)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            return True
        logger.warning("Quick verification shown but Continue button not visible.")
        return False
    except Exception as e:
        logger.warning(f"Failed to click Quick verification Continue: {e}")
        return False


def _is_login_page(page):
    """Return True if the current page looks like a login form.
    We use multiple weak signals OR'd together so we catch all of:
      - URL contains /login/, /signin/, /accounts/login
      - There's a password input that's NOT inside a registration form
      - The page title / body text says "Log in" / "Sign in" prominently
    """
    try:
        sig = page.evaluate(
            """() => {
                const url = (location.pathname + location.search).toLowerCase();
                const inLoginUrl = /\\/(login|sign[-_]?in|signin)(\\/|$|\\?)/.test(url);

                // Password field with no register/sign-up fields next to it.
                const pw = document.querySelector('input[type="password"]');
                let hasPwOnly = false;
                if (pw) {
                    const form = pw.closest('form');
                    if (form) {
                        const inputs = form.querySelectorAll('input');
                        // Registration forms typically have >=3 inputs (user
                        // + pw + secret/confirm), login has 2.
                        const text = (form.textContent || '').toLowerCase();
                        const looksRegister = /register|sign[-_\\s]?up|create (an )?account/i.test(text);
                        hasPwOnly = inputs.length <= 3 && !looksRegister;
                    } else {
                        hasPwOnly = true;
                    }
                }

                const title = (document.title || '').toLowerCase();
                const body = (document.body ? document.body.innerText : '').slice(0, 600).toLowerCase();
                const loginText = /\\blog[\\s-]?in\\b|\\bsign[\\s-]?in\\b/.test(title)
                    || /forgot (your )?password|remember me|log in to your account|sign in to your account/.test(body);

                return { inLoginUrl, hasPwOnly, loginText,
                         url: location.pathname + location.search };
            }"""
        )
    except Exception as e:
        logger.debug(f"  _is_login_page probe failed: {e}")
        return False
    # At least two signals to be confident.
    score = int(bool(sig.get("inLoginUrl"))) + int(bool(sig.get("hasPwOnly"))) \
            + int(bool(sig.get("loginText")))
    if score >= 2:
        logger.warning(f"  Page looks like a login form: {sig}")
        return True
    return False


def do_login(page, base_url, username, password, debug_dir=None):
    """Log in to Styx using known username/password. Used as recovery when
    a navigation lands us on the login page (e.g. session expired during
    the long blockchain confirmation wait).
    """
    login_url = base_url.rstrip("/") + "/accounts/login/"
    logger.info(f"Attempting re-login at: {login_url}  (user={username})")
    try:
        page.goto(login_url, wait_until="domcontentloaded", timeout=45000)
    except Exception as e:
        logger.error(f"  failed to open login page: {e}")
        return False
    time.sleep(random.uniform(1.0, 1.8))
    # Login page may ALSO trigger Quick Verification before showing the form.
    do_quick_verification(page, debug_dir=debug_dir, screenshot_prefix="29")
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "30_login_form.png"),
                        full_page=True)

    ok_u = smart_fill(page,
        ["input[name='username']",
         "input[name='login']",
         "input[name='email']",
         "input[type='text']",
         "input.input__input"],
        username, label="login_username")
    ok_p = smart_fill(page,
        ["input[name='password']",
         "input[type='password']",
         "input.input__input[type='password']"],
        password, label="login_password")
    if not (ok_u and ok_p):
        logger.error(f"  could not fill login form (user={ok_u}, pw={ok_p})")
        return False

    # Submit. Try several strategies; first that succeeds wins.
    submitted = False
    submit_strategies = [
        lambda: page.get_by_role("button", name="Log in", exact=False).first.click(timeout=4000),
        lambda: page.get_by_role("button", name="Sign in", exact=False).first.click(timeout=4000),
        lambda: page.locator("button[type='submit']:visible").first.click(timeout=4000),
        lambda: page.locator("button:has-text('Log in'):visible").first.click(timeout=4000),
        lambda: page.locator("button:has-text('Sign in'):visible").first.click(timeout=4000),
        lambda: page.keyboard.press("Enter"),
    ]
    for i, s in enumerate(submit_strategies, 1):
        try:
            s()
            submitted = True
            logger.info(f"  -> submitted login form (strategy {i})")
            break
        except Exception as e:
            logger.debug(f"  login submit strategy {i} failed: {e}")
    if not submitted:
        logger.error("  could not submit login form.")
        return False

    # Wait for navigation away from /login/.
    try:
        page.wait_for_function(
            """() => !/\\/(login|sign[-_]?in|signin)(\\/|$|\\?)/.test(
                location.pathname + location.search)""",
            timeout=15000,
        )
    except Exception as e:
        logger.debug(f"  post-login navigation wait timed out: {e}")
    time.sleep(random.uniform(1.0, 1.6))
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "31_after_login.png"),
                        full_page=True)

    # Final verification: are we still on a login page?
    if _is_login_page(page):
        logger.error(f"  Re-login failed (still on login page). URL: {page.url}")
        return False
    logger.success(f"  Re-login succeeded. URL: {page.url}")
    return True


def do_buy_product(page, seller_url, product_name, debug_dir=None,
                   login_username=None, login_password=None):
    """Navigate to a seller's profile page, find a product by name, add it to
    the cart via the row's small shopping-cart icon, then open the header
    cart, click Buy, and confirm Yes.

    Args:
        page: Playwright Page (already logged in).
        seller_url: seller profile URL, e.g.
            "https://styxmarket.si/accounts/profile/SCENARIO/?vue=true&user_id=28868"
        product_name: visible product label, e.g. "Firstmail.ltd E-Mail Accounts".
        debug_dir: optional dir for screenshots.
        login_username, login_password: kept for backward-compatibility with
            the caller; no longer used (the previous "auto re-login as first
            recovery strategy" was removed per user request — clicking the
            Quick Verification 'Continue' button is what kills the session, so
            we now recover by avoiding it entirely: go_back first, catalog-
            warm second).

    Returns True on success.
    """
    logger.info(f"Navigating to seller page for purchase: {seller_url}")
    try:
        page.goto(seller_url, wait_until="domcontentloaded", timeout=45000)
    except Exception as e:
        logger.error(f"Failed to open seller page: {e}")
        return False
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass
    time.sleep(random.uniform(1.5, 2.5))

    # ----- Quick Verification slider recovery --------------------------------
    # Styx re-runs its "Quick verification" challenge (credit-card slider +
    # Continue button) on this navigation because the previous page was
    # idling for several minutes during the blockchain-confirmation wait.
    #
    # IMPORTANT (field-verified): clicking the slider's "Continue" button
    # invalidates the session and redirects us to /accounts/login/ - a full
    # session reset. So we DEFINITELY DO NOT click Continue here. Instead we
    # nudge the browser into a different navigation state and re-try the
    # seller URL, hoping Styx no longer issues the challenge.
    #
    # Recovery ladder (no re-login, no Continue click - per user spec):
    #   1. page.go_back() -> scroll to top -> click the big Styx logo
    #      (warm-navigates to the homepage while keeping the session) ->
    #      goto seller_url.
    #   2. goto the public catalog page, then re-goto seller_url.
    # If both fail we bail out and leave the browser open for manual handling.
    def _slider_or_login():
        """Return True if the current page is the QV slider OR a login wall."""
        try:
            sig = page.evaluate(
                """() => {
                    const text = (document.body ? document.body.innerText : '').toLowerCase();
                    const hasTitle = /quick verification|short check to continue|please complete a short check/i.test(text);
                    const hasContinueText = /your connection will be encrypted/i.test(text);
                    return hasTitle || hasContinueText;
                }"""
            )
        except Exception:
            sig = False
        if sig:
            return True
        return _is_login_page(page)

    def _click_styx_logo():
        """Find & click the big Styx wordmark / logo in the top-left header
        to navigate to the homepage while keeping the current session warm.

        Returns True on a successful click that left the seller URL, False
        otherwise. We pick the FIRST visible top-of-page anchor whose href
        is the site root (/), whose alt/text/class hints at "logo" or
        "styx", and which sits in the upper ~120px of the viewport.
        """
        try:
            box = page.evaluate(
                """() => {
                    const norm = s => (s || '').toLowerCase();
                    const cands = Array.from(document.querySelectorAll(
                        'a[href="/"], a[href$="://styxmarket.si/"], a[href$="://styxmarket.si"], '
                        + 'a[href$="://www.styxmarket.si/"], a[href$="://www.styxmarket.si"], '
                        + 'a.logo, a[class*="logo" i], header a, nav a'
                    ));
                    let best = null;
                    let bestScore = -1;
                    for (const a of cands) {
                        try {
                            const r = a.getBoundingClientRect();
                            if (r.width < 20 || r.height < 16) continue;
                            if (r.top > 160 || r.left > window.innerWidth * 0.5) continue;
                            const cs = getComputedStyle(a);
                            if (cs.visibility === 'hidden' || cs.display === 'none'
                                || parseFloat(cs.opacity) === 0) continue;
                            let score = 0;
                            const cls = norm(a.className && a.className.toString ? a.className.toString() : '');
                            const txt = norm(a.innerText || a.textContent || '');
                            const inner = norm(a.innerHTML || '');
                            if (/logo/.test(cls)) score += 50;
                            if (/styx/.test(txt)) score += 30;
                            if (/styx/.test(inner)) score += 20;
                            if (a.querySelector('img, svg')) score += 15;
                            const href = (a.getAttribute('href') || '').toLowerCase();
                            if (href === '/' || /styxmarket\\.si\\/?$/.test(href)) score += 40;
                            // upper-left bias
                            score += Math.max(0, 60 - r.top) * 0.5;
                            score += Math.max(0, 100 - r.left) * 0.3;
                            if (score > bestScore) {
                                bestScore = score;
                                best = { x: r.left + r.width / 2, y: r.top + r.height / 2,
                                         href: href, score: score };
                            }
                        } catch (e) {}
                    }
                    return best;
                }"""
            )
        except Exception as e:
            logger.debug(f"  logo-locator JS failed: {e}")
            return False

        if not box:
            logger.warning("  could not locate the Styx logo in the header.")
            return False

        logger.info(
            f"  clicking Styx logo at ({box['x']:.0f}, {box['y']:.0f}) "
            f"href={box.get('href')} score={box.get('score'):.0f}")
        url_before = page.url
        try:
            page.mouse.click(float(box["x"]), float(box["y"]))
        except Exception as e:
            logger.debug(f"  mouse.click on logo failed: {e}")
            return False
        try:
            page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass
        time.sleep(random.uniform(1.0, 1.8))
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        # Success if URL changed AND we are no longer on the slider/login.
        moved = page.url != url_before
        if moved and not _slider_or_login():
            logger.success(f"  logo click navigated us to: {page.url}")
            return True
        logger.warning(
            f"  logo click did not warm-navigate (url_before={url_before}, "
            f"now={page.url}, slider/login={_slider_or_login()}).")
        return False

    if _slider_or_login():
        logger.warning(
            "Quick verification slider (or login wall) detected on seller "
            "page. Avoiding the 'Continue' click (which kills the session) "
            "and falling back to the marketplace 'buy any cheap product' "
            "flow per user spec."
        )
        if debug_dir:
            try:
                page.screenshot(
                    path=os.path.join(debug_dir, "19_slider_detected.png"),
                    full_page=True,
                )
            except Exception:
                pass

        recovered = False

        # ----- MARKETPLACE FALLBACK ------------------------------------------
        # Per user 2026-06-19: when the slider blocks the seller page, give
        # up on the user-requested product. Instead:
        #   1. go_back to the deposit-success page (which is still logged in
        #      and not behind the slider).
        #   2. Click the big Styx logo to warm-navigate to the homepage,
        #      which IS the marketplace.
        #   3. Type "1" into the "$ Max" filter input on the left sidebar.
        #   4. Wait for the product list to filter (max $1 products only).
        #   5. Pick the first product whose visible price is <= $0.50 (or
        #      <= $1 as a fallback - the $1 cap is already enforced by step
        #      3 so anything in the filtered list qualifies as "cheap").
        #   6. Override `product_name` with the picked product's visible
        #      label, set recovered=True, and fall through to the existing
        #      find-row + click-cart + open-cart + buy + confirm code.
        logger.info(
            "  marketplace fallback: go_back -> click logo -> filter $Max=1 "
            "-> pick cheap product -> continue with existing buy flow."
        )

        # Step 1: go_back to deposit success page.
        try:
            page.go_back(wait_until="domcontentloaded", timeout=15000)
            time.sleep(random.uniform(1.5, 2.5))
        except Exception as e:
            logger.debug(f"  go_back failed: {e}")
        try:
            page.evaluate(
                "() => window.scrollTo({ top: 0, behavior: 'instant' })")
        except Exception:
            try:
                page.evaluate("() => window.scrollTo(0, 0)")
            except Exception as e:
                logger.debug(f"  scroll-to-top after go_back failed: {e}")
        time.sleep(random.uniform(0.6, 1.0))
        if debug_dir:
            try:
                page.screenshot(
                    path=os.path.join(debug_dir, "19a_after_goback.png"),
                    full_page=False,
                )
            except Exception:
                pass

        # Step 2: click the Styx logo -> homepage (marketplace).
        logo_ok = _click_styx_logo()
        if not logo_ok:
            # Fallback: hard-navigate to the homepage.
            logger.warning(
                "  logo click failed; hard-navigating to the homepage.")
            try:
                page.goto("https://www.styxmarket.si/",
                          wait_until="domcontentloaded", timeout=30000)
                time.sleep(random.uniform(1.5, 2.5))
            except Exception as e:
                logger.debug(f"  hard goto homepage failed: {e}")
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        if debug_dir:
            try:
                page.screenshot(
                    path=os.path.join(debug_dir, "19b_after_logo_click.png"),
                    full_page=False,
                )
            except Exception:
                pass
        # Bail if the slider is STILL up - logo click didn't help.
        if _slider_or_login():
            logger.error(
                "  marketplace fallback: slider/login STILL showing after "
                "logo click. Aborting recovery."
            )
            if debug_dir:
                try:
                    page.screenshot(
                        path=os.path.join(debug_dir, "19_slider_still_up.png"),
                        full_page=True,
                    )
                except Exception:
                    pass
            return False

        # Step 3: type "1" into the "$ Max" filter input.
        # The filter sidebar shows two inputs side by side: $ Min and $ Max.
        # We want the SECOND one. We try a strict positional locator first
        # (sibling of a "$ Min" input, or the input directly after one with
        # placeholder containing 'Min'), and fall back to any visible input
        # whose placeholder contains 'Max' or which sits below a "PRICE"
        # label.
        logger.info("  applying $ Max = 1 filter...")
        max_filter_applied = page.evaluate(
            """() => {
                const norm = s => (s || '').toLowerCase();
                // Find every visible input.
                const inputs = Array.from(document.querySelectorAll('input'));
                const visible = inputs.filter(el => {
                    const r = el.getBoundingClientRect();
                    if (r.width < 10 || r.height < 10) return false;
                    const cs = getComputedStyle(el);
                    return cs.visibility !== 'hidden' && cs.display !== 'none'
                        && parseFloat(cs.opacity) !== 0;
                });
                // Helper to set value via native setter so Vue updates.
                function setVal(el, v) {
                    const proto = window.HTMLInputElement.prototype;
                    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                    setter.call(el, v);
                    el.dispatchEvent(new Event('input',  { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    el.dispatchEvent(new Event('blur',   { bubbles: true }));
                }
                // Strategy A: placeholder contains 'max' (case-insensitive),
                // ignore the search bar ("What do you need today?").
                let target = visible.find(el => {
                    const ph = norm(el.placeholder || '');
                    return ph.includes('max') && !ph.includes('today');
                });
                // Strategy B: a "$ Min" input exists; pick its right-neighbour.
                if (!target) {
                    const minInp = visible.find(el => norm(el.placeholder || '').includes('min'));
                    if (minInp) {
                        const minRect = minInp.getBoundingClientRect();
                        target = visible.find(el => {
                            if (el === minInp) return false;
                            const r = el.getBoundingClientRect();
                            // same row (overlap on Y), to the right of $ Min,
                            // close in X.
                            return Math.abs(r.top - minRect.top) < 20
                                && r.left > minRect.left
                                && r.left - (minRect.left + minRect.width) < 60;
                        });
                    }
                }
                // Strategy C: any sibling input under a PRICE-labelled section.
                if (!target) {
                    const priceHdrs = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6,div,span,p'))
                        .filter(el => /price/i.test((el.innerText||el.textContent||'').trim()));
                    for (const hdr of priceHdrs) {
                        let scope = hdr.parentElement;
                        for (let i = 0; i < 5 && scope; i++) {
                            const ins = Array.from(scope.querySelectorAll('input'))
                                .filter(el => visible.includes(el));
                            if (ins.length >= 2) {
                                // assume layout: [Min][Max] -> take second.
                                target = ins[1]; break;
                            }
                            scope = scope.parentElement;
                        }
                        if (target) break;
                    }
                }
                if (!target) {
                    return { ok: false, reason: 'no $ Max input found',
                             visibleCount: visible.length,
                             placeholders: visible.map(el => el.placeholder).slice(0, 10) };
                }
                target.scrollIntoView({behavior: 'instant', block: 'center'});
                target.focus();
                setVal(target, '1');
                return { ok: true, placeholder: target.placeholder || '',
                         x: target.getBoundingClientRect().left,
                         y: target.getBoundingClientRect().top };
            }"""
        )
        logger.info(f"  $ Max filter result: {max_filter_applied}")
        if not (max_filter_applied and max_filter_applied.get("ok")):
            logger.warning(
                "  $ Max filter input not found; will try to pick a cheap "
                "product from the unfiltered list anyway.")
        # Press Enter on the focused input (some Vue forms only apply the
        # filter on Enter / blur). The setVal above already dispatches
        # blur, but Enter is a safe extra nudge.
        try:
            page.keyboard.press("Enter")
        except Exception:
            pass
        # Wait for the product list to refresh.
        time.sleep(2.5)
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        if debug_dir:
            try:
                page.screenshot(
                    path=os.path.join(debug_dir, "19c_after_max_filter.png"),
                    full_page=False,
                )
            except Exception:
                pass

        # Step 4-5: pick first product whose visible price is <= $0.50
        # (else <= $1, which is anything in the filtered list).
        logger.info("  scanning marketplace for cheap product (target <= $0.50, else <= $1)...")
        picked = page.evaluate(
            """() => {
                // A 'product row' shows a price like "$X.YY" on the right.
                // We scan all elements whose own text matches /^\\$[0-9.]+$/
                // (price labels), walk up to a row-shaped ancestor, then
                // capture both the row and its visible product name (the
                // truncated label like "F..." or the full visible name).
                const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
                const priceEls = Array.from(document.querySelectorAll('div, span, p, b, strong'))
                    .filter(el => {
                        const own = norm(Array.from(el.childNodes)
                            .filter(n => n.nodeType === 3)
                            .map(n => n.textContent).join(''));
                        return /^\\$\\s*\\d+(\\.\\d+)?$/.test(own);
                    });
                const results = [];
                for (const pe of priceEls) {
                    try {
                        const own = norm(Array.from(pe.childNodes)
                            .filter(n => n.nodeType === 3)
                            .map(n => n.textContent).join(''));
                        const price = parseFloat(own.replace(/[^0-9.]/g, ''));
                        if (!isFinite(price) || price <= 0) continue;
                        // Walk up to a row container (similar logic to the
                        // product-row finder).
                        let row = null, cur = pe.parentElement;
                        for (let i = 0; i < 8 && cur; i++) {
                            const r = cur.getBoundingClientRect();
                            if (r.width > 500 && r.height > 40 && r.height < 250) {
                                row = cur; break;
                            }
                            cur = cur.parentElement;
                        }
                        if (!row) continue;
                        const rr = row.getBoundingClientRect();
                        // Skip rows that aren't currently in (or just below)
                        // the viewport - "SOLD OUT" rows often have weird
                        // layouts.
                        if (rr.bottom < 0) continue;
                        // Look for "SOLD OUT" badge inside the row.
                        const rowText = norm(row.innerText || row.textContent || '');
                        if (/sold out|out of stock/i.test(rowText)) continue;
                        // Capture a likely product-name element: the
                        // largest text node inside the row that isn't a
                        // price, badge, or number-of-items ("x55").
                        const labels = Array.from(row.querySelectorAll('div, span, p, h1, h2, h3, h4, h5, b, strong'))
                            .map(el => norm(Array.from(el.childNodes)
                                .filter(n => n.nodeType === 3)
                                .map(n => n.textContent).join('')))
                            .filter(t => t && t.length >= 3 && t.length <= 80
                                       && !/^\\$\\s*\\d/.test(t)
                                       && !/^x\\d+$/i.test(t)
                                       && !/^high quality$|^verified$|^sold out$/i.test(t)
                                       && !/^\\d+$/.test(t));
                        // Heuristic: pick the LONGEST visible label - that's
                        // usually the product name (e.g. "Microsoft Outlook
                        // mails Selfreg Type-1") vs short badges.
                        labels.sort((a, b) => b.length - a.length);
                        const name = labels[0] || '';
                        if (!name) continue;
                        results.push({
                            price, name,
                            rowX: rr.left + rr.width / 2,
                            rowY: rr.top + rr.height / 2,
                        });
                    } catch (e) {}
                }
                // Sort by ascending price, then prefer the first <= 0.50.
                results.sort((a, b) => a.price - b.price);
                return { count: results.length,
                         items: results.slice(0, 10) };
            }"""
        )
        logger.info(f"  marketplace scan: {picked}")
        if not picked or not picked.get("items"):
            logger.error(
                "  marketplace fallback: could not find any product rows "
                "with a visible price. Bailing out.")
            if debug_dir:
                try:
                    page.screenshot(
                        path=os.path.join(debug_dir, "19_no_products.png"),
                        full_page=True,
                    )
                except Exception:
                    pass
            return False

        items = picked["items"]
        # Prefer <= $0.50, fall back to cheapest available.
        cheap = [it for it in items if it["price"] <= 0.50]
        chosen = cheap[0] if cheap else items[0]
        logger.success(
            f"  marketplace fallback chose: '{chosen['name']}' "
            f"@ ${chosen['price']:.2f}"
        )
        # Scroll the chosen row into view so the existing find-row code
        # can locate it.
        try:
            page.evaluate(
                "([y]) => window.scrollTo({top: window.scrollY + y - window.innerHeight/2, behavior: 'instant'})",
                [chosen["rowY"]],
            )
            time.sleep(0.6)
        except Exception as e:
            logger.debug(f"  scroll-to-cheap-row failed: {e}")
        if debug_dir:
            try:
                page.screenshot(
                    path=os.path.join(debug_dir, "19d_picked_cheap_row.png"),
                    full_page=False,
                )
            except Exception:
                pass
        # OVERRIDE product_name so the existing find-row + cart-icon code
        # finds THIS product instead of the original user-requested one.
        product_name = chosen["name"]
        recovered = True

        if not recovered:
            logger.error(
                "Marketplace fallback failed. The browser stays open so "
                "you can complete it manually."
            )
            return False

    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "20_seller_page.png"),
                        full_page=True)

    # --- 1) Find the product row & click its cart-icon -----------------------
    # Strategy: find any element whose visible text contains the product name,
    # walk up to the nearest "row" container (the seller's product card), then
    # find a cart-shaped element inside it. The cart icon is typically:
    #   <svg class="...cart..."> or <i class="...cart...">, or a button with
    #   data-tip / aria-label / title containing "cart" / "add to cart".
    logger.info(f"Locating product row: '{product_name}'")
    found = page.evaluate(
        """({label}) => {
            const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
            const want = norm(label).toLowerCase();

            // 1) Find any element whose visible text starts-with or contains
            //    the product name. Truncated text (e.g. "Firstmail.ltd E-Mail
            //    Ac...") needs prefix match too.
            const all = Array.from(document.querySelectorAll('div, span, p, a, h1, h2, h3, h4, h5, button'));
            const wantPrefix = want.slice(0, Math.min(want.length, 18));
            let matched = all.find(el => {
                const own = norm(Array.from(el.childNodes)
                    .filter(n => n.nodeType === 3)
                    .map(n => n.textContent).join('')).toLowerCase();
                if (!own) return false;
                return own === want
                    || own.startsWith(wantPrefix)
                    || own.includes(wantPrefix);
            });
            if (!matched) {
                // Fallback: descendant text match anywhere in the tree.
                matched = all.find(el => {
                    const t = norm(el.textContent || '').toLowerCase();
                    return t.includes(wantPrefix) && t.length < want.length + 80;
                });
            }
            if (!matched) {
                const labels = all.filter(el => {
                    const own = norm(Array.from(el.childNodes)
                        .filter(n => n.nodeType === 3)
                        .map(n => n.textContent).join(''));
                    return own && own.length > 6 && own.length < 80;
                }).map(el => norm(Array.from(el.childNodes)
                        .filter(n => n.nodeType === 3)
                        .map(n => n.textContent).join('')))
                  .slice(0, 30);
                return { ok: false, reason: 'no product text match', sampleLabels: labels };
            }

            // 2) Walk up to the row container. Prefer ancestors with class
            //    containing 'product', 'card', 'item', or 'row'. We REQUIRE
            //    moving up at least one level (the matched element itself is
            //    the product name/label, not the row) and that the candidate
            //    is meaningfully wider than the matched text (the actual row
            //    spans the full content width, ~600px+, not just the label).
            const rowRx = /product|card|item|row|tile|listing/i;
            const matchedRect = matched.getBoundingClientRect();
            let row = null;
            let cur = matched.parentElement;  // skip matched itself
            for (let i = 0; i < 8 && cur; i++) {
                const cls = (cur.className && cur.className.toString)
                                ? cur.className.toString() : '';
                const rr = cur.getBoundingClientRect();
                // Filter out per-cell wrappers like .product-name / .item-title
                // which would otherwise match the rowRx. The real row is at
                // least ~1.5x wider than the matched label.
                if (rowRx.test(cls)
                    && rr.width > Math.max(400, matchedRect.width * 1.3)
                    && rr.height >= 30 && rr.height < 250) {
                    row = cur;
                    break;
                }
                cur = cur.parentElement;
            }
            // Fallback: take a generous-sized ancestor (whole product row is
            // usually 800+px wide on desktop).
            if (!row) {
                cur = matched.parentElement;
                for (let i = 0; i < 8 && cur; i++) {
                    const r = cur.getBoundingClientRect();
                    if (r.width > 500 && r.height > 40 && r.height < 250
                        && r.width > matchedRect.width * 1.3) {
                        row = cur; break;
                    }
                    cur = cur.parentElement;
                }
            }
            if (!row) return { ok: false, reason: 'no row ancestor',
                                productText: norm(matched.textContent).slice(0, 80) };

            // 3) Find a cart-shaped element INSIDE this row, not the header.
            //
            // KEY LESSON from a previous run: the mail/envelope ("Write
            // seller") icon sometimes won over the cart because the previous
            // SVG-path fallback was too permissive (it returned true for any
            // SVG with a long `d` attribute, which matches every icon).
            // The new logic:
            //   - EXCLUDE obvious non-cart icons (mail/envelope/message/star/
            //     heart/info/share/copy/external/link/bell/notification).
            //   - INCLUDE cart-related tokens explicitly.
            //   - Returns a RANKED LIST of candidates; the Python caller
            //     clicks each in turn and verifies by header-cart-badge
            //     increment, falling through if the wrong one was picked.
            const includeRx = /(^|[\\s_-])cart([\\s_-]|$)|shopping[-_]?cart|add[-_]?to[-_]?cart|buy[-_]?now|basket|trolley/i;
            const excludeRx = /mail|envelope|message|chat|letter|email|inbox|star|fav(ou)?rite|heart|like|bookmark|info|question|tooltip|share|copy|external[-_]?link|link[-_]?icon|bell|notification|delete|remove|trash|edit|menu|hamburger|search|filter|sort|user[-_]?avatar|profile/i;

            const allEls = Array.from(row.querySelectorAll('button, a, svg, i, span, div'));
            const idOf = (el) => {
                const cls = (el.className && el.className.toString)
                                ? el.className.toString() : '';
                const al = el.getAttribute && (
                       el.getAttribute('aria-label')
                    || el.getAttribute('title')
                    || el.getAttribute('data-tip')
                    || el.getAttribute('data-tooltip')
                    || el.getAttribute('data-original-title')
                    || '');
                return { cls, al, allText: cls + ' ' + (al || '') };
            };
            const isExcluded = (el) => excludeRx.test(idOf(el).allText);
            const isIncluded = (el) => includeRx.test(idOf(el).allText);
            const isClickableShape = (el) => {
                const r = el.getBoundingClientRect();
                if (r.width < 8 || r.width > 100 || r.height < 8 || r.height > 100) return false;
                return true;
            };

            // Walk-up helper: when we found an SVG/inner icon, prefer its
            // closest <button>/<a> ancestor for a more reliable click.
            const toClickable = (el) => {
                if (!el) return el;
                let cur = el;
                for (let i = 0; i < 4; i++) {
                    if (!cur || !cur.parentElement) break;
                    const p = cur.parentElement;
                    const tag = p.tagName && p.tagName.toLowerCase();
                    if (tag === 'button' || tag === 'a' || (p.onclick != null)) {
                        return p;
                    }
                    cur = p;
                }
                return el;
            };

            // Tier 1: explicit cart matches that are NOT excluded.
            let tier1 = allEls.filter(el =>
                isClickableShape(el) && isIncluded(el) && !isExcluded(el));

            // Tier 2: non-excluded, icon-shaped clickables (last-resort -
            // pick by position, leftmost wins). Used only if tier 1 empty.
            let tier2 = [];
            if (tier1.length === 0) {
                tier2 = allEls.filter(el => {
                    if (!isClickableShape(el)) return false;
                    if (isExcluded(el)) return false;
                    const tag = el.tagName && el.tagName.toLowerCase();
                    // Must look like an icon button: clickable wrapper, OR an
                    // svg/i with a clickable ancestor in the row.
                    if (tag === 'button' || tag === 'a') return true;
                    if (tag === 'svg' || tag === 'i') {
                        const wrap = toClickable(el);
                        return wrap && wrap !== el;
                    }
                    return false;
                });
                // Dedupe by clickable wrapper so we don't list both the svg
                // and its parent button.
                const seen = new Set();
                tier2 = tier2.map(toClickable).filter(el => {
                    if (seen.has(el)) return false;
                    seen.add(el);
                    return true;
                });
                // Sort left-to-right (cart is the first icon after the
                // product name on Styx).
                tier2.sort((a, b) => a.getBoundingClientRect().left
                                   - b.getBoundingClientRect().left);
            } else {
                // Also dedupe tier1 by clickable wrapper.
                const seen = new Set();
                tier1 = tier1.map(toClickable).filter(el => {
                    if (seen.has(el)) return false;
                    seen.add(el);
                    return true;
                });
            }

            const ranked = [...tier1, ...tier2];

            if (ranked.length === 0) {
                // Diagnostic dump: what's in this row?
                const inv = Array.from(row.querySelectorAll('*'))
                    .filter(el => {
                        const cls = (el.className && el.className.toString)
                                        ? el.className.toString() : '';
                        return cls && cls.length < 100;
                    })
                    .slice(0, 30)
                    .map(el => ({
                        tag: el.tagName,
                        cls: el.className.toString().slice(0, 80),
                        aria: el.getAttribute && el.getAttribute('aria-label'),
                        title: el.getAttribute && el.getAttribute('title'),
                        tip: el.getAttribute && (el.getAttribute('data-tip')
                                              || el.getAttribute('data-tooltip')
                                              || el.getAttribute('data-original-title')),
                    }));
                return { ok: false, reason: 'no cart-like icon in row',
                         productText: norm(matched.textContent).slice(0, 80),
                         rowCls: row.className.toString().slice(0, 100),
                         rowItems: inv };
            }

            try { ranked[0].scrollIntoView({block: 'center', inline: 'center'}); } catch (e) {}

            // Return the ranked list of candidates with their boxes & ids.
            // The Python caller will click them in order and verify via
            // header-cart-badge increment.
            const candidatesOut = ranked.slice(0, 6).map(el => {
                const r = el.getBoundingClientRect();
                const id = idOf(el);
                return {
                    x: r.left + r.width / 2,
                    y: r.top + r.height / 2,
                    w: r.width, h: r.height,
                    tag: el.tagName,
                    cls: id.cls.slice(0, 100),
                    aria: (id.al || '').slice(0, 80),
                };
            });
            const top = candidatesOut[0];
            return {
                ok: true,
                x: top.x, y: top.y, w: top.w, h: top.h,
                tag: top.tag, cls: top.cls, aria: top.aria,
                tier: tier1.length > 0 ? 1 : 2,
                productText: norm(matched.textContent).slice(0, 80),
                candidates: candidatesOut,
            };
        }""",
        {"label": product_name},
    )
    logger.info(f"  product lookup result: {found}")
    if not (found and found.get("ok")):
        logger.error(f"Could not find product '{product_name}' or its cart icon.")
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "21_no_product.png"),
                            full_page=True)
        return False

    # Read header-cart badge BEFORE clicking, so we can detect whether the
    # click actually added an item (vs clicked the wrong icon).
    def _read_cart_badge():
        try:
            return page.evaluate(
                """() => {
                    // Find a cart-like icon in the header area and look at
                    // its neighbouring number/badge.
                    const vw = window.innerWidth;
                    const cartRx = /(^|[\\s_-])cart([\\s_-]|$)|shopping[-_]?cart|basket|header-?cart/i;
                    const all = Array.from(document.querySelectorAll('a, button, span, div, i, svg'));
                    const headerCart = all.find(el => {
                        const r = el.getBoundingClientRect();
                        if (r.top > 140 || r.left < vw * 0.4) return false;
                        const cls = (el.className && el.className.toString) ? el.className.toString() : '';
                        const al = el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('title') || '') || '';
                        return cartRx.test(cls) || (al && cartRx.test(al));
                    });
                    if (!headerCart) return { ok: false, count: null };

                    // Find badge: a child / sibling number within the cart's
                    // bounding box, or a [class*="badge"] / [class*="count"]
                    // / superscript span nearby.
                    const parseInt10 = s => {
                        const m = (s || '').match(/\\d+/);
                        return m ? parseInt(m[0], 10) : null;
                    };
                    // 1) descendants
                    const desc = Array.from(headerCart.querySelectorAll('*'));
                    for (const d of desc) {
                        const cls = (d.className && d.className.toString) ? d.className.toString() : '';
                        if (/badge|count|cart[-_]?(num|count|amount)/i.test(cls)) {
                            const n = parseInt10(d.textContent);
                            if (n !== null) return { ok: true, count: n, src: 'desc-class' };
                        }
                    }
                    // 2) any descendant pure-number text
                    for (const d of desc) {
                        const own = (d.textContent || '').trim();
                        if (/^\\s*\\d+\\s*$/.test(own)) {
                            const n = parseInt10(own);
                            if (n !== null && n < 100) return { ok: true, count: n, src: 'desc-num' };
                        }
                    }
                    // 3) sibling badge (badge floats next to the cart icon)
                    const parent = headerCart.parentElement;
                    if (parent) {
                        const sibDesc = Array.from(parent.querySelectorAll('*'));
                        for (const d of sibDesc) {
                            const cls = (d.className && d.className.toString) ? d.className.toString() : '';
                            if (/badge|count/i.test(cls)) {
                                const n = parseInt10(d.textContent);
                                if (n !== null) return { ok: true, count: n, src: 'sib-class' };
                            }
                        }
                    }
                    // No badge visible -> assume count is 0.
                    return { ok: true, count: 0, src: 'no-badge' };
                }"""
            )
        except Exception as e:
            logger.debug(f"  read cart badge failed: {e}")
            return None

    before_badge = _read_cart_badge()
    before_n = (before_badge or {}).get("count") if before_badge else None
    logger.info(f"  header cart badge BEFORE click: {before_badge}")

    # Click candidates in ranked order. Verification strategy:
    #
    # The header cart badge on the seller-profile page does NOT auto-refresh
    # after add-to-cart on Styx (the badge only updates on next navigation).
    # So we can't rely on `badge.count++` as a synchronous verification - it
    # often stays the same even after a successful add.
    #
    # Strategy:
    #   * Tier 1 = `class*=cart` / `aria*=cart` candidate. The heuristic is
    #     strict (excludes mail/star/etc), so a tier-1 hit is basically
    #     guaranteed correct. We click, wait long enough for the server to
    #     process (~6s), then proceed.
    #   * Tier 2 = positional fallback (no cart-class match). Here we DO
    #     need to verify, because we don't know we picked right. We poll
    #     the badge for up to 20s and require a real increment, otherwise
    #     dismiss any popup and try the next candidate.
    #   * Failures in the header-cart step (next phase) catch the rare case
    #     where the tier-1 click was visually right but the server rejected
    #     it (out of stock, etc.) - we'll see "cart is empty" there.
    candidates = found.get("candidates") or [
        {"x": found.get("x"), "y": found.get("y"),
         "tag": found.get("tag"), "cls": found.get("cls"),
         "aria": found.get("aria"), "w": found.get("w"), "h": found.get("h")}
    ]
    tier = found.get("tier", 1)
    logger.info(
        f"  ranked candidates (tier {tier}) for '{product_name}': {candidates}"
    )

    add_ok = False
    for idx, cand in enumerate(candidates, 1):
        try:
            jx = float(cand["x"]) + random.uniform(-2.0, 2.0)
            jy = float(cand["y"]) + random.uniform(-2.0, 2.0)
            page.mouse.move(jx, jy, steps=random.randint(8, 14))
            time.sleep(random.uniform(0.1, 0.25))
            page.mouse.click(jx, jy, delay=random.randint(40, 90))
            logger.info(
                f"  -> tried candidate #{idx} at ({jx:.1f},{jy:.1f}) "
                f"tag={cand.get('tag')} cls='{(cand.get('cls') or '')[:60]}' "
                f"aria='{(cand.get('aria') or '')[:40]}'"
            )
        except Exception as e:
            logger.debug(f"  click on candidate #{idx} failed: {e}")
            continue

        if tier == 1:
            # Strict cart-class match - trust it. Just wait for the server
            # so the next step (opening the header cart) sees fresh state.
            time.sleep(random.uniform(6.0, 8.0))
            after_badge = _read_cart_badge()
            logger.info(
                f"  tier-1 click on candidate #{idx}: trusting the heuristic. "
                f"badge (may be stale): {after_badge}"
            )
            add_ok = True
            break

        # tier 2: positional fallback - we need to verify. Poll the badge.
        poll_deadline = time.time() + 20.0
        last_after_badge = None
        added = False
        while time.time() < poll_deadline:
            time.sleep(0.6)
            last_after_badge = _read_cart_badge()
            after_n = (last_after_badge or {}).get("count") if last_after_badge else None
            if (isinstance(after_n, int) and isinstance(before_n, int)
                    and after_n > before_n):
                added = True
                break
            if (after_n is not None and after_n >= 1
                  and (before_n is None or before_n == 0)):
                added = True
                break
        after_badge = last_after_badge
        after_n = (after_badge or {}).get("count") if after_badge else None
        logger.info(
            f"  tier-2 badge AFTER candidate #{idx} (polled up to 20s): {after_badge}"
        )

        if added:
            add_ok = True
            logger.success(
                f"  Item added to cart (badge {before_n} -> {after_n}) via "
                f"candidate #{idx} tag={cand.get('tag')} "
                f"cls='{(cand.get('cls') or '')[:60]}'"
            )
            before_n = after_n
            break

        # No increment -> wrong icon was likely clicked. Try the next one.
        try:
            page.keyboard.press("Escape")
            time.sleep(0.3)
            page.keyboard.press("Escape")
        except Exception:
            pass
        logger.info(
            f"  candidate #{idx} did not increment cart badge "
            f"(before={before_n}, after={after_n}); trying next candidate."
        )

    if not add_ok:
        logger.error(
            f"Tried {len(candidates)} candidate(s) but none looked added. "
            f"Cannot reliably add '{product_name}' to cart."
        )
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "22b_no_add_to_cart.png"),
                            full_page=True)
        return False

    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "22_added_to_cart.png"),
                        full_page=True)

    # --- 2) Open the header cart ---------------------------------------------
    # The site header (top-right) has icons:  flag | mail | bell | CART | menu.
    # The cart icon usually carries a badge with the item count.
    #
    # IMPORTANT: scroll to the top of the page FIRST. The previous step may
    # have scrolled the product row into view (Styx's product list is long,
    # the row was at y~368), and Styx's header is not always position:fixed -
    # so it can be entirely off-screen by the time we look for it. Without
    # this scroll, `getBoundingClientRect()` returns negative top/bottom and
    # our viewport filter kicks the header cart out.
    logger.info("Opening header cart...")
    try:
        page.evaluate("() => window.scrollTo({ top: 0, behavior: 'instant' })")
    except Exception:
        # 'instant' isn't supported in all engines; fall back to default.
        try:
            page.evaluate("() => window.scrollTo(0, 0)")
        except Exception as e:
            logger.debug(f"  scroll-to-top failed (non-fatal): {e}")
    time.sleep(random.uniform(0.6, 1.0))
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "23a_scrolled_to_top.png"),
                        full_page=False)

    header_cart = page.evaluate(
        """() => {
            // Find an element in the top-area of the page (small y, large x)
            // that looks cart-shaped. We mirror the strict tier-1/tier-2
            // selection used for the row cart:
            //   * INCLUDE only elements whose class / aria / title /
            //     data-tip contains an explicit cart token.
            //   * EXCLUDE mail / star / heart / info / share / bell / etc.
            //     - critical because the Styx header has a mail icon (with
            //     its own count badge) right next to the cart, and we used
            //     to mis-pick it.
            //   * Among matches, pick the one closest to the TOP-RIGHT
            //     corner of the viewport.
            const vh = window.innerHeight;
            const vw = window.innerWidth;
            const includeRx = /(^|[\\s_-])cart([\\s_-]|$)|shopping[-_]?cart|basket|trolley|header[-_]?cart/i;
            const excludeRx = /mail|envelope|message|chat|letter|email|inbox|star|fav(ou)?rite|heart|like|bookmark|info|question|tooltip|share|copy|external[-_]?link|link[-_]?icon|bell|notification|delete|remove|trash|edit|menu|hamburger|search|filter|sort|user[-_]?avatar|profile|account[-_]?menu/i;
            const idOf = (el) => {
                const cls = (el.className && el.className.toString)
                                ? el.className.toString() : '';
                const al = el.getAttribute && (
                       el.getAttribute('aria-label') || el.getAttribute('title')
                    || el.getAttribute('data-tip')   || el.getAttribute('data-tooltip')
                    || el.getAttribute('data-original-title') || '');
                const href = el.getAttribute && (el.getAttribute('href') || '');
                return cls + ' ' + (al || '') + ' ' + (href || '');
            };
            const all = Array.from(document.querySelectorAll('a, button, svg, div, span, i'));

            // Top-area candidates: must be in the upper ~180px AND in the
            // right half of the screen. After our scrollTo(0,0) the header
            // should be at the top of the viewport.
            const inTopRight = (el) => {
                const r = el.getBoundingClientRect();
                if (r.top < 0 || r.top > 180) return false;
                if (r.bottom < 0) return false;
                if (r.left < vw * 0.4) return false;
                if (r.width < 8 || r.height < 8) return false;
                return true;
            };

            // Tier 1: explicit cart-tagged, not excluded, in top-right.
            const tier1 = all.filter(el => {
                if (!inTopRight(el)) return false;
                const t = idOf(el);
                return includeRx.test(t) && !excludeRx.test(t);
            });

            // Tier 2: ANY element in the top-right whose href contains
            // '/cart' or '/basket' - useful if cart is a link without
            // any cart-class but with a meaningful URL.
            const tier2 = all.filter(el => {
                if (!inTopRight(el)) return false;
                const href = (el.getAttribute && el.getAttribute('href')) || '';
                return /\\/cart|\\/basket|\\/checkout/i.test(href);
            });

            const sortByCornerDistance = (arr) => {
                arr.sort((a, b) => {
                    const ra = a.getBoundingClientRect();
                    const rb = b.getBoundingClientRect();
                    const da = Math.hypot(vw - ra.right, ra.top);
                    const db = Math.hypot(vw - rb.right, rb.top);
                    return da - db;
                });
                return arr;
            };

            // Walk up to the most-clickable ancestor (button/a) - the SVG
            // child usually isn't the click target.
            const toClickable = (el) => {
                let cur = el;
                for (let i = 0; i < 4 && cur && cur.parentElement; i++) {
                    const p = cur.parentElement;
                    const tag = (p.tagName || '').toLowerCase();
                    if (tag === 'button' || tag === 'a' || p.onclick != null) {
                        return p;
                    }
                    cur = p;
                }
                return el;
            };

            sortByCornerDistance(tier1);
            sortByCornerDistance(tier2);

            // Build the final list and dedupe by clickable wrapper.
            const ranked = [];
            const seen = new Set();
            const tier1Wrapped = tier1.map(toClickable);
            const tier2Wrapped = tier2.map(toClickable);
            for (const el of [...tier1Wrapped, ...tier2Wrapped]) {
                if (seen.has(el)) continue;
                seen.add(el);
                ranked.push(el);
            }

            if (ranked.length === 0) {
                // Diagnostic dump for the top-right area, sorted by distance.
                const dump = all.filter(inTopRight).slice(0, 25).map(el => ({
                    tag: el.tagName,
                    cls: (el.className && el.className.toString)
                            ? el.className.toString().slice(0, 80) : '',
                    aria: el.getAttribute && el.getAttribute('aria-label'),
                    title: el.getAttribute && el.getAttribute('title'),
                    href: el.getAttribute && el.getAttribute('href'),
                    rect: (() => { const r = el.getBoundingClientRect();
                                   return { l: r.left, t: r.top, r: r.right, b: r.bottom }; })(),
                }));
                return { ok: false, reason: 'no header-cart candidate', topRightItems: dump };
            }

            const target = ranked[0];
            try { target.scrollIntoView({block: 'center'}); } catch (e) {}
            const r = target.getBoundingClientRect();
            return {
                ok: true,
                x: r.left + r.width / 2,
                y: r.top + r.height / 2,
                w: r.width, h: r.height,
                tag: target.tagName,
                cls: (target.className && target.className.toString)
                        ? target.className.toString().slice(0, 100) : '',
                href: target.getAttribute && target.getAttribute('href'),
                tier: tier1.length > 0 ? 1 : 2,
                rankedCount: ranked.length,
            };
        }"""
    )
    logger.info(f"  header-cart lookup: {header_cart}")
    if not (header_cart and header_cart.get("ok")):
        logger.error("Could not locate header cart icon.")
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "23_no_header_cart.png"),
                            full_page=True)
        return False

    try:
        jx = float(header_cart["x"]) + random.uniform(-2.0, 2.0)
        jy = float(header_cart["y"]) + random.uniform(-2.0, 2.0)
        page.mouse.move(jx, jy, steps=random.randint(8, 14))
        time.sleep(random.uniform(0.1, 0.25))
        page.mouse.click(jx, jy, delay=random.randint(40, 90))
        logger.info(f"  -> clicked header cart at ({jx:.1f},{jy:.1f})")
    except Exception as e:
        logger.error(f"  failed to click header cart: {e}")
        return False

    time.sleep(random.uniform(1.0, 1.6))
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "24_cart_open.png"),
                        full_page=True)

    # --- 3) Click "Buy" ------------------------------------------------------
    logger.info("Clicking 'Buy' in cart...")
    buy_strategies = [
        lambda: page.get_by_role("button", name="Buy", exact=True).first.click(timeout=5000),
        lambda: page.get_by_role("button", name="BUY", exact=True).first.click(timeout=5000),
        lambda: page.locator("button:has-text('Buy')").first.click(timeout=5000),
        lambda: page.locator("button:has-text('BUY')").first.click(timeout=5000),
        # Force-click as fallback (handles overlay interceptors).
        lambda: page.locator("button:has-text('Buy')").first.click(timeout=5000, force=True),
        lambda: page.get_by_text("Buy", exact=True).first.click(timeout=5000, force=True),
    ]
    buy_clicked = False
    for i, s in enumerate(buy_strategies, 1):
        try:
            s()
            buy_clicked = True
            logger.info(f"  -> clicked Buy (strategy {i})")
            break
        except Exception as e:
            logger.debug(f"  Buy strategy {i} failed: {e}")
    if not buy_clicked:
        logger.error("Could not click 'Buy' button in cart.")
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "25_no_buy_btn.png"),
                            full_page=True)
        return False

    time.sleep(random.uniform(0.8, 1.3))
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "26_after_buy.png"),
                        full_page=True)

    # --- 4) Confirm with "Yes" on the "Are you sure?" modal ------------------
    logger.info("Confirming 'Yes' on 'Are you sure?'...")
    yes_strategies = [
        lambda: page.get_by_role("button", name="Yes", exact=True).first.click(timeout=5000),
        lambda: page.get_by_role("button", name="YES", exact=True).first.click(timeout=5000),
        lambda: page.locator("button:has-text('Yes')").first.click(timeout=5000),
        lambda: page.locator("button:has-text('YES')").first.click(timeout=5000),
        lambda: page.locator("button:has-text('Yes')").first.click(timeout=5000, force=True),
        lambda: page.get_by_text("Yes", exact=True).first.click(timeout=5000, force=True),
    ]
    yes_clicked = False
    for i, s in enumerate(yes_strategies, 1):
        try:
            s()
            yes_clicked = True
            logger.info(f"  -> clicked Yes (strategy {i})")
            break
        except Exception as e:
            logger.debug(f"  Yes strategy {i} failed: {e}")
    if not yes_clicked:
        logger.error("Could not click 'Yes' on confirmation modal.")
        if debug_dir:
            page.screenshot(path=os.path.join(debug_dir, "27_no_yes_btn.png"),
                            full_page=True)
        return False

    time.sleep(random.uniform(1.5, 2.5))
    if debug_dir:
        page.screenshot(path=os.path.join(debug_dir, "28_purchase_submitted.png"),
                        full_page=True)
    logger.success(f"Purchase submitted for '{product_name}'.")
    return True


# ---------- registration flow --------------------------------------------------
def process_registration(page, url, max_captcha_retries=3, debug_dir=None,
                         topup_amount=0, topup_currency="BNB (BEP20)",
                         keep_open=False, keep_session=False,
                         buy_seller_url=None, buy_product_name=None,
                         deposit_timeout=900, web3_send=True,
                         web3_amount_bnb=0.001):
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
                topup_ok = do_topup(page, amount=topup_amount, base_url=base,
                                    currency_label=topup_currency,
                                    debug_dir=debug_dir)
            except Exception as e:
                logger.error(f"Top-up flow failed: {e}")
                topup_ok = False

            # If a buy-flow is requested AND top-up was submitted, wait for
            # the deposit to be confirmed on-chain, then proceed to the
            # seller page and purchase.
            if topup_ok and buy_product_name and buy_seller_url:
                # ----- Auto-send 0.001 BNB on BSC mainnet ---------------------
                # Per user spec: extract the deposit address from the top-up
                # page, send a native BNB transfer from the configured wallet,
                # retry up to 3x, and HALT THE ENTIRE RUN on total failure
                # (sys.exit(2)).
                if web3_send:
                    addr = extract_deposit_address(
                        page, timeout_seconds=30, debug_dir=debug_dir)
                    if not addr:
                        logger.error(
                            "Could not scrape the deposit address from the "
                            "top-up page; cannot auto-send BNB. Halting run "
                            "(per --no-web3-send safety policy).")
                        sys.exit(2)
                    try:
                        tx_hash = send_bnb_deposit(
                            to_address=addr,
                            amount_bnb=web3_amount_bnb,
                            max_retries=3,
                            debug_dir=debug_dir,
                        )
                        logger.success(
                            f"On-chain deposit sent for {username}: "
                            f"{web3_amount_bnb} BNB -> {addr} (tx {tx_hash})")
                    except RuntimeError as e:
                        logger.error(
                            f"BNB auto-send FAILED 3x for {username}: {e}")
                        logger.error(
                            "Halting the entire script per user policy. "
                            "Fix the wallet/RPC, then re-run.")
                        # Persist anything we have so the account is not lost.
                        try:
                            _persist_results([{
                                "url": url, "username": username,
                                "password": password, "secret": secret,
                            }], "accounts.csv")
                        except Exception:
                            pass
                        sys.exit(2)
                else:
                    logger.warning(
                        "--no-web3-send is set; skipping on-chain BNB transfer. "
                        "Send the funds manually if you want the deposit to "
                        "confirm.")

                try:
                    confirmed = wait_for_deposit_confirmed(
                        page,
                        timeout_seconds=deposit_timeout,
                        debug_dir=debug_dir,
                    )
                except Exception as e:
                    logger.error(f"Deposit-wait crashed: {e}")
                    confirmed = False
                if confirmed:
                    try:
                        do_buy_product(page,
                                       seller_url=buy_seller_url,
                                       product_name=buy_product_name,
                                       debug_dir=debug_dir,
                                       login_username=username,
                                       login_password=password)
                    except Exception as e:
                        logger.error(f"Buy-flow crashed: {e}")
                else:
                    logger.warning(
                        "Skipping buy-flow because deposit was not confirmed. "
                        "The browser stays open so you can complete it manually."
                    )

        # Keep the browser session open until the user closes the tab.
        if keep_open:
            logger.info("=" * 64)
            logger.info("Session is now YOURS. Close the browser window when done.")
            logger.info("(The script will exit automatically when you close the tab.)")
            logger.info("=" * 64)
            # Persist the result IMMEDIATELY (before waiting on close), so we
            # never lose creds if the user kills the script or the browser
            # crashes. Thread-safe via _PERSIST_LOCK.
            try:
                _persist_results([result], "accounts.csv")
            except Exception as e:
                logger.debug(f"persist on keep_open failed: {e}")
            try:
                page.wait_for_event("close", timeout=0)
            except Exception as e:
                logger.debug(f"wait_for_event('close') ended: {e}")
            logger.info("Browser tab closed by user - exiting.")
            # Return None so the caller doesn't double-persist this result via
            # the final _persist_results(results, args.output) in main().
            return None

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


def _run_with_patchright(args, urls, debug_dir, results, results_lock, profile_dir):
    """Patchright (Chrome + stealth patches) launcher. One worker = one
    persistent context = one Chrome profile. Each parallel worker MUST pass a
    distinct `profile_dir` so Chromium's profile lock doesn't collide."""
    # CRITICAL: with channel="chrome" we MUST NOT override user_agent / viewport /
    # timezone_id / locale, because real Chrome already has perfectly consistent
    # values for all of those. Overriding even one creates a mismatch between
    # the JS-reported value and the build/OS/binary, which Cloudflare's worker
    # cross-checks. Let Chrome be itself.
    launch_kwargs = dict(
        user_data_dir=profile_dir,
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
                        buy_seller_url=args.buy_seller_url if args.buy_after_topup else None,
                        buy_product_name=args.buy_product if args.buy_after_topup else None,
                        deposit_timeout=args.deposit_timeout,
                        web3_send=not args.no_web3_send,
                        web3_amount_bnb=args.web3_amount,
                    )
                    if result:
                        with results_lock:
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

        try:
            context.close()
        except Exception:
            pass


def _run_with_camoufox(args, urls, debug_dir, results, results_lock, profile_dir):
    """Camoufox (Firefox + C++ fingerprint injection) launcher. Per-worker
    profile dir for parallel-safe operation."""
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
        user_data_dir=profile_dir,
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
                        buy_seller_url=args.buy_seller_url if args.buy_after_topup else None,
                        buy_product_name=args.buy_product if args.buy_after_topup else None,
                        deposit_timeout=args.deposit_timeout,
                        web3_send=not args.no_web3_send,
                        web3_amount_bnb=args.web3_amount,
                    )
                    if result:
                        with results_lock:
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


def _worker_entry(args, urls, debug_dir, results, results_lock, worker_idx, profile_dir):
    """Run one worker (one browser/context) in its own thread. Each worker has
    its own profile dir to avoid Chromium/Firefox profile-lock collisions."""
    try:
        if args.engine == "camoufox":
            _run_with_camoufox(args, urls, debug_dir, results, results_lock, profile_dir)
        else:
            _run_with_patchright(args, urls, debug_dir, results, results_lock, profile_dir)
    except Exception as e:
        logger.error(f"[worker {worker_idx}] crashed: {e}")
    finally:
        logger.info(f"[worker {worker_idx}] done.")


def _run_parallel(args, urls, debug_dir, results, results_lock):
    """Spawn args.threads worker threads, each running an independent browser
    context with its own profile dir.

    Note: each worker still processes urls × args.count internally - so the
    total accounts registered = args.threads * len(urls) * args.count.
    With defaults (--count 1, one URL) this means --threads 3 -> 3 accounts
    registered in parallel across 3 independent Chrome windows.
    """
    n = max(1, int(args.threads))
    if n == 1:
        # Fast path: same behavior as before (no thread pool overhead).
        _worker_entry(args, urls, debug_dir, results, results_lock,
                      worker_idx=0, profile_dir=args.profile)
        return

    logger.info(f"Starting {n} parallel workers (each = 1 browser).")
    if args.use_real_chrome_profile:
        logger.warning(
            "--use-real-chrome-profile is incompatible with --threads > 1 "
            "(profile lock collision). Workers will use isolated profile dirs."
        )

    base_profile = args.profile
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix="styx") as pool:
        futures = []
        for i in range(n):
            # Each worker gets its OWN profile dir so Chromium's
            # SingletonLock doesn't fight between workers.
            wprof = f"{base_profile}.t{i}"
            # Stagger worker starts so the IP-trust pre-flight and browser
            # launches don't all happen at the exact same millisecond.
            if i > 0:
                time.sleep(random.uniform(1.5, 3.5))
            futures.append(pool.submit(
                _worker_entry, args, urls, debug_dir, results,
                results_lock, i, wprof
            ))
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.error(f"worker future error: {e}")


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
    parser.add_argument("--topup",    type=float, default=15,
                        help="After registration: navigate to the wallet "
                             "top-up page, select the crypto from --topup-currency, "
                             "enter this amount and click TOP UP BALANCE. "
                             "Pass 0 to skip the top-up step. Default: 15.")
    parser.add_argument("--topup-currency", default="BNB (BEP20)",
                        help="Crypto tile label to click on the top-up page. "
                             "Examples: 'BNB (BEP20)', 'USDT (TRC20)', 'Bitcoin', "
                             "'Ethereum (ERC20)'.")
    parser.add_argument("--keep-open", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Leave the browser open after the top-up is "
                             "submitted, until you manually close the tab. "
                             "Default: ON. Pass --no-keep-open to auto-close.")
    parser.add_argument("--keep-session", action="store_true",
                        help="Skip the per-run cookie/localStorage reset for the "
                             "target site. By default the script clears Styx "
                             "cookies + storage on every run so each registration "
                             "sees a fresh visitor (the Chrome profile itself is "
                             "still reused to keep Cloudflare trust). Use this "
                             "flag if you instead want to resume the previous "
                             "logged-in session.")
    parser.add_argument("--threads",  type=int, default=1,
                        help="Number of parallel workers (=parallel browser "
                             "windows). Each worker uses its OWN isolated "
                             "profile dir ('<profile>.t<i>') so Chromium's "
                             "profile lock won't collide. Total accounts "
                             "registered = --threads * --count * (URLs). "
                             "Default 1 (sequential).")
    # ----- Post-topup auto-buy flow ------------------------------------------
    # After top-up is submitted, optionally wait for the deposit to be
    # confirmed on-chain (deposit block disappears + green checkmark) and
    # then auto-purchase a product from a seller page.
    parser.add_argument("--buy-after-topup",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="After top-up is submitted, wait for the deposit "
                             "to be confirmed on-chain, then auto-buy the "
                             "product specified by --buy-product on the "
                             "seller page at --buy-seller-url. "
                             "Default: ON. Pass --no-buy-after-topup to skip.")
    parser.add_argument("--buy-seller-url",
                        default="https://styxmarket.si/accounts/profile/"
                                "SCENARIO/?vue=true&user_id=28868",
                        help="Seller-profile URL to navigate to for the "
                             "auto-buy step. Default: SCENARIO's page.")
    parser.add_argument("--buy-product",
                        default="Firstmail.ltd E-Mail Accounts",
                        help="Visible product name to find on the seller "
                             "page. The script clicks the small cart icon "
                             "in this row to add it to the cart, then opens "
                             "the header cart, clicks Buy, and confirms Yes. "
                             "Default: 'Firstmail.ltd E-Mail Accounts'.")
    parser.add_argument("--deposit-timeout", type=int, default=900,
                        help="Max seconds to wait for the deposit to be "
                             "confirmed on-chain before giving up on the "
                             "auto-buy step. Default: 900 (15 minutes).")
    parser.add_argument("--no-web3-send", action="store_true",
                        help="Disable the automated on-chain BNB transfer. "
                             "When unset (default), after TOP UP BALANCE is "
                             "submitted the script scrapes the deposit "
                             "address from the page and sends "
                             "--web3-amount BNB from the wallet whose "
                             "BSC_PRIVATE_KEY lives in /app/.env. Retries up "
                             "to 3x, then HALTS the whole run if all 3 fail.")
    parser.add_argument("--web3-amount", type=float, default=0.001,
                        help="BNB amount (native, not USD) to auto-send per "
                             "deposit. Default: 0.001 (~$0.58). Ignored when "
                             "--no-web3-send is set.")
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
        urls.append("https://www.styxmarket.si/accounts/register/?ref=VM2FCOM6")

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
    results_lock = threading.Lock()
    _run_parallel(args, urls, debug_dir, results, results_lock)

    if results:
        _persist_results(results, args.output)
    else:
        # If nothing reached the final list it usually means keep_open mode
        # persisted in-flow (or all workers failed). Not an error.
        logger.info("No new accounts to save at exit "
                    "(keep-open mode persists in-flow).")


# Module-level lock for thread-safe CSV writes (shared across workers).
_PERSIST_LOCK = threading.Lock()


def _persist_results(results, output_path):
    """Append accounts to the CSV (creates with header if needed).

    Thread-safe via `_PERSIST_LOCK` so parallel workers don't interleave rows
    or race on header creation.
    """
    if not results:
        return
    with _PERSIST_LOCK:
        new_file = not os.path.isfile(output_path)
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["url", "username", "password", "secret"])
            if new_file:
                w.writeheader()
            w.writerows(results)
        logger.info(f"Saved {len(results)} account(s) -> {output_path}")


if __name__ == "__main__":
    main()
