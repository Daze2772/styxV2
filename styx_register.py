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
def solve_clock(image_path):
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Could not read clock image at {image_path}")
        return "12:00"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    h, w = thresh.shape
    cx, cy = w // 2, h // 2

    lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi / 180,
                            threshold=30, minLineLength=20, maxLineGap=5)
    if lines is None:
        return "12:00"

    hands = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) \
               / (math.hypot(y2 - y1, x2 - x1) + 1e-5)
        if dist < min(w, h) * 0.15:
            d1 = math.hypot(x1 - cx, y1 - cy)
            d2 = math.hypot(x2 - cx, y2 - cy)
            far_x, far_y = (x1, y1) if d1 > d2 else (x2, y2)
            hands.append({'length': max(d1, d2), 'x': far_x, 'y': far_y})

    if not hands:
        return "12:00"

    hands.sort(key=lambda h: h['length'], reverse=True)
    min_hand = hands[0]
    hour_hand = None
    for hand in hands[1:]:
        a1 = math.atan2(min_hand['y'] - cy, min_hand['x'] - cx)
        a2 = math.atan2(hand['y'] - cy, hand['x'] - cx)
        if abs(a1 - a2) > 0.2:
            hour_hand = hand
            break
    if not hour_hand:
        hour_hand = min_hand

    def get_angle(x, y):
        return (math.degrees(math.atan2(y - cy, x - cx)) + 90) % 360

    minute = int(round((get_angle(min_hand['x'], min_hand['y']) / 360.0) * 60)) % 60
    hour   = int(round((get_angle(hour_hand['x'], hour_hand['y']) / 360.0) * 12)) % 12
    if hour == 0:
        hour = 12
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


# ---------- registration flow --------------------------------------------------
def process_registration(page, url, max_captcha_retries=3, debug_dir=None):
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

    username = f"user_{generate_random_string(8)}"
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
    success = False
    for attempt in range(1, max_captcha_retries + 1):
        try:
            logger.info(f"Waiting for clock CAPTCHA (attempt {attempt})...")
            page.wait_for_selector("text=To confirm that you are not a robot",
                                   timeout=10000)

            modal = page.locator("div").filter(
                has_text="To confirm that you are not a robot").last
            clock_el = modal.locator("canvas, img, svg").first

            clock_path = os.path.join(debug_dir or ".", f"clock_{attempt}.png")
            clock_el.screenshot(path=clock_path)
            logger.info(f"Clock screenshot -> {clock_path}")

            time_str = solve_clock(clock_path)
            logger.info(f"Solved time: {time_str}")

            smart_fill(page,
                ["input[placeholder='00:00']",
                 "input[name='captcha_time']",
                 "input.input__input"],
                time_str, label="captcha_time")

            page.locator("button:has-text('OK')").click(timeout=5000)
            page.wait_for_timeout(2500)

            err = page.locator("text=Incorrect time, text=Error")
            if err.count() > 0 and err.first.is_visible():
                logger.warning("Wrong time, retrying...")
                continue

            logger.info("CAPTCHA accepted.")
            success = True
            break

        except PlaywrightTimeoutError:
            # Look for validation errors that may have prevented CAPTCHA
            err = page.locator(".error-message, .invalid-feedback, "
                               "text=This field is required").first
            if err.count() and err.is_visible():
                logger.error(f"Form validation error: {err.inner_text()}")
                break
            logger.info("No CAPTCHA appeared — likely already registered.")
            success = True
            break
        except Exception as e:
            logger.error(f"CAPTCHA error: {e}")
            break

    if success:
        logger.success(f"Registered: {username}")
        return {"url": url, "username": username, "password": password, "secret": secret}
    logger.error("Registration flow failed.")
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
                    result = process_registration(page, url, debug_dir=debug_dir)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Critical error on {url}: {e}")
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass
                # Human-like pause between registrations
                if args.count > 1 and i < args.count - 1:
                    time.sleep(random.uniform(4.0, 8.0))

        context.close()


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
                    result = process_registration(page, url, debug_dir=debug_dir)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Critical error on {url}: {e}")
                finally:
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
        new_file = not os.path.isfile(args.output)
        with open(args.output, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["url", "username", "password", "secret"])
            if new_file:
                w.writeheader()
            w.writerows(results)
        logger.info(f"Saved {len(results)} account(s) -> {args.output}")
    else:
        logger.warning("No accounts saved.")


if __name__ == "__main__":
    main()
