# Styx Market — Automated Registration

End-to-end script that:
1. Bypasses Cloudflare's bot challenge using a stealth-patched Chrome (or, optionally, Camoufox).
2. Fills the registration form with random credentials.
3. Solves the analog-clock CAPTCHA with OpenCV.
4. Saves the credentials to a CSV.

---

## 1. Install (one time)

```bash
pip install -r requirements.txt
patchright install chrome
```

> **Important:** the script will *refuse to start* if `patchright` is missing.
> Vanilla `playwright` is instantly detected by Cloudflare, so the fallback that
> previously masked this problem has been removed.

### Optional — install Camoufox (stronger Cloudflare bypass)

If `patchright` still gets blocked on your environment, use the Firefox-based
Camoufox engine:

```bash
pip install "camoufox[geoip]"
camoufox fetch
```

Then run with `--engine camoufox`.

---

## 2. Run

```bash
# Default: single registration, headed Chrome, persistent profile in temp dir.
python styx_register.py

# Multiple accounts:
python styx_register.py --count 5

# Custom URL / batch file / CSV output:
python styx_register.py --url "https://styxmarket.si/accounts/register/?ref=YOURREF"
python styx_register.py --file urls.txt --output my_accounts.csv

# Debug artifacts (screenshots + HTML dumps) to ./debug_out/
python styx_register.py --debug

# Use a residential proxy (HIGHLY recommended if you're on a cloud VM):
python styx_register.py --proxy "http://user:pass@proxy.example.com:8080"
# ...or set it once via env var:
export STYX_PROXY="http://user:pass@proxy.example.com:8080"
python styx_register.py

# Use the strongest engine (Camoufox) if Chrome still gets blocked:
python styx_register.py --engine camoufox

# Wipe the persistent profile if a previous run got CF-flagged:
python styx_register.py --fresh-profile

# Use your real installed Chrome profile (cookies, history → looks human):
# WARNING: close all your Chrome windows first.
python styx_register.py --use-real-chrome-profile

# Register, then auto-load /wallet/top-up/, select BNB (BEP20), enter 15,
# click TOP UP BALANCE, and leave the browser open until you close the tab:
python styx_register.py --topup 15 --keep-open

# Same flow with a different crypto:
python styx_register.py --topup 25 --topup-currency "USDT (TRC20)" --keep-open
```

---

## 3. Why this works (and what to do if it doesn't)

Cloudflare's Bot Management runs **three** layers of detection:

| Layer | What it checks | What we do |
|---|---|---|
| TLS | JA3/JA4 handshake fingerprint | Use `patchright` (Chrome) or `camoufox` (Firefox) — both match real browser handshakes |
| Browser | `navigator.webdriver`, plugins, WebGL vendor, CDP traces | Stealth patches strip these leaks |
| Network | IP reputation, ASN, geolocation | **You** must supply a clean IP (residential, mobile, or a paid residential proxy) |

### Datacenter IP warning

If you run this on an **Azure / AWS / GCP / DigitalOcean / OVH / Hetzner / Linode / Vultr** VM, the script auto-detects this and prints a loud warning at startup. Cloudflare assigns near-zero trust to those IP ranges — even a perfect fingerprint will be blocked.

**Mitigations**, in order of effectiveness:
1. Run from a residential machine (your home, a friend's, a coffee shop).
2. Use a paid **residential** proxy (Bright Data, IPRoyal, Smartproxy, Oxylabs).
3. Use a paid **mobile** proxy (most trusted, most expensive).
4. Datacenter proxies *do not work* against Cloudflare. Don't bother.

### When patchright still gets blocked

1. Pass `--fresh-profile` (a previous flagged session might be poisoning cookies).
2. Try `--engine camoufox`.
3. Try `--use-real-chrome-profile` so Cloudflare sees your real browsing history.
4. Combine with a residential proxy.

---

## 4. Output

Successful registrations are appended to `accounts.csv` (default) with columns:

```
url,username,password,secret
```
