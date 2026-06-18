"""
Test if patchright (a stealth-patched Playwright) can pass the Quick Verification 
without being flagged by Cloudflare.
"""
from patchright.sync_api import sync_playwright
import time, os

OUT = "/app/debug_out"
os.makedirs(OUT, exist_ok=True)

def main():
    with sync_playwright() as p:
        # Use a persistent context with patchright - this is the recommended way
        # for maximum stealth (matches a real user's profile)
        ctx = p.chromium.launch_persistent_context(
            user_data_dir="/tmp/styx_profile",
            channel="chromium",      # patchright's patched chromium
            headless=True,
            no_viewport=True,
            args=[]                  # patchright's defaults already include best stealth args
        )

        page = ctx.new_page()
        url = "https://styxmarket.si/accounts/register/?ref=7QXIWQR1"
        print(f"[STEP] Navigating to {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)
        page.screenshot(path=f"{OUT}/pr_01_initial.png")
        print(f"[STATE] URL={page.url}, Title={page.title()}")

        # Click Continue
        try:
            print("[STEP] Clicking Continue...")
            page.locator("button:has-text('Continue'), .continue-btn, #continue").first.click(timeout=10000)
        except Exception as e:
            print(f"[INFO] No continue button: {e}")

        # Wait long enough for the loading-bar animation (~5s) plus navigation
        time.sleep(10)
        page.screenshot(path=f"{OUT}/pr_02_after_continue.png")
        print(f"[STATE] URL after continue={page.url}, Title={page.title()}")

        html = page.content()
        print(f"[INFO] HTML length: {len(html)}")
        print(f"[INFO] Contains 'password': {'password' in html.lower()}")
        print(f"[INFO] Contains 'sign-up__form': {'sign-up__form' in html}")
        print(f"[INFO] Contains 'input__input': {'input__input' in html}")
        print(f"[INFO] Contains 'blocked': {'blocked' in html.lower()}")
        print(f"[INFO] Contains 'Cloudflare': {'Cloudflare' in html}")

        try:
            count = page.evaluate("document.querySelectorAll('input').length")
            pwd = page.evaluate("document.querySelectorAll('input[name=\"password\"]').length")
            print(f"[INFO] Inputs total={count} password={pwd}")
        except Exception as e:
            print(f"[ERR] eval: {e}")

        ctx.close()

if __name__ == "__main__":
    main()
