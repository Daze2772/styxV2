"""
Debug script to investigate why Playwright can't find the password input.
Tests multiple hypotheses:
  H1: Quick Verification opens form in NEW TAB/POPUP
  H2: Verification redirects but loading bar must finish first
  H3: Form is inside iframe
  H4: Shadow DOM
  H5: HTML snapshot evaluation
"""
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
import time, os

OUT = "/app/debug_out"
os.makedirs(OUT, exist_ok=True)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        )

        # Track all new pages (popups)
        opened_pages = []
        def on_page(pg):
            print(f"[EVENT] New page opened: {pg.url}")
            opened_pages.append(pg)
        context.on("page", on_page)

        page = context.new_page()
        Stealth().apply_stealth_sync(page)

        url = "https://styxmarket.si/accounts/register/?ref=7QXIWQR1"
        print(f"[STEP] Navigating to {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)
        page.screenshot(path=f"{OUT}/01_initial.png")
        print(f"[STATE] URL={page.url}, Title={page.title()}")

        # Check for iframes
        print(f"[INFO] iframes count: {len(page.frames)}")
        for f in page.frames:
            print(f"  - frame name='{f.name}' url='{f.url}'")

        # Try clicking Continue
        try:
            print("[STEP] Clicking Continue button...")
            with context.expect_page(timeout=10000) as new_page_info:
                page.locator("button:has-text('Continue'), .continue-btn, #continue").first.click(timeout=10000)
            new_page = new_page_info.value
            print(f"[!] Continue opened NEW PAGE: {new_page.url}")
            new_page.wait_for_load_state("domcontentloaded")
            time.sleep(3)
            new_page.screenshot(path=f"{OUT}/02_newtab.png")
            print(f"[STATE] new_page URL={new_page.url}")
            print(f"[INFO] new_page iframes count: {len(new_page.frames)}")
            page = new_page
        except Exception as e:
            print(f"[INFO] No new page opened (might be same page): {e}")
            # Wait for loading animation - the loading bar in screenshot suggests it animates for a few seconds
            time.sleep(8)
            page.screenshot(path=f"{OUT}/02_after_continue.png")
            print(f"[STATE] URL after continue={page.url}")

        # Wait extra for any animations to settle
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        time.sleep(2)
        page.screenshot(path=f"{OUT}/03_final.png", full_page=True)
        print(f"[STATE] FINAL URL={page.url}, Title={page.title()}")

        # Dump full HTML
        html = page.content()
        with open(f"{OUT}/page.html", "w") as f:
            f.write(html)
        print(f"[INFO] HTML length: {len(html)}")
        print(f"[INFO] Contains 'password': {'password' in html.lower()}")
        print(f"[INFO] Contains 'sign-up__form': {'sign-up__form' in html}")
        print(f"[INFO] Contains 'input__input': {'input__input' in html}")

        # Check ALL pages in context
        print(f"\n[INFO] All pages in context: {len(context.pages)}")
        for i, pg in enumerate(context.pages):
            print(f"  Page {i}: url={pg.url}")
            # Try JS evaluation on each page
            try:
                count = pg.evaluate("document.querySelectorAll('input').length")
                pwd_count = pg.evaluate("document.querySelectorAll('input[name=\"password\"]').length")
                print(f"    -> input total: {count}, password inputs: {pwd_count}")
            except Exception as e:
                print(f"    -> evaluate error: {e}")

        # Check frames on the final page
        print(f"\n[INFO] Frames on final page: {len(page.frames)}")
        for f in page.frames:
            print(f"  - frame name='{f.name}' url='{f.url}'")
            try:
                pwd = f.evaluate("document.querySelectorAll('input[name=\"password\"]').length")
                print(f"    -> password inputs in this frame: {pwd}")
            except Exception as e:
                print(f"    -> err: {e}")

        # Try a direct JS query
        try:
            inputs_info = page.evaluate("""
                () => {
                    const arr = [];
                    document.querySelectorAll('input').forEach(el => {
                        arr.push({
                            name: el.name,
                            type: el.type,
                            cls: el.className,
                            id: el.id,
                            visible: el.offsetParent !== null,
                            rect: el.getBoundingClientRect().toJSON()
                        });
                    });
                    return arr;
                }
            """)
            print(f"\n[INFO] All inputs on page (JS eval): {len(inputs_info)}")
            for inp in inputs_info:
                print(f"  {inp}")
        except Exception as e:
            print(f"[ERR] JS eval failed: {e}")

        browser.close()

if __name__ == "__main__":
    main()
