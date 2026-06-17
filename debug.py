from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()
        Stealth().apply_stealth_sync(page)
        
        url = "https://styxmarket.si/accounts/register/?ref=7QXIWQR1"
        page.goto(url, wait_until="domcontentloaded")
        
        page.screenshot(path="/app/step1.png")
        try:
            page.locator("button:has-text('Continue'), .continue-btn, #continue").first.click(timeout=5000)
            page.wait_for_timeout(5000)
            page.screenshot(path="/app/step2.png")
        except Exception as e:
            print("No continue button or error:", e)
        
        browser.close()

if __name__ == "__main__":
    main()
