import argparse
import random
import string
import csv
import os
import sys
import math
import cv2
import numpy as np
from loguru import logger
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from playwright_stealth import stealth_sync

def generate_random_string(length=10):
    """Generate a random alphanumeric string."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_password(length=14):
    """Generate a strong password meeting standard complexity requirements."""
    upper = random.choice(string.ascii_uppercase)
    lower = random.choice(string.ascii_lowercase)
    digit = random.choice(string.digits)
    special = random.choice("!@#$%^&*")
    rest = ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%^&*", k=length-4))
    pwd = list(upper + lower + digit + special + rest)
    random.shuffle(pwd)
    return ''.join(pwd)

def solve_clock(image_path):
    """
    Solves the analog clock captcha by analyzing the hands using OpenCV.
    Returns the time in HH:MM format.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Could not read clock image at {image_path}")
        return "12:00"
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to extract the dark hands against the white background
    # (Assuming clock face is white and hands are black, THRESH_BINARY_INV makes hands white for detection)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    h, w = thresh.shape
    cx, cy = w // 2, h // 2
    
    # Find lines representing the hands using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
    
    if lines is None:
        logger.warning("No lines detected in the clock image.")
        return "12:00"
        
    hands = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate perpendicular distance from center to the line
        # This ensures the line we detect originates from the center of the clock (the hands)
        dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / (math.hypot(y2 - y1, x2 - x1) + 1e-5)
        
        if dist < min(w, h) * 0.15: # Verify it passes close to the center
            d1 = math.hypot(x1 - cx, y1 - cy)
            d2 = math.hypot(x2 - cx, y2 - cy)
            far_x, far_y = (x1, y1) if d1 > d2 else (x2, y2)
            length = max(d1, d2)
            hands.append({'length': length, 'x': far_x, 'y': far_y})
            
    if not hands:
        return "12:00"
        
    # Sort lines by length descending to distinguish minute (long) and hour (short) hands
    hands.sort(key=lambda item: item['length'], reverse=True)
    
    min_hand = hands[0]
    hour_hand = None
    
    for hand in hands[1:]:
        angle1 = math.atan2(min_hand['y'] - cy, min_hand['x'] - cx)
        angle2 = math.atan2(hand['y'] - cy, hand['x'] - cx)
        # Ensure the hour hand is distinctly different from the minute hand (angle delta)
        if abs(angle1 - angle2) > 0.2:
            hour_hand = hand
            break
            
    if not hour_hand:
        hour_hand = min_hand # Fallback if only one hand is clearly detected
        
    def get_angle(x, y, cx, cy):
        angle = math.degrees(math.atan2(y - cy, x - cx))
        # Shift so 12 o'clock is 0 degrees, and moves clockwise
        angle = (angle + 90) % 360
        return angle
        
    min_angle = get_angle(min_hand['x'], min_hand['y'], cx, cy)
    hour_angle = get_angle(hour_hand['x'], hour_hand['y'], cx, cy)
    
    minute = int(round((min_angle / 360.0) * 60)) % 60
    hour = int(round((hour_angle / 360.0) * 12)) % 12
    if hour == 0:
        hour = 12
        
    return f"{hour:02d}:{minute:02d}"

def process_registration(page, url, max_captcha_retries=3):
    logger.info(f"Navigating to {url}")
    page.goto(url, wait_until="domcontentloaded")
    
    # 1. Quick Verification (Cloudflare/DDoS check)
    try:
        logger.info("Looking for Quick Verification...")
        # Use robust selectors accommodating multiple possible implementations
        continue_btn = page.locator("button:has-text('Continue'), .continue-btn, #continue").first
        if continue_btn.is_visible(timeout=5000):
            continue_btn.click()
            logger.info("Clicked 'Continue' for Quick Verification.")
    except PlaywrightTimeoutError:
        logger.debug("No quick verification step found or timed out, proceeding.")

    # 2. Registration Form
    logger.info("Waiting for Registration form to load...")
    # Wait for the username field to be present
    page.wait_for_selector("input[placeholder*='Username'], input[name='username']", timeout=15000)
    
    username = f"user_{generate_random_string(8)}"
    password = generate_password()
    secret = generate_random_string(12)
    
    logger.info(f"Generated - User: {username}")
    
    page.locator("input[placeholder*='Username'], input[name='username']").fill(username)
    page.locator("input[placeholder*='Password'], input[name='password']").fill(password)
    page.locator("input[placeholder*='Secret'], input[name='secret_code']").fill(secret)
    
    page.locator("button:has-text('Sign up'), input[value='Sign up'], button[type='submit']").click()
    logger.info("Submitted Registration Form.")

    # 3. Time Verification / Clock CAPTCHA
    success = False
    for attempt in range(1, max_captcha_retries + 1):
        try:
            logger.info(f"Waiting for Clock CAPTCHA (Attempt {attempt})...")
            # Wait for the modal instruction text
            page.wait_for_selector("text=To confirm that you are not a robot", timeout=10000)
            
            # Locate the clock image within the modal (it might be a canvas, svg, or img)
            modal = page.locator("div").filter(has_text="To confirm that you are not a robot").last
            clock_el = modal.locator("canvas, img, svg").first
            
            clock_path = f"clock_tmp_{attempt}.png"
            clock_el.screenshot(path=clock_path)
            logger.info(f"Captured clock screenshot to {clock_path}")
            
            time_str = solve_clock(clock_path)
            logger.info(f"Calculated Time via OpenCV: {time_str}")
            
            # Fill the calculated time into the input field
            page.locator("input[placeholder='00:00'], input[name='captcha_time']").fill(time_str)
            page.locator("button:has-text('OK')").click()
            
            # Wait briefly to see if modal disappears or an error appears
            page.wait_for_timeout(2000)
            error_msg = page.locator("text=Incorrect time, text=Error")
            if error_msg.is_visible():
                logger.warning("Incorrect time entered. Retrying if possible...")
                continue
            else:
                logger.info("Captcha accepted!")
                success = True
                break
                
        except PlaywrightTimeoutError:
            # If the modal doesn't appear, the registration might have succeeded without CAPTCHA
            logger.info("No CAPTCHA modal detected or it disappeared quickly.")
            success = True
            break
        except Exception as e:
            logger.error(f"Error solving CAPTCHA: {str(e)}")
            break

    if success:
        logger.success(f"Successfully registered: {username}")
        return {
            "url": url,
            "username": username,
            "password": password,
            "secret": secret
        }
    else:
        logger.error("Failed to pass registration flow.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Automate account registration on Styx Market.")
    parser.add_argument("--url", type=str, help="Single registration URL to process.")
    parser.add_argument("--file", type=str, help="Path to a text file containing URLs (one per line).")
    parser.add_argument("--output", type=str, default="accounts.csv", help="Output CSV file for saved credentials.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    args = parser.parse_args()

    urls = []
    if args.url:
        urls.append(args.url)
    if args.file:
        if os.path.exists(args.file):
            with open(args.file, 'r') as f:
                urls.extend([line.strip() for line in f if line.strip()])
        else:
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
            
    if not urls:
        logger.info("No URL provided. Using default test URL.")
        urls.append("https://styxmarket.si/accounts/register/?ref=7QXIWQR1")

    results = []
    with sync_playwright() as p:
        # Launch browser with evasion techniques
        browser = p.chromium.launch(
            headless=args.headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        )
        
        for url in urls:
            page = context.new_page()
            # Apply playwright-stealth to avoid bot detection
            stealth_sync(page)
            try:
                result = process_registration(page, url)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Critical error on {url}: {e}")
            finally:
                page.close()
                
        browser.close()

    if results:
        # Save successfully generated credentials to CSV
        file_exists = os.path.isfile(args.output)
        with open(args.output, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["url", "username", "password", "secret"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)
        logger.info(f"Saved {len(results)} accounts to {args.output}")

if __name__ == "__main__":
    main()
