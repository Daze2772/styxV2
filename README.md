# Styx Market Auto-Registration Bot

This is an automated Python script designed to handle the registration flow for Styx Market, bypassing basic bot protections and automatically solving the custom analog clock CAPTCHA using computer vision.

## Features
- **Playwright Automation**: Fast and reliable browser orchestration.
- **Stealth Mode**: Leverages `playwright-stealth` and custom Chromium flags to bypass basic bot detection systems.
- **Computer Vision CAPTCHA Solver**: Uses `OpenCV` to detect the analog clock's hands, calculate their respective angles, and derive the correct `HH:MM` time automatically.
- **Auto-Generation**: Automatically generates strong passwords, randomized usernames, and secret strings.
- **Bulk Processing**: Can read from a text file of multiple referral URLs.
- **CSV Export**: Saves successfully created credentials to a CSV file.

## Setup Instructions

1. Ensure you have Python 3.8+ installed.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the Playwright browser binaries:
   ```bash
   playwright install chromium
   ```

## Usage Guide

You can run the script via the command line with various arguments:

**Run with default test URL (Visible browser mode):**
```bash
python styx_register.py
```

**Run with a specific URL in headless mode:**
```bash
python styx_register.py --url "https://styxmarket.si/accounts/register/?ref=YOUR_REF" --headless
```

**Run bulk registrations from a text file:**
*(Create a file named `links.txt` containing one URL per line)*
```bash
python styx_register.py --file links.txt --output successful_accounts.csv
```

### Output
The script logs its progress to the console using `loguru`. Successful registrations are appended to `accounts.csv` (or the file specified via `--output`), containing the `url`, `username`, `password`, and `secret`.
