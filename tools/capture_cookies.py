#!/usr/bin/env python3
"""One-time utility to capture LinkedIn session cookies for use with --search-url.

Run this script once, log in to LinkedIn in the browser window that opens,
then press Enter. Cookies are saved to config/linkedin_cookies.json.

Usage:
    python tools/capture_cookies.py
"""
import json
import sys
from pathlib import Path


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(
            "playwright is not installed.\n"
            "Run: pip install playwright && playwright install chromium"
        )

    output_file = Path("config/linkedin_cookies.json")

    print("Opening browser for LinkedIn login...")
    print("1. Log in to LinkedIn in the browser window that opens.")
    print("2. Once logged in and on the LinkedIn home page, return here and press Enter.")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()
        page.goto("https://www.linkedin.com/login")

        input("Press Enter after you have logged in to LinkedIn...")

        cookies = context.cookies()
        browser.close()

    output_file.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
    print(f"\nSaved {len(cookies)} cookies to {output_file}")
    print(
        "Set LINKEDIN_COOKIE_FILE in .env if the file lives somewhere other than "
        "config/linkedin_cookies.json."
    )


if __name__ == "__main__":
    main()
