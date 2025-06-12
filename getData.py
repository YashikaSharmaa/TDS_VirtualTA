from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
from datetime import datetime
import time
import re

# Define cutoff date
CUTOFF_DATE = datetime(2025, 4, 15)

# List of URLs to scrape
start_urls = [
    "https://tds.s-anand.net/#/2025-01/",
    "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"
]

USERNAME = "24f2001055@ds.study.iitm.ac.in"
PASSWORD = "--"

def extract_date_from_text_or_url(text, url):
    """
    Extracts a date from either the page content or URL.
    """
    date_patterns = [
        r'(\d{1,2}\s+\w+\s+\d{4})',  # 14 April 2025
        r'(\w+\s+\d{1,2},\s+\d{4})',  # April 14, 2025
        r'(\d{4}-\d{2}-\d{2})'        # 2025-04-14
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            for fmt in ['%d %B %Y', '%B %d, %Y', '%Y-%m-%d']:
                try:
                    return datetime.strptime(match.group(1), fmt)
                except:
                    continue

    # Try extracting from URL if not found in text
    match = re.search(r'(\d{4}-\d{2}-\d{2})', url)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d")
        except:
            pass

    return None

with sync_playwright() as p:
    DEBUG_MODE = False
    browser = p.chromium.launch(headless=not DEBUG_MODE)
    context = browser.new_context()
    page = context.new_page()

    for start_url in start_urls:
        page.goto(start_url)
        page.wait_for_load_state("networkidle")

        # Check for login
        login_required = False
        login_button_keywords = ["login", "sign in", "log in", "authenticate", "access account"]
        login_input_fields = ["login-account-name", "login-account-password", "username", "password", "email"]

        buttons = page.query_selector_all("button")
        for button in buttons:
            label = (button.inner_text() or button.get_attribute("aria-label") or "").strip().lower()
            if any(keyword in label for keyword in login_button_keywords):
                login_required = True
                break

        inputs = page.query_selector_all("input")
        for input_tag in inputs:
            name = input_tag.get_attribute("name")
            if name and name.lower() in login_input_fields:
                login_required = True
                break

        if login_required:
            print("🔐 Login required. Attempting login...")
            login_links = page.query_selector_all("a")
            for link in login_links:
                text = (link.inner_text() or "").strip().lower()
                if any(keyword in text for keyword in login_button_keywords):
                    print(f"🔘 Clicking login link: {text}")
                    link.click()
                    page.wait_for_load_state("networkidle")
                    time.sleep(1)
                    break

            try:
                page.wait_for_selector('#login-account-name', timeout=10000)
                page.fill('#login-account-name', USERNAME)
                page.fill('#login-account-password', PASSWORD)
                page.click("#login-button")
                page.wait_for_load_state("networkidle")
                time.sleep(2)
                print("✅ Login successful.")
            except:
                print("❌ Login failed. Skipping this URL.")
                continue

        # Extract links
        print("🔗 Collecting page links...")
        links = page.query_selector_all("a")
        hrefs = []
        for link in links:
            href = link.get_attribute("href")
            if href and not (
                href.startswith("mailto:") or 
                "linkedin.com" in href.lower() or
                "@" in href
            ):
                full_url = urljoin(start_url, href)
                hrefs.append(full_url)

        print(f"🔍 Found {len(hrefs)} valid links.")

        contents = []
        trusted_domains = ["tds.s-anand.net"]

        for href in hrefs:
            try:
                print(f"🌐 Visiting: {href}")
                page.goto(href)
                page.wait_for_load_state("networkidle")
                body = page.query_selector("body")
                content = body.inner_text() if body else ""
                post_date = extract_date_from_text_or_url(content, href)
                domain = href.split("/")[2]
                
                if post_date and post_date >= CUTOFF_DATE and domain not in trusted_domains:
                    print(f"⏭️ Skipped (post date {post_date.date()} after cutoff): {href}")
                    continue

               # Build content with source link
                content_with_link = f"{content}\n\n[Source]({href})"

                contents.append(content_with_link)
                print(f"✅ Extracted from: {href} (date: {post_date.date() if post_date else 'N/A'})")

                """
                if post_date:
                    if post_date < CUTOFF_DATE:
                        contents.append(content)
                        print(f"✅ Extracted from: {href} (date: {post_date.date()})")
                    else:
                        print(f"⏭️ Skipped (post date {post_date.date()} after cutoff): {href}")
                elif domain in trusted_domains:
                    contents.append(content)
                    print(f"✅ Extracted from: {href} (no date, trusted domain)")
                else:
                    print(f"❓ Skipped (no date found): {href}")
                """
            except Exception as e:
                print(f"⚠️ Failed to extract from {href}: {e}")
                continue
        
        # Instead of storing content alone, store content + source link
        content_with_link = f"{content.strip()}\n\n[Source]({href})"
        contents.append(content_with_link)

        # ...later, when writing to the file:
        with open("extracted_contents_filtered_v1.doc", "a", encoding="utf-8") as f:
            for content in contents:
                f.write(content.strip() + "\n\n---\n\n")

        print(f"📁 Contents saved for URL: {start_url}")

    context.close()
    browser.close()
    print("✅ All done.")
