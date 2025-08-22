#!/usr/bin/env python3
"""
linksOnPage2pdf.py
-------------------
Extract all links from a given web page, fetch each linked page, and merge the
content into a single Markdown and PDF file.

This CLI reuses shared logic from ``site2pdf_core`` for fetching and
conversion. Only the initial link discovery on the index page is performed
here.
"""

import argparse
import sys

from site2pdf_core import (
    normalize_url,
    fetch,
    extract_links_from_page,
    crawl_and_output,
)


def main():
    ap = argparse.ArgumentParser(
        description="Extract links from an index page and build Markdown+PDF"
    )
    ap.add_argument(
        "page_url",
        help="The page to scan for links (e.g. https://example.com/path/page)",
    )
    ap.add_argument(
        "--same-origin",
        action="store_true",
        help="Keep only links with the same scheme+host as the page",
    )
    ap.add_argument(
        "--delay", type=float, default=0.35, help="Delay between page fetches (seconds)"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max linked pages to fetch (0 = no limit)",
    )
    args = ap.parse_args()

    page_url = normalize_url(args.page_url)

    print(f"[+] Fetching index page: {page_url}")
    status, content, _ = fetch(page_url)
    if status != 200 or not content:
        print(f"[-] Failed to fetch page ({status}).", file=sys.stderr)
        sys.exit(2)

    links = extract_links_from_page(page_url, content, args.same_origin)
    if args.limit > 0:
        links = links[:args.limit]
    print(f"[+] Found {len(links)} link(s).")

    crawl_and_output(page_url, links, args.delay)


if __name__ == "__main__":
    main()
