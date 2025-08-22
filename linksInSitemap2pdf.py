#!/usr/bin/env python3
"""
linksInSitemap2pdf.py
---------------------
Fetch a sitemap.xml file, crawl each URL listed, and compile the content into
combined Markdown and PDF outputs.

This CLI shares crawling and conversion logic with ``site2pdf_core``.
"""

import argparse
import sys
import urllib.parse

from site2pdf_core import (
    normalize_url,
    fetch,
    extract_links_from_sitemap,
    crawl_and_output,
    discover_sitemap_url,
)


def main():
    ap = argparse.ArgumentParser(
        description="Crawl all links in a sitemap.xml and build Markdown+PDF"
    )
    ap.add_argument(
        "url",
        help="Website root or sitemap.xml URL (e.g. https://example.com)",
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

    raw_url = normalize_url(args.url)
    if raw_url.lower().endswith(".xml"):
        sitemap_url = raw_url
        p = urllib.parse.urlparse(sitemap_url)
        base_url = f"{p.scheme}://{p.netloc}"
    else:
        base_url = raw_url.rstrip("/")
        try:
            sitemap_url = discover_sitemap_url(base_url)
        except RuntimeError as e:
            print(f"[-] {e}", file=sys.stderr)
            sys.exit(2)

    print(f"[+] Fetching sitemap: {sitemap_url}")
    status, content, _ = fetch(sitemap_url)
    if status != 200 or not content:
        print(f"[-] Failed to fetch sitemap ({status}).", file=sys.stderr)
        sys.exit(2)

    links = extract_links_from_sitemap(content)
    if args.limit > 0:
        links = links[:args.limit]
    print(f"[+] Found {len(links)} link(s) in sitemap.")

    crawl_and_output(base_url, links, args.delay)


if __name__ == "__main__":
    main()
