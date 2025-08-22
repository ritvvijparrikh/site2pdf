#!/usr/bin/env python3
"""
onepage_links_to_pdf.py (robust extraction + inline formatting + PDF summary)

- Extract all <a> links from an index page (optional same-origin).
- Fetch each link politely (Session + retries/backoff).
- Convert HTML -> "md+" (markdown structure + inline **bold**, *italics*, <u>underline</u>, line breaks).
  * Handles legacy <table>/<td>/<font> and <br><br> paragraphs.
  * Fallback to plaintext paragraphs if structured pass is too thin.
- Build Markdown + PDF.
- PDF first page summary includes:
  - How many urls found
  - How many urls scraped
  - How many urls has less than 10 characters
  - How many urls has less than 10 words
  - List of scraped URLs + words included from each page

Output:
  - Files are saved under ./output/<netloc_path>.md and ./output/<netloc_path>.pdf

Usage:
  python onepage_links_to_pdf.py https://example.com/page
  python onepage_links_to_pdf.py https://example.com/page --same-origin --limit 100
"""

import argparse
import io
import os
import re
import sys
import time
import urllib.parse
from typing import List, Set, Dict, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup, Tag, NavigableString

# PDF generation (pure Python)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_LEFT

DEFAULT_HEADERS = {
    "User-Agent": "OnePageLinksToPDF/1.3 (+educational; respectful-bot)",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "close",
}

OUTPUT_DIR = "output"

# ----------------------------
# HTTP session with retries
# ----------------------------

def make_session():
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = make_session()

def fetch(url: str, timeout: float = 20.0):
    try:
        resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
        ctype = resp.headers.get("Content-Type", "") or ""
        return resp.status_code, resp.content, ctype
    except Exception:
        return -1, b"", ""

# ----------------------------
# URL helpers
# ----------------------------

def normalize_url(base_url: str) -> str:
    p = urllib.parse.urlparse(base_url)
    if not p.scheme:
        return "https://" + base_url.lstrip("/")
    return base_url

def strip_fragment(u: str) -> str:
    p = urllib.parse.urlparse(u)
    p = p._replace(fragment="")
    return urllib.parse.urlunparse(p)

def same_origin(u1: str, u2: str) -> bool:
    p1, p2 = urllib.parse.urlparse(u1), urllib.parse.urlparse(u2)
    return (p1.scheme, p1.netloc) == (p2.scheme, p2.netloc)

def make_output_stem(page_url: str) -> str:
    p = urllib.parse.urlparse(page_url)
    netloc = p.netloc or "output"
    path = p.path.strip("/").replace("/", "_") or "index"
    return f"{netloc}_{path}"

# ----------------------------
# Link extraction
# ----------------------------

def extract_links_from_page(page_url: str, html_bytes: bytes, only_same_origin: bool) -> List[str]:
    soup = BeautifulSoup(html_bytes, "lxml")
    links: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        abs_url = urllib.parse.urljoin(page_url, href)
        abs_url = strip_fragment(abs_url)
        if not abs_url.lower().startswith(("http://", "https://")):
            continue
        links.add(abs_url)

    if only_same_origin:
        links = {u for u in links if same_origin(u, page_url)}

    return sorted(links)

# ----------------------------
# HTML -> Markdown (with inline formatting)
# ----------------------------

def clean_html(html: bytes) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for bad in soup(["script", "style", "noscript", "template", "svg", "canvas", "iframe"]):
        bad.decompose()
    for bad in soup.find_all(True, {"aria-hidden": "true"}):
        bad.decompose()
    return soup

def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# Convert a node to inline "md+" string preserving **, *, <u>, and explicit <br/>
def rich_text_of(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    name = (node.name or "").lower()

    # line breaks become explicit <br/> markers to preserve "New Line" in MD and PDF
    if name == "br":
        return "<br/>"

    # Inline formatting
    if name in {"b", "strong"}:
        inner = "".join(rich_text_of(c) for c in node.children)
        return f"**{inner}**"
    if name in {"i", "em"}:
        inner = "".join(rich_text_of(c) for c in node.children)
        return f"*{inner}*"
    if name == "u":
        inner = "".join(rich_text_of(c) for c in node.children)
        return f"<u>{inner}</u>"

    # Links: keep text; (optionally append URL) -> keeping just text for clean copy
    if name == "a":
        inner = "".join(rich_text_of(c) for c in node.children)
        return inner

    # Other inline containers (span, font, etc.)
    if name in {"span", "font", "small", "big", "time", "mark", "abbr", "code"}:
        return "".join(rich_text_of(c) for c in node.children)

    # Block-ish tags handled elsewhere; here, return concatenated text
    return "".join(rich_text_of(c) for c in node.children)

# Legacy blocklike containers common on older sites
BLOCKLIKE = {"div","section","article","td","blockquote","center","pre","font"}

def soup_to_plain_paragraphs(soup: BeautifulSoup) -> List[str]:
    """Plain-text fallback: preserve <br> as <br/>; split on blank lines."""
    for br in soup.find_all("br"):
        br.replace_with("<br/>")
    raw = soup.get_text(separator="\n", strip=True)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    # Convert remaining single newlines within paras to <br/>
    paras = [p.replace("\n", "<br/>") for p in paras]
    return paras

def iter_blocks_rich(soup: BeautifulSoup):
    """
    Yield (type, text, level) for headings, paragraphs, lists, with inline **/*/<u> and <br/>.
    Also handles legacy layouts (<td>/<font>) and <br><br> paragraph breaks.
    """
    body = soup.body or soup

    # Headings
    for h in body.find_all(re.compile(r"^h[1-6]$", re.I)):
        txt = collapse_ws(rich_text_of(h))
        if txt:
            yield ("heading", txt, int(h.name[1]))
        h.decompose()  # avoid double counting

    # Lists
    for ul in body.find_all("ul"):
        for li in ul.find_all("li", recursive=False):
            txt = collapse_ws(rich_text_of(li))
            if txt:
                yield ("li-ul", txt, None)
        ul.decompose()

    for ol in body.find_all("ol"):
        for li in ol.find_all("li", recursive=False):
            txt = collapse_ws(rich_text_of(li))
            if txt:
                yield ("li-ol", txt, None)
        ol.decompose()

    # Standard paragraphs
    for p in body.find_all("p"):
        # Preserve <br> breaks within a paragraph as <br/>
        for br in p.find_all("br"):
            br.replace_with("<br/>")
        # keep inline formatting by re-walking with rich_text_of
        txt = "".join(rich_text_of(c) for c in p.children)
        # split on blank lines/newline groups into paras
        for para in re.split(r"(?:<br/>\s*){2,}", txt):
            # keep single <br/> inside a paragraph
            para = collapse_ws(para).replace(" <br/> ", "<br/>")
            if para:
                yield ("para", para, None)
        p.decompose()

    # Legacy blocklike containers: treat double <br/> as paragraph breaks
    for blk in body.find_all(BLOCKLIKE):
        for br in blk.find_all("br"):
            br.replace_with("<br/>")
        # rebuild with inline formatting
        txt = "".join(rich_text_of(c) for c in blk.children)
        for para in re.split(r"(?:<br/>\s*){2,}", txt):
            para = collapse_ws(para).replace(" <br/> ", "<br/>")
            if para:
                yield ("para", para, None)
        blk.decompose()

def html_to_markdown(html: bytes, page_url: str) -> str:
    soup = clean_html(html)
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    md_lines = [f"# {title or page_url}", f"_Source: {page_url}_", ""]
    current_list = None
    word_counter_text: List[str] = []

    for kind, txt, lvl in iter_blocks_rich(soup):
        # For counting, strip our inline tags; keep words only
        word_counter_text.append(re.sub(r"<[^>]+>", "", txt))

        if kind == "heading":
            if current_list:
                current_list = None; md_lines.append("")
            md_lines.append(f"{'#'*max(1,min(6,lvl))} {txt}")
            md_lines.append("")
        elif kind == "para":
            if current_list:
                current_list = None; md_lines.append("")
            md_lines.append(txt); md_lines.append("")
        elif kind == "li-ul":
            if current_list != "ul":
                if current_list: md_lines.append("")
                current_list = "ul"
            md_lines.append(f"- {txt}")
        elif kind == "li-ol":
            if current_list != "ol":
                if current_list: md_lines.append("")
                current_list = "ol"
            md_lines.append(f"1. {txt}")

    if current_list:
        md_lines.append("")

    out = "\n".join(md_lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"

    # Fallback if too thin (typical of legacy layouts that defeated structure)
    wc = len(re.findall(r"\b\w+\b", " ".join(word_counter_text)))
    if wc < 80:
        paras = soup_to_plain_paragraphs(soup)
        rebuilt = [f"# {title or page_url}", f"_Source: {page_url}_", ""]
        for p in paras:
            rebuilt.append(p); rebuilt.append("")
        out = re.sub(r"\n{3,}", "\n\n", "\n".join(rebuilt)).strip() + "\n"

    return out

# ----------------------------
# Markdown -> PDF (preserve **, *, <u>, <br/>) with tag-nesting repair
# ----------------------------

_TAG_RE = re.compile(r"</?(b|i|u|br)\s*/?>", re.I)

def _fix_tag_nesting(s: str) -> str:
    """
    Repair mis-nested <b>/<i>/<u> produced by overlapping **/* markers.
    Ensures tags are properly nested and closed in LIFO order.
    """
    out: List[str] = []
    stack: List[str] = []
    i = 0
    for m in _TAG_RE.finditer(s):
        start, end = m.span()
        tag = m.group(1).lower()
        is_close = s[start+1] == '/'
        # emit text before tag
        if start > i:
            out.append(s[i:start])
        i = end

        if tag == "br" and not is_close:
            out.append("<br/>")
            continue

        if not is_close:
            out.append(f"<{tag}>")
            stack.append(tag)
        else:
            # Close tags until we match 'tag'
            if tag in stack:
                while stack and stack[-1] != tag:
                    out.append(f"</{stack.pop()}>")
                out.append(f"</{tag}>")
                stack.pop()
            else:
                # unmatched close: drop it
                pass
    # trailing text
    out.append(s[i:])
    # close any remaining tags
    while stack:
        out.append(f"</{stack.pop()}>")
    return "".join(out)

def md_inline_to_reportlab(s: str) -> str:
    """
    Convert inline md markers to ReportLab paragraph XML:
      **bold** -> <b>bold</b>
      *italics* or _italics_ -> <i>italics</i>
      <u>underline</u> kept
      <br/> line breaks kept
    We escape HTML, then unescape allowed tags; finally, repair nesting.
    """
    # Handle bold+italic combined first to reduce overlap (***text***)
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", s)

    # Bold: handle **...** non-greedy, multiple occurrences
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

    # Italics from *...* not part of **...**
    s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", s)

    # Italics from _..._ (avoid matching _Source: ...)
    s = re.sub(r"_(?!Source:)(.+?)_", r"<i>\1</i>", s)

    # Escape everything
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Unescape only the allowed tags (including <br/>)
    for tag in ("b", "i", "u", "br"):
        s = s.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
        s = s.replace(f"&lt;{tag}/&gt;", f"<{tag}/>")
    # Repair any mis-nesting
    s = _fix_tag_nesting(s)
    return s

def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    H = {
        1: ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, leading=24, spaceAfter=8, spaceBefore=12),
        2: ParagraphStyle("H2", parent=styles["Heading2"], fontSize=18, leading=22, spaceAfter=6, spaceBefore=10),
        3: ParagraphStyle("H3", parent=styles["Heading3"], fontSize=16, leading=20, spaceAfter=6, spaceBefore=8),
        4: ParagraphStyle("H4", parent=styles["Heading4"], fontSize=14, leading=18, spaceAfter=6, spaceBefore=8),
        5: ParagraphStyle("H5", parent=styles["Heading5"], fontSize=12, leading=16, spaceAfter=4, spaceBefore=6),
        6: ParagraphStyle("H6", parent=styles["Heading6"], fontSize=11, leading=14, spaceAfter=4, spaceBefore=6),
    }
    BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=11, leading=15, alignment=TA_LEFT, spaceAfter=6)
    ITAL = ParagraphStyle("Ital", parent=BODY, fontName="Times-Italic")

    flow = []
    lines = markdown_text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()

        # Headings: "# ...."
        m = re.match(r"^(#{1,6})\s+(.*)$", raw)
        if m:
            lvl = len(m.group(1)); txt = m.group(2).strip()
            flow.append(Paragraph(md_inline_to_reportlab(txt), H[lvl])); i += 1; continue

        # Ordered list
        if re.match(r"^\s*\d+\.\s+.+$", raw):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+.+$", lines[i]):
                item_text = re.sub(r"^\s*\d+\.\s+", "", lines[i]).strip()
                items.append(ListItem(Paragraph(md_inline_to_reportlab(item_text), BODY)))
                i += 1
            flow.append(ListFlowable(items, bulletType='1', leftPadding=18)); flow.append(Spacer(1, 6)); continue

        # Unordered list
        if re.match(r"^\s*-\s+.+$", raw):
            items = []
            while i < len(lines) and re.match(r"^\s*-\s+.+$", lines[i]):
                item_text = re.sub(r"^\s*-\s+", "", lines[i]).strip()
                items.append(ListItem(Paragraph(md_inline_to_reportlab(item_text), BODY)))
                i += 1
            flow.append(ListFlowable(items, bulletType='bullet', leftPadding=18)); flow.append(Spacer(1, 6)); continue

        # Blank line -> spacing
        if raw.strip() == "":
            flow.append(Spacer(1, 6)); i += 1; continue

        # Standalone _..._ line (kept as italics)
        if re.match(r"^_.*_$", raw.strip()):
            txt = raw.strip()[1:-1]
            flow.append(Paragraph(md_inline_to_reportlab(txt), ITAL)); i += 1; continue

        # Default: paragraph with inline formatting and <br/> support
        flow.append(Paragraph(md_inline_to_reportlab(raw), BODY)); i += 1

    doc.build(flow)
    return buf.getvalue()

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract links, fetch pages, build Markdown+PDF (keeps paragraphs, new lines, bullets, headings, bold/italic/underline).")
    ap.add_argument("page_url", help="The page to scan for links (e.g. https://example.com/path/page)")
    ap.add_argument("--same-origin", action="store_true", help="Keep only links with the same scheme+host as the page")
    ap.add_argument("--delay", type=float, default=0.35, help="Delay between page fetches (seconds)")
    ap.add_argument("--limit", type=int, default=0, help="Max linked pages to fetch (0 = no limit)")
    args = ap.parse_args()

    page_url = normalize_url(args.page_url)

    print(f"[+] Fetching index page: {page_url}")
    status, content, ctype = fetch(page_url)
    if status != 200 or not content:
        print(f"[-] Failed to fetch page ({status}).", file=sys.stderr)
        sys.exit(2)

    links = extract_links_from_page(page_url, content, args.same_origin)
    if args.limit > 0:
        links = links[:args.limit]
    print(f"[+] Found {len(links)} link(s).")

    scraped_md_parts: List[str] = []
    per_url_wordcount: Dict[str, int] = {}
    tiny_content_urls: List[Tuple[str, int, int]] = []  # (url, char_count, word_count)
    non_html_skipped: List[str] = []

    for idx, u in enumerate(links, 1):
        try:
            s, c, ct = fetch(u)
            if s != 200 or not c:
                print(f"[!] Skip ({s}): {u}")
                time.sleep(args.delay)
                continue

            # Gate: prefer HTML/XHTML; allow HTML-like by sniffing first bytes
            ct_lower = (ct or "").lower()
            is_html_header = ("text/html" in ct_lower) or ("application/xhtml+xml" in ct_lower)
            html_like = is_html_header or (c.strip().lower().startswith(b"<!doctype html") or b"<html" in c[:4096].lower())

            if not html_like:
                non_html_skipped.append(u)
                print(f"[!] Non-HTML content-type '{ct}'; skip: {u}")
                time.sleep(args.delay)
                continue

            md = html_to_markdown(c, u)

            # Count words/chars for summary: remove our md markers and inline tags
            md_for_count_lines = []
            for line in md.splitlines():
                if line.startswith("_Source:"):
                    continue
                md_for_count_lines.append(line)
            joined = "\n".join(md_for_count_lines).strip()
            joined_plain = re.sub(r"<[^>]+>", "", joined)                # drop <u>, <br/>
            joined_plain = re.sub(r"\*\*|__", "", joined_plain)          # drop **
            joined_plain = re.sub(r"(?:^|\s)[*_](.+?)[*_](?=\s|$)", r"\1", joined_plain)  # drop lone * or _
            joined_plain = re.sub(r"\s+", " ", joined_plain).strip()

            char_count = len(joined_plain)
            word_count = len(re.findall(r"\b\w+\b", joined_plain))

            if char_count < 10:
                tiny_content_urls.append((u, char_count, word_count))

            per_url_wordcount[u] = word_count
            scraped_md_parts.append(md)

            print(f"[{idx}/{len(links)}] OK ({word_count} words): {u}")
            time.sleep(args.delay)
        except Exception as e:
            print(f"[!] Error: {u} ({e})")
            continue

    if not scraped_md_parts:
        print("[-] Nothing extracted from linked pages.", file=sys.stderr)
        sys.exit(3)

    # Summary preface
    total_found = len(links)
    total_scraped = len(scraped_md_parts)
    total_tiny_chars = len(tiny_content_urls)
    total_tiny_words = sum(1 for _, _, w in tiny_content_urls if w < 10)

    summary_lines = []
    summary_lines.append("# Crawl Summary")
    summary_lines.append("")
    summary_lines.append(f"- How many urls found: **{total_found}**")
    summary_lines.append(f"- How many urls scraped: **{total_scraped}**")
    summary_lines.append(f"- How many urls has less than 10 characters: **{total_tiny_chars}**")
    summary_lines.append(f"- How many urls has less than 10 words: **{total_tiny_words}**")
    summary_lines.append("")
    summary_lines.append("## URLs scraped and word counts")
    for u, wc in sorted(per_url_wordcount.items(), key=lambda kv: (-kv[1], kv[0])):
        summary_lines.append(f"- {u} — {wc} words")

    if tiny_content_urls:
        summary_lines.append("")
        summary_lines.append("## URLs flagged as tiny (chars, words)")
        for u, cc, wc in tiny_content_urls:
            summary_lines.append(f"- {u} — {cc} chars, {wc} words")

    if non_html_skipped:
        summary_lines.append("")
        summary_lines.append("## Skipped non-HTML (Content-Type not HTML)")
        for u in non_html_skipped[:50]:
            summary_lines.append(f"- {u}")
        if len(non_html_skipped) > 50:
            summary_lines.append(f"- ... and {len(non_html_skipped)-50} more")

    summary_md = "\n".join(summary_lines).strip() + "\n"
    combined_md = summary_md + "\n---\n\n" + "\n\n---\n\n".join(scraped_md_parts).strip() + "\n"

    # Ensure output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write outputs under ./output/
    stem = make_output_stem(page_url)
    md_path = os.path.join(OUTPUT_DIR, f"{stem}.md")
    pdf_path = os.path.join(OUTPUT_DIR, f"{stem}.pdf")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(combined_md)
    print(f"[+] Wrote Markdown: {md_path}")

    pdf_bytes = markdown_to_pdf_bytes(combined_md)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"[+] Wrote PDF: {pdf_path}")

if __name__ == "__main__":
    main()