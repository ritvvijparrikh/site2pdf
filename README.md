## Setup

python3.12 -m venv venv

source venv/bin/activate

pip3 install -r requirements.txt

## Running the scripts

# Crawl links found directly on a page
python3 linksOnPage2pdf.py https://example.com --same-origin

# Crawl links listed in a sitemap.xml
# Pass a sitemap.xml directly
python3 linksInSitemap2pdf.py https://example.com/sitemap.xml

# Or let the script discover the sitemap from the site's root
python3 linksInSitemap2pdf.py https://example.com
