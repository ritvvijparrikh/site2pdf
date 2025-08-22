## Setup

python3.12 -m venv venv

source venv/bin/activate

pip3 install -r requirements.txt

## Running the scripts

python3 linksInSitemap2pdf.py https://invertedpassion.com/ --limit 10

python3 linksOnPage2pdf.py https://invertedpassion.com --same-origin
