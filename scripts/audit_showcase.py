"""Validate local evidence links and exported provenance without a browser."""
from pathlib import Path
from html.parser import HTMLParser
import json,re
ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.urls=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        for key in ['href','src','poster']:
            if key in a:self.urls.append(a[key])
def main():
    page=ROOT/'docs/showcase/index.html';p=Links();p.feed(page.read_text(encoding='utf-8'))
    urls=p.urls+re.findall(r"(?:\.\./|media/)[^\s'\"<>]+",(page.parent/'app.js').read_text(encoding='utf-8'))
    for url in urls:
        if url.startswith(('http:','https:','data:')):continue
        if url.startswith('#'):assert url=='#' or url[1:] in p.ids,url
        else:assert (page.parent/url.split('#')[0]).is_file(),url
    d=json.loads((page.parent/'data/overview.json').read_text());assert d['total_training_steps']==65536
    assert d['baseline']['strict_success'] is False
    assert d['student']['episodes']==197 and d['student']['unique_sources']==20
    assert (page.parent/'media/penspin-preview.gif').read_bytes().startswith(b'GIF')
    print(json.dumps(dict(local_links_checked=len(urls),actual_training_steps=d['total_training_steps'],strict_quality_claim_consistent=True)))
if __name__=='__main__':main()
