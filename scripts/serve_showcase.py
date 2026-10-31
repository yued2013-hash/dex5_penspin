"""Serve the repository locally, so evidence links and videos work offline."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
import argparse,webbrowser

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8831);p.add_argument('--no-open',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];url=f'http://127.0.0.1:{a.port}/docs/showcase/'
    server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(SimpleHTTPRequestHandler,directory=str(root)))
    print('Dex5 showcase: '+url,flush=True)
    if not a.no_open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
