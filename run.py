"""Portable entry point: python run.py demo | benchmark | test | doctor."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["demo","benchmark","test","doctor"])
    args,rest=p.parse_known_args()
    if args.command=="doctor":
        import platform
        print("Python:",sys.version,"\nOS:",platform.platform())
        for name in ("numpy","cv2","isaacsim"):
            print(name, "available" if importlib.util.find_spec(name) else "not available in this Python")
        print("Isaac must run with its own Python launcher on the Ubuntu RTX machine.")
    elif args.command=="test":
        import unittest
        suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    elif args.command=="benchmark":
        from drishti.experiment import main as run
        sys.argv=[sys.argv[0],*rest];run()
    else:
        import functools,http.server,webbrowser
        handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(ROOT/"demo"))
        with http.server.ThreadingHTTPServer(("127.0.0.1",8765),handler) as server:
            print("Open http://127.0.0.1:8765 — local synthetic planning replay",flush=True)
            webbrowser.open("http://127.0.0.1:8765")
            try:server.serve_forever()
            except KeyboardInterrupt:pass

if __name__=="__main__":main()
