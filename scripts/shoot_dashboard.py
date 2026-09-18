#!/usr/bin/env python3
"""SIH26145 — render the operator console headlessly and capture it.

Starts the API and the static server, drives synthetic windows through /score so
the console has live telemetry, then screenshots it at desktop and narrow widths
and reports any console errors / failed requests. Used to check the UI actually
renders rather than assuming it does.

Usage: python scripts/shoot_dashboard.py [--out data/shots]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
API = "http://127.0.0.1:8200"
DASH = "http://127.0.0.1:8401"


def wait_http(url: str, tries: int = 60) -> bool:
    import urllib.request
    for _ in range(tries):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:                                    # noqa: BLE001
            time.sleep(0.5)
    return False


def _chromium() -> str | None:
    """Reuse whatever chromium is already cached.

    The pip-installed playwright pins a browser build number that may differ from
    what is on disk; rather than downloading another ~150MB we point it at the
    existing binary. Returns None to fall back to playwright's own default.
    """
    root = Path.home() / ".cache/ms-playwright"
    cands = sorted(root.glob("chromium-*/chrome-linux64/chrome")) + \
        sorted(root.glob("chromium_headless_shell-*/chrome-headless-shell-linux64/"
                         "chrome-headless-shell"))
    return str(cands[-1]) if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/shots")
    ap.add_argument("--keep-open", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    procs = [
        subprocess.Popen([PY, "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1",
                          "--port", "8200"], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([PY, "-m", "http.server", "8401", "--bind", "127.0.0.1",
                          "--directory", "dashboard"], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    ]
    try:
        if not wait_http(f"{API}/health"):
            raise SystemExit("API did not come up")
        if not wait_http(DASH):
            raise SystemExit("dashboard server did not come up")

        # give the console telemetry to draw
        subprocess.run([PY, str(ROOT / "scripts/drive_demo.py"),
                        "--rounds", "2", "--delay", "0.01"],
                       cwd=ROOT, stdout=subprocess.DEVNULL, check=False)

        from playwright.sync_api import sync_playwright
        errors, failed = [], []
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=_chromium())
            for name, w, h in (("console-desktop", 1680, 1500),
                               ("console-laptop", 1280, 1400),
                               ("console-narrow", 720, 1700)):
                page = b.new_page(viewport={"width": w, "height": h},
                                  device_scale_factor=1)
                page.on("console", lambda m: errors.append(f"{m.type}: {m.text}")
                        if m.type in ("error", "warning") else None)
                page.on("requestfailed",
                        lambda r: failed.append(f"{r.method} {r.url} {r.failure}"))
                page.goto(DASH, wait_until="networkidle")
                page.wait_for_timeout(2500)
                # a few more windows so the socket path is exercised while open
                subprocess.run([PY, str(ROOT / "scripts/drive_demo.py"),
                                "--rounds", "1", "--delay", "0.01"],
                               cwd=ROOT, stdout=subprocess.DEVNULL, check=False)
                page.wait_for_timeout(1500)
                page.screenshot(path=str(out / f"{name}.png"), full_page=True)
                print(f"  wrote {out / (name + '.png')}")
                if name == "console-desktop":
                    for probe in ("#idx-value", "#s-scored", "#feed tr", "#trend polyline",
                                  "#classbars .cbrow", "#diode-verdict", "#families .famrow"):
                        n = page.locator(probe).count()
                        txt = ""
                        if n:
                            try:      # SVG nodes have no inner_text
                                txt = page.locator(probe).first.inner_text()[:40]
                            except Exception:                # noqa: BLE001
                                txt = "<svg>"
                        print(f"    {probe:<26} count={n:<4} {txt!r}")
                page.close()
            b.close()
        if errors:
            print("\nconsole messages:")
            for e in dict.fromkeys(errors):
                print("   ", e)
        if failed:
            print("\nfailed requests:")
            for f in dict.fromkeys(failed):
                print("   ", f)
        if not errors and not failed:
            print("\nno console errors, no failed requests")
    finally:
        if not args.keep_open:
            for pr in procs:
                pr.terminate()
            for pr in procs:
                try:
                    pr.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pr.kill()


if __name__ == "__main__":
    sys.exit(main())
