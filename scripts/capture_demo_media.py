"""Capture the demo page's screenshot and walkthrough video with headless Chrome.

    python capture_media.py <site dir> <out png> <out mp4> [scene index for the screenshot]

The video is Playwright's recording of the page while this script uses the real controls.
"""
import shutil, subprocess, sys, tempfile, time
from pathlib import Path
from playwright.sync_api import sync_playwright
import functools, threading, http.server

class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

def serve(directory):
    handler = functools.partial(_Quiet, directory=directory)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


site, png, mp4 = sys.argv[1], sys.argv[2], sys.argv[3]
scene = sys.argv[4] if len(sys.argv) > 4 else "0"
server = serve(site)
url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
size = {"width": 1440, "height": 900}
state = "() => window.tinydoom.state()"
# Optional sixth argument: a note stamped on the capture (not part of the page itself).
note = sys.argv[5] if len(sys.argv) > 5 else ""
BANNER = """(text) => {
  const b = document.createElement('p');
  b.textContent = text;
  b.style.cssText = 'margin:0 0 12px;padding:10px 14px;border:2px solid #ffc440;border-radius:8px;color:#ffc440;font-weight:700';
  document.querySelector('main').prepend(b);
}"""

def step(page, n):
    for _ in range(n):
        t = page.evaluate(state)["t"]
        page.click("#step")
        page.wait_for_function("t => window.tinydoom.state().t !== t", arg=t)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)

    # Screenshot: let the chart fill in, then stop a few steps into a stretch.
    page = browser.new_page(viewport=size)
    page.goto(url + "?speed=8.75&idle=0")
    page.evaluate("() => window.tinydoom.ready")
    if note: page.evaluate(BANNER, note)
    page.wait_for_function("() => window.tinydoom.timings().length >= 150", timeout=120000)
    page.click("#play")
    page.select_option("#scene", scene)
    page.wait_for_function("s => window.tinydoom.state().scene === Number(s) && window.tinydoom.state().k === null", arg=scene)
    step(page, 6)
    page.screenshot(path=png, full_page=True)
    print("screenshot at", page.evaluate(state))
    page.close()

    # Video.
    videos = tempfile.mkdtemp()
    context = browser.new_context(viewport=size, record_video_dir=videos, record_video_size=size)
    page = context.new_page()
    page.goto(url)
    page.evaluate("() => window.tinydoom.ready")
    if note: page.evaluate(BANNER, note)
    time.sleep(9)                                   # plays by itself, a real frame every 10 steps
    page.select_option("#speed", "8.75"); time.sleep(6)      # game speed
    page.select_option("#interval", "1"); time.sleep(5)      # a real frame every step
    page.select_option("#interval", "0"); time.sleep(8)      # never again
    page.select_option("#interval", "10"); page.select_option("#speed", "4"); time.sleep(4)
    page.click("#play"); time.sleep(1)                       # pause
    step(page, 4); time.sleep(1)                             # single steps
    page.click("#reset"); time.sleep(2)                      # reset
    page.select_option("#scene", scene); time.sleep(1.5)     # pick an episode
    page.click("#play"); time.sleep(6)
    page.click("#view-play"); time.sleep(0.5)
    for _ in range(12):
        page.evaluate("() => window.tinydoom.playStepOnce(0)")
        time.sleep(0.12)
    page.evaluate("() => window.tinydoom.playStepOnce(1)")
    page.evaluate("() => window.tinydoom.playStepOnce(2)")
    time.sleep(0.5)
    page.click("#play-reset"); time.sleep(1.5)
    print("video ends at", page.evaluate(state))
    path = page.video.path()
    context.close()
    browser.close()

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-movflags", "+faststart", "-an", mp4], check=True)
shutil.rmtree(videos)
print(Path(png).stat().st_size, Path(mp4).stat().st_size)
