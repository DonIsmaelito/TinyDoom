"""Checks of the demo page in a real browser, on random-weight models and synthetic episodes.

    pip install playwright        # then, if Chrome is not installed: playwright install chromium
    pytest test_demo_page.py -s   # -s prints the measured step times

Skipped when Playwright, a browser for it, or onnxruntime is missing.
"""

import functools
import http.server
import shutil
import statistics
import threading
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest

sync_api = pytest.importorskip("playwright.sync_api")

from test_demo import MODELS, export_random_demo  # noqa: E402

DEMO_DIR = Path(__file__).parent / "demo"
STATIC_FILES = ("index.html", "app.js", "style.css", "vendor")
# The game runs 35 ticks a second and the data uses 4 ticks per action: 114 ms per step.
REAL_TIME_MS = 1000 / (35 / 4)
SERIES = MODELS + ("repeat_last_frame",)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """The page from demo/ with a fresh export, served on localhost."""
    work_dir = tmp_path_factory.mktemp("page")
    site_dir = work_dir / "site"
    site_dir.mkdir()
    for name in STATIC_FILES:
        source = DEMO_DIR / name
        if source.is_dir():
            shutil.copytree(source, site_dir / name)
        else:
            shutil.copy(source, site_dir / name)
    manifest, _, episodes = export_random_demo(work_dir, site_dir)
    handler = functools.partial(QuietHandler, directory=str(site_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {
        "url": f"http://127.0.0.1:{server.server_address[1]}/index.html",
        "manifest": manifest,
        "episodes": {ep.episode_id: ep for ep in episodes},
    }
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        errors = []
        for options in ({}, {"channel": "chrome"}):
            try:
                browser = playwright.chromium.launch(**options)
                break
            except Exception as error:
                errors.append(str(error).splitlines()[0])
        else:
            pytest.skip(f"No browser for Playwright: {errors}")
        yield browser
        browser.close()


@pytest.fixture
def context(browser):
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    context.set_default_timeout(60_000)
    yield context
    context.close()


def open_page(context, site, query="?autoplay=0&idle=0"):
    page = context.new_page()
    page.goto(site["url"] + query)
    page.evaluate("() => window.tinydoom.ready")
    return page


def state(page):
    return page.evaluate("() => window.tinydoom.state()")


def click_step(page, times=1):
    """Press the Step button and wait for each step to be drawn."""
    for _ in range(times):
        t = state(page)["t"]
        page.click("#step")
        page.wait_for_function("t => window.tinydoom.state().t !== t", arg=t)


def pictures(page):
    """What every panel shows right now, as PNG data, with the step counter."""
    return page.evaluate(
        """() => ({
            panels: [...document.querySelectorAll('#panels canvas')].map((c) => c.toDataURL()),
            steps: document.getElementById('steps').textContent,
        })"""
    )


def test_page_error_matches_the_fixture(context, site):
    """With a real frame every 10 steps, the page's MAE at 1, 5, and 10 is evaluate.py's.

    The page and PyTorch agree to about 1e-7. A wrong action, a skipped warm-up frame, or a
    memory that is not carried each move a number by more than 1e-4 on these models.
    """
    fixture = site["manifest"]["fixture"]
    page = open_page(context, site)
    measured = page.evaluate("interval => window.tinydoom.measure(interval)", fixture["interval"])
    assert measured["rollouts"] == fixture["rollouts"] > 0
    worst = 0.0
    for name in SERIES:
        for step in (1, 5, 10):
            difference = abs(measured["mae"][name][step] - fixture[name][str(step)]["mae"])
            worst = max(worst, difference)
            assert difference < 1e-5, (name, step)
    print(f"\nlargest difference between the page's MAE and the fixture: {worst:.2e}")


def test_page_decodes_the_real_frames_exactly(context, site):
    page = open_page(context, site)
    for index, entry in enumerate(site["manifest"]["episodes"]):
        frames = site["episodes"][entry["id"]].frames
        for t in (0, 17, len(frames) - 1):
            got = page.evaluate("([i, t]) => window.tinydoom.frameBytes(i, t)", [index, t])
            assert np.array_equal(np.array(got, dtype=np.uint8).reshape(frames[t].shape), frames[t])


def test_reset_reproduces_the_same_frames(context, site):
    page = open_page(context, site)
    start = state(page)
    assert start["playing"] is False and start["k"] is None
    first = pictures(page)
    assert len(first["panels"]) == 1 + len(MODELS) and first["steps"] == "0"
    # At the start every panel shows the same real frame.
    assert len(set(first["panels"])) == 1

    runs = []
    for _ in range(3):
        click_step(page, 12)
        after = state(page)
        assert after["t"] == start["t"] + 12 and after["k"] == 2
        runs.append(pictures(page))
        page.click("#reset")
        page.wait_for_function("t => window.tinydoom.state().t === t", arg=start["t"])
        assert pictures(page) == first
    assert runs[0] == runs[1] == runs[2]
    assert runs[0]["steps"] == "2"
    # The models' pictures are their own, not copies of the real frame or of each other.
    assert len(set(runs[0]["panels"])) == 1 + len(MODELS)


def test_controls_change_how_the_models_are_fed(context, site):
    page = open_page(context, site)
    page.select_option("#interval", "1")
    click_step(page, 4)
    assert state(page)["k"] == 1 and state(page)["interval"] == 1
    page.select_option("#interval", "0")
    click_step(page, 14)
    assert state(page)["k"] == 15
    assert page.inner_text("#steps") == "15"

    page.select_option("#speed", "8.75")
    assert state(page)["speed"] == 8.75
    scenes = page.locator("#scene option").count()
    assert scenes == sum(len(e["starts"]) for e in site["manifest"]["episodes"])
    page.select_option("#scene", str(scenes - 1))
    page.wait_for_function("s => window.tinydoom.state().scene === s", arg=scenes - 1)
    last = site["manifest"]["episodes"][-1]
    assert state(page)["episode"] == last["id"] and state(page)["t"] == last["starts"][-1]

    page.click("#play")
    page.wait_for_function("t => window.tinydoom.state().t > t + 3", arg=state(page)["t"])
    assert page.inner_text("#play") == "Pause"
    page.click("#play")
    assert state(page)["playing"] is False and page.inner_text("#play") == "Play"


def test_no_request_leaves_localhost(context, site):
    urls = []
    context.on("request", lambda request: urls.append(request.url))
    page = open_page(context, site, "?speed=8.75&idle=0")
    page.wait_for_function("() => window.tinydoom.timings().length >= 25")
    hosts = {urlparse(url).hostname for url in urls if urlparse(url).scheme in ("http", "https")}
    assert hosts == {"127.0.0.1"}, sorted(urls)
    assert all(urlparse(url).scheme in ("http", "data", "blob") for url in urls), sorted(urls)
    names = {Path(urlparse(url).path).name for url in urls}
    assert {"manifest.json", "cnn.onnx", "convgru.onnx", "ort-wasm-simd-threaded.wasm"} <= names


def test_both_models_and_drawing_keep_up_with_the_game(context, site):
    page = open_page(context, site, "?speed=8.75&idle=0")
    page.wait_for_function("() => window.tinydoom.timings().length >= 80")
    times = sorted(page.evaluate("() => window.tinydoom.timings()"))
    median = statistics.median(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    print(
        f"\n{len(times)} steps with a real frame every 10: median {median:.1f} ms, "
        f"95th percentile {p95:.1f} ms, slowest {times[-1]:.1f} ms, "
        f"real time is {REAL_TIME_MS:.1f} ms per step"
    )
    assert median < REAL_TIME_MS
    assert p95 < REAL_TIME_MS


def test_paused_page_plays_again_by_itself(context, site):
    page = open_page(context, site, "?autoplay=0&idle=2")
    assert state(page)["playing"] is False
    assert "plays again by itself after 2 seconds" in page.inner_text("#idle-note")
    page.wait_for_function("() => window.tinydoom.state().playing", timeout=10_000)
    page.wait_for_function("() => window.tinydoom.state().k > 0", timeout=10_000)


def test_clip_is_shown_when_the_models_cannot_load(context, site):
    context.route("**/*.onnx", lambda route: route.abort())
    page = context.new_page()
    page.goto(site["url"])
    page.wait_for_function("() => document.getElementById('status').classList.contains('error')")
    assert "recording of the same comparison" in page.inner_text("#status")
    assert page.is_visible("#clip") and not page.is_visible("#panels")
    assert not page.is_visible("#compare") and not page.is_visible("#chart-section")
    page.wait_for_function("() => document.getElementById('clip').naturalWidth > 0")


def test_page_says_so_when_nothing_was_exported(context, site):
    context.route("**/manifest.json", lambda route: route.fulfill(status=404, body="not found"))
    page = context.new_page()
    page.goto(site["url"])
    page.wait_for_function("() => document.getElementById('status').classList.contains('error')")
    assert "python export_demo.py" in page.inner_text("#status")
    assert not page.is_visible("#clip") and not page.is_visible("#panels")
