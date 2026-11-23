'use strict';

// TinyDoom demo page. Everything runs in the browser: the two models are ONNX files run by
// ONNX Runtime Web, and the real frames come from test episodes exported by export_demo.py.

(() => {
  const DEFAULTS = { interval: 10, speed: 4 };

  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // ---- Real frames -------------------------------------------------------------------------

  // One recorded episode. All of its frames are in one image, `columns` frames per row.
  class Episode {
    constructor(entry, sheet, manifest) {
      this.id = entry.id;
      this.actions = entry.actions;
      this.starts = entry.starts;
      this.count = entry.frames;
      this.size = manifest.frame_size;
      this.columns = manifest.sheet_columns;
      this.sheet = sheet;
      this.rgbaCache = new Array(this.count);
      this.chwCache = new Array(this.count);
    }

    // Frame t as RGBA bytes, for drawing.
    rgba(t) {
      if (!this.rgbaCache[t]) {
        const { size, columns, sheet } = this;
        const out = new Uint8ClampedArray(size * size * 4);
        const left = (t % columns) * size;
        const top = Math.floor(t / columns) * size;
        for (let y = 0; y < size; y++) {
          const from = ((top + y) * sheet.width + left) * 4;
          out.set(sheet.data.subarray(from, from + size * 4), y * size * 4);
        }
        this.rgbaCache[t] = out;
      }
      return this.rgbaCache[t];
    }

    // Frame t as the models take it: float32, channel by channel, values 0 to 1.
    chw(t) {
      if (!this.chwCache[t]) {
        const rgba = this.rgba(t);
        const n = this.size * this.size;
        const out = new Float32Array(3 * n);
        for (let i = 0; i < n; i++) {
          out[i] = rgba[4 * i] / 255;
          out[n + i] = rgba[4 * i + 1] / 255;
          out[2 * n + i] = rgba[4 * i + 2] / 255;
        }
        this.chwCache[t] = out;
      }
      return this.chwCache[t];
    }
  }

  async function decodeImage(blob) {
    try {
      return await createImageBitmap(blob, {
        colorSpaceConversion: 'none',
        premultiplyAlpha: 'none',
      });
    } catch (error) {
      const image = new Image();
      image.src = URL.createObjectURL(blob);
      await image.decode();
      return image;
    }
  }

  async function loadEpisode(entry, manifest) {
    const response = await fetch(entry.sheet);
    if (!response.ok) throw new Error(`${entry.sheet}: HTTP ${response.status}`);
    const image = await decodeImage(await response.blob());
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0);
    const sheet = context.getImageData(0, 0, canvas.width, canvas.height);
    return new Episode(entry, sheet, manifest);
  }

  // ---- Models ------------------------------------------------------------------------------

  // One model. It holds the frame it will predict from, and its memory if it has one.
  class StepModel {
    constructor(entry, session, size) {
      this.name = entry.name;
      this.label = entry.label;
      this.parameters = entry.parameters;
      this.contextFrames = entry.context_frames;
      this.hiddenShape = entry.hidden_shape;
      this.session = session;
      this.frameShape = [1, 3, size, size];
      this.frame = null;
      this.hidden = null;
    }

    async run(frame, action) {
      const feeds = {
        frame: new ort.Tensor('float32', frame, this.frameShape),
        action: new ort.Tensor('int64', BigInt64Array.from([BigInt(action)]), [1]),
      };
      if (this.hiddenShape) {
        feeds.hidden = new ort.Tensor('float32', this.hidden, this.hiddenShape);
      }
      const out = await this.session.run(feeds);
      if (this.hiddenShape) this.hidden = out.hidden_out.data;
      return out.next_frame.data;
    }

    // Start again from real frame t. A model with memory starts with an empty memory and
    // first observes the real frames just before t, with the actions that were taken.
    async showRealFrame(episode, t) {
      if (this.hiddenShape) {
        this.hidden = new Float32Array(this.hiddenShape.reduce((a, b) => a * b, 1));
        for (let i = Math.max(0, t - (this.contextFrames - 1)); i < t; i++) {
          await this.run(episode.chw(i), episode.actions[i]);
        }
      }
      this.frame = episode.chw(t);
    }

    // Predict the next frame from the model's own current frame.
    async predict(action) {
      this.frame = await this.run(this.frame, action);
      return this.frame;
    }
  }

  async function loadModel(entry, manifest) {
    const response = await fetch(entry.file);
    if (!response.ok) throw new Error(`${entry.file}: HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return new StepModel(entry, session, manifest.frame_size);
  }

  function meanAbsDiff(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum / a.length;
  }

  // ---- Playing an episode ------------------------------------------------------------------

  // Plays one episode from a start frame. The models predict from their own output and are
  // shown a real frame every `interval` steps (0 means never again after the start).
  class Player {
    constructor(models, episode, start, interval) {
      this.models = models;
      this.episode = episode;
      this.interval = interval;
      this.t = start; // the frame the panels show
      this.k = null; // steps since the models saw a real frame; null before the first one
      this.anchor = start; // the last real frame the models saw
    }

    // One step. Returns null when the episode has nothing left to play.
    async step() {
      const { episode, models } = this;
      const last = episode.count - 1;
      if (this.t >= last) return null;
      if (this.k === null || (this.interval > 0 && this.k >= this.interval)) {
        // Only start a stretch that fits, so every stretch has the same number of steps.
        if (this.interval > 0 && this.t + this.interval > last) return null;
        for (const model of models) await model.showRealFrame(episode, this.t);
        this.anchor = this.t;
        this.k = 0;
      }
      const action = episode.actions[this.t];
      const frames = {};
      for (const model of models) frames[model.name] = await model.predict(action);
      this.t += 1;
      this.k += 1;
      const real = episode.chw(this.t);
      const errors = { repeat_last_frame: meanAbsDiff(episode.chw(this.anchor), real) };
      for (const model of models) errors[model.name] = meanAbsDiff(frames[model.name], real);
      return { t: this.t, k: this.k, action, frames, errors };
    }
  }

  // Mean error at each number of steps since the last real frame.
  class ErrorStats {
    constructor() {
      this.sums = {};
      this.counts = [];
    }

    add(k, errors) {
      this.counts[k] = (this.counts[k] || 0) + 1;
      for (const [name, value] of Object.entries(errors)) {
        if (!this.sums[name]) this.sums[name] = [];
        this.sums[name][k] = (this.sums[name][k] || 0) + value;
      }
    }

    // means()[name][k] is the mean error of `name` at step k; index 0 is unused.
    means() {
      const out = {};
      for (const [name, sums] of Object.entries(this.sums)) {
        out[name] = sums.map((sum, k) => sum / this.counts[k]);
      }
      return out;
    }
  }

  // ---- The page ----------------------------------------------------------------------------

  const app = {
    manifest: null,
    episodes: [],
    models: [],
    scenes: [], // { episode: index into episodes, start: frame }
    scene: 0,
    player: null,
    stats: new ErrorStats(),
    playing: false,
    interval: DEFAULTS.interval,
    speed: DEFAULTS.speed,
    loop: 0,
    timings: [],
    panels: {},
  };

  // The models run one call at a time. Everything that uses them goes through this queue.
  let queue = Promise.resolve();
  function enqueue(job) {
    const result = queue.then(job);
    queue = result.catch(() => {});
    return result;
  }

  function parameterText(count) {
    return count >= 1e6 ? `${(count / 1e6).toFixed(1)}M` : `${Math.round(count / 1e3)}K`;
  }

  function addPanel(series, name, about) {
    const size = app.manifest.frame_size;
    const figure = document.createElement('figure');
    figure.className = 'panel';
    figure.dataset.series = series;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const caption = document.createElement('figcaption');
    const title = document.createElement('span');
    title.className = 'name';
    title.textContent = name;
    const detail = document.createElement('span');
    detail.className = 'about';
    detail.textContent = about;
    caption.append(title, detail);
    figure.append(canvas, caption);
    $('panels').append(figure);
    const context = canvas.getContext('2d');
    app.panels[series] = { figure, context, image: context.createImageData(size, size) };
  }

  function buildPanels() {
    addPanel('real', 'Real game', 'what the game actually showed');
    for (const model of app.models) {
      const memory = model.hiddenShape
        ? 'remembers earlier frames'
        : 'sees only its current frame';
      addPanel(model.name, model.label, `${parameterText(model.parameters)} parameters, ${memory}`);
    }
  }

  function drawBytes(series, rgba) {
    const { context, image } = app.panels[series];
    image.data.set(rgba);
    context.putImageData(image, 0, 0);
  }

  function drawFloat(series, chw) {
    const { context, image } = app.panels[series];
    const data = image.data;
    const n = chw.length / 3;
    for (let i = 0; i < n; i++) {
      data[4 * i] = chw[i] * 255;
      data[4 * i + 1] = chw[n + i] * 255;
      data[4 * i + 2] = chw[2 * n + i] * 255;
      data[4 * i + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }

  // Show the real frame the scene starts from in every panel.
  function renderStart() {
    const { episode, t } = app.player;
    drawBytes('real', episode.rgba(t));
    for (const model of app.models) drawBytes(model.name, episode.rgba(t));
  }

  function renderStep(result) {
    drawBytes('real', app.player.episode.rgba(result.t));
    for (const model of app.models) drawFloat(model.name, result.frames[model.name]);
  }

  function startScene(index) {
    app.scene = index;
    const scene = app.scenes[index];
    app.player = new Player(app.models, app.episodes[scene.episode], scene.start, app.interval);
    renderStart();
  }

  // After an episode ends, play the next episode from its best start frame.
  function nextScene() {
    const episode = (app.scenes[app.scene].episode + 1) % app.episodes.length;
    return app.scenes.findIndex((scene) => scene.episode === episode);
  }

  async function tick() {
    const started = performance.now();
    let result = await app.player.step();
    if (result === null) {
      startScene(nextScene());
      result = await app.player.step();
    }
    app.stats.add(result.k, result.errors);
    renderStep(result);
    app.timings.push(performance.now() - started);
    if (app.timings.length > 2000) app.timings.shift();
  }

  async function playLoop(loop) {
    while (app.playing && loop === app.loop) {
      const started = performance.now();
      await enqueue(tick);
      await sleep(Math.max(0, 1000 / app.speed - (performance.now() - started)));
    }
  }

  function setPlaying(playing) {
    app.playing = playing;
    app.loop += 1;
    if (playing) playLoop(app.loop).catch(fail);
  }

  // ---- Loading -----------------------------------------------------------------------------

  function setStatus(text, isError) {
    $('status').textContent = text;
    $('status').classList.toggle('error', Boolean(isError));
  }

  function fail(error) {
    console.error(error);
    setPlaying(false);
    $('panels').hidden = true;
    if (app.manifest && app.manifest.clip) {
      showClip();
      setStatus(
        'The models could not run in this browser, so this is a recording of the same ' +
          `comparison. (${error.message})`,
        true
      );
    } else {
      setStatus(`The demo could not start. (${error.message})`, true);
    }
  }

  function showClip() {
    const clip = $('clip');
    clip.onerror = () => {
      clip.hidden = true;
    };
    clip.src = app.manifest.clip;
    clip.hidden = false;
  }

  async function load() {
    const response = await fetch('manifest.json');
    if (!response.ok) {
      throw new Error('No exported models here yet. Run "python export_demo.py" first.');
    }
    app.manifest = await response.json();
    showClip();

    if (typeof ort === 'undefined') throw new Error('ONNX Runtime Web did not load.');
    // GitHub Pages cannot serve the headers that WebAssembly threads need, so use one thread.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = new URL('vendor/', document.baseURI).href;

    for (const entry of app.manifest.models) {
      app.models.push(await loadModel(entry, app.manifest));
    }
    app.episodes = await Promise.all(
      app.manifest.episodes.map((entry) => loadEpisode(entry, app.manifest))
    );
    app.episodes.forEach((episode, index) => {
      for (const start of episode.starts) app.scenes.push({ episode: index, start });
    });
    if (!app.scenes.length) throw new Error('The export has no episodes to play.');

    buildPanels();
    startScene(0);
    $('clip').hidden = true;
    $('panels').hidden = false;
    setStatus('');
  }

  // ---- For the tests in test_demo_page.py --------------------------------------------------

  // Plays every exported episode the way the fixture in manifest.json was computed, without
  // drawing, and returns the mean error at each step. Playback goes back to its start.
  function measure(interval) {
    return enqueue(async () => {
      const stats = new ErrorStats();
      let rollouts = 0;
      for (const episode of app.episodes) {
        const player = new Player(app.models, episode, app.manifest.fixture.first_frame, interval);
        for (let result = await player.step(); result; result = await player.step()) {
          stats.add(result.k, result.errors);
          if (result.k === 1) rollouts += 1;
        }
      }
      startScene(app.scene);
      return { rollouts, mae: stats.means() };
    });
  }

  const params = new URLSearchParams(location.search);
  const ready = load().then(() => {
    if (params.get('autoplay') !== '0') setPlaying(true);
    return true;
  });
  ready.catch(fail);

  window.tinydoom = {
    ready,
    measure,
    frameBytes: (episode, t) => {
      const rgba = app.episodes[episode].rgba(t);
      return Array.from(rgba.filter((_, i) => i % 4 !== 3));
    },
    state: () => ({
      playing: app.playing,
      scene: app.scene,
      episode: app.player.episode.id,
      t: app.player.t,
      k: app.player.k,
      interval: app.interval,
      speed: app.speed,
    }),
    timings: () => app.timings.slice(),
  };
})();
