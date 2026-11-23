# Vendored: ONNX Runtime Web 1.31.1

The demo page runs the models with ONNX Runtime Web. The files are stored here, unchanged, so
the page works without any network access beyond its own folder.

They come from the `dist/` folder of the npm package `onnxruntime-web`, version 1.31.1:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `ort.wasm.min.js` | 53,796 | `b8a78c408c40da65fdc8ba0846f7b2948e92a4c6d6b23cc8e4165e428ad95443` |
| `ort-wasm-simd-threaded.mjs` | 24,738 | `30ed0b5cd1182ede931fea448afb567d4dc705ec957af09f108a2be5f770dac3` |
| `ort-wasm-simd-threaded.wasm` | 14,848,852 | `4b884c203b0fa48d057be3332c65513ff85aa3c5bddd6946acc0fab7ecda001d` |

This is the CPU-only (WebAssembly) build. The page sets it to one thread, because a page on
GitHub Pages cannot use WebAssembly threads.

To update, install the new version with `npm install onnxruntime-web@<version>`, copy the same
three files from `node_modules/onnxruntime-web/dist/`, update this table, and run
`pytest test_demo_page.py`.

## License

The package declares the MIT license, and `ort.wasm.min.js` carries this notice:

    ONNX Runtime Web v1.31.1
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License.

MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
