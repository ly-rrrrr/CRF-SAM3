# Frame Gallery Agent

A small local tool for browsing raw dataset frames by `video_id`.

It has:

- a Python backend that searches known dataset roots and serves frame files
- a React frontend that queries the backend and renders a frame gallery

Supported dataset roots by default:

- `davis17/{train,valid}/JPEGImages`
- `mevis/{train,valid,valid_u}/JPEGImages`
- `ref-youtube-ros/valid/JPEGImages`
- `ref-youtube-ros-21/21/{train,valid}/JPEGImages`

The backend can search across all of them, so a query like `05d7715782` will return the matching raw frame directory if it exists.

## Backend

```bash
python tools/frame_gallery_agent/backend/server.py
```

Optional:

```bash
python tools/frame_gallery_agent/backend/server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --data-root /mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas
```

## Frontend

```bash
cd tools/frame_gallery_agent/frontend
npm install
npm run dev
```

The Vite dev server proxies `/api/*` and `/frame/*` to `http://127.0.0.1:8765` by default.

## Optional production build

Build the frontend:

```bash
cd tools/frame_gallery_agent/frontend
npm install
npm run build
```

After that, the backend can also serve the built frontend from:

- `tools/frame_gallery_agent/frontend/dist`

Then you only need:

```bash
python tools/frame_gallery_agent/backend/server.py
```
