# Live Camera Scanner — SmartFreshAI

User-facing live camera scanning experience: point a phone (or webcam) at a
fruit, let YOLO detect it, wait for a **stable** detection, auto-capture one
frozen frame, run the full production inference pipeline exactly once, and
show freshness + shelf life + nutrition on a result card.

---

## 1. Architecture

```
frontend/                     React 18 + Vite (JavaScript/JSX only, no TypeScript)
  index.html
  vite.config.js              dev server + vitest config
  .env.example                VITE_API_BASE_URL template
  src/
    main.jsx                  React root
    App.jsx                   screen state machine + health probe
    config/scannerConfig.js   ALL scanner magic numbers (single source)
    components/
      CameraScanner.jsx       camera lifecycle + detection loop + capture
      DetectionOverlay.jsx    bbox projection + status chip overlay
      ResultCard.jsx          result screen (freshness/shelf-life/nutrition)
      StorageSelector.jsx     ambient / refrigerated toggle (user context)
      ErrorView.jsx           friendly error screen
    services/
      camera.js               getUserMedia + frame capture helpers
      inferenceApi.js         API client (health / preview / full inference)
      detectionService.js     stability gate state machine (pure, no DOM)
      nutritionService.js     static nutrition DB access
    data/
      nutrition.json          typical nutrition per 100 g for all 10 classes
      fruitEmojis.js          display glyphs per fruit
    styles/styles.css         mobile-first styling (plain CSS custom props)
    testUtils/                jsdom render helper + setup polyfill
```

Backend (unchanged behaviour; one small additive endpoint):

* `POST /api/v1/inference/image` — existing FULL pipeline (YOLO → crop →
  EfficientNet freshness → stabilization → shelf life). Called **exactly once**
  per scan.
* `POST /api/v1/detection/preview` — **new**, lightweight, YOLO-only endpoint
  used by the live preview loop. Runs no freshness/shelf-life work.
* `GET /health` — readiness probe before camera access is offered.

Data honesty contract (also enforced in the UI copy):

| Value        | Source                                                        |
|--------------|---------------------------------------------------------------|
| Fruit ID     | Computer vision (YOLO11n)                                     |
| Freshness    | ML (EfficientNet-B0) — supported fruit classes only           |
| Shelf life   | Deterministic heuristic over metadata + freshness + storage   |
| Nutrition    | Static database (`nutrition.json`), typical values per 100 g  |

The UI never claims calories/freshness are extracted from image pixels.

## 2. Camera flow

1. `App` probes `GET /health`. If unreachable → friendly "server unavailable"
   screen with retry.
2. `CameraScanner` mounts and calls
   `navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false })`.
3. The stream is attached to a local `<video playsInline muted autoPlay>`
   element. Frames stay on-device except for the throttled preview JPEGs.
4. On unmount / Scan Again every track is stopped and intervals cleared — no
   leaked MediaStreams or duplicate loops.

Handled failure modes:

| Condition                    | User message                                        |
|------------------------------|-----------------------------------------------------|
| Permission denied            | "Camera access is required to scan a fruit."        |
| No camera / no devices       | "No camera was found on this device."               |
| Other init failure           | "Could not start the camera. Please try again."     |

Note: browsers only expose `getUserMedia` in secure contexts (HTTPS or
`localhost`). See §10/§11.

## 3. Stable detection logic

The preview loop feeds raw detections into `DetectionStabilityGate`
(`src/services/detectionService.js`), a pure state machine driven entirely by
`src/config/scannerConfig.js`:

| Config                     | Default | Meaning                                          |
|----------------------------|---------|--------------------------------------------------|
| `MIN_DETECTION_CONFIDENCE` | 0.5     | Frontend gate on top of backend threshold (0.45) |
| `REQUIRED_STABLE_FRAMES`   | 4       | Consecutive same-fruit frames needed to lock     |
| `DETECTION_INTERVAL_MS`    | 300     | Preview request period (~3 req/s worst case)     |
| `MAX_BBOX_DRIFT_RATIO`     | 0.2     | Max bbox-centre drift (fraction of box size)     |
| `PREVIEW_WIDTH/HEIGHT`     | 320     | Downscaled preview JPEG dimension                |
| `MAX_CAPTURE_WIDTH/HEIGHT` | 640     | Frozen-frame upload resolution (= YOLO imgsz)    |
| `INFERENCE_TIMEOUT_MS`     | 30000   | Full-analysis abort timeout                      |

A capture fires only when ALL hold across consecutive frames:

1. Exactly ONE detection above `MIN_DETECTION_CONFIDENCE`.
2. Same fruit label as the previous frame (label-keyed).
3. Bounding-box centre drift <= `MAX_BBOX_DRIFT_RATIO` vs previous frame.
4. Streak length >= `REQUIRED_STABLE_FRAMES`.

Any miss (no fruit / different fruit / big jump) resets the streak, so a
shaky scene or an appearing/disappearing object never auto-captures.
Multiple valid detections produce "Please place one fruit in the frame." —
the scanner never silently picks one.

## 4. Capture process

1. Gate verdict = CAPTURE -> the current video frame is drawn once to an
   offscreen canvas at <=640 px (`captureFrame`) producing JPEG Blob + dataURL.
2. The live `<video>` is hidden and the frozen frame is shown instead; status
   becomes ANALYZING ("Analyzing..."). The preview interval is stopped.
3. Exactly ONE multipart POST goes to `/api/v1/inference/image`.
4. Success switches to the RESULT screen; failure shows the mapped friendly
   error and returns to LIVE on retry.
5. Concurrent inference requests are impossible: the loop stops during
   CAPTURING/ANALYZING and a pending-request flag guards overlapping previews.

## 5. API integration

`src/services/inferenceApi.js` is the ONLY place that knows URLs.

* Base URL: `import.meta.env.VITE_API_BASE_URL`, default
  `http://127.0.0.1:8000` (dev). Never hardcode localhost elsewhere.
* Full analysis body: `FormData { image: capture.jpg, storage_condition }`.
  No explicit Content-Type header is set — the browser generates the
  multipart boundary.
* All responses resolve to `{ ok, data?, error?, code? }`; network/5xx errors
  map to fixed user-facing strings ("SmartFreshAI server is unavailable.",
  "Analysis took too long. Please try again.", ...). Raw tracebacks never
  reach the UI.
* CORS: FastAPI allows the configured dev origins (`http://localhost:5173`,
  `http://127.0.0.1:5173`); see `src/api/app.py`.

## 6. Freshness states

Rendered straight from the backend class — no remapping to "fresh":

| Backend class | Display               | Confidence shown |
|---------------|-----------------------|------------------|
| `fresh`       | Fresh                 | model confidence |
| `stale`       | Stale                 | model confidence |
| `rotten`      | Rotten                | model confidence |
| `uncertain`   | Freshness uncertain   | none             |
| `unsupported` | Freshness unavailable | none             |
| `unknown`     | Freshness unknown     | none             |

`uncertain`/`unsupported` are NEVER upgraded to fresh, and no confidence is
fabricated when the backend sends null.

## 7. Shelf-life semantics

Consumed verbatim from the response's `shelf_life` object (`remaining_days`,
`typical_min_days`, `typical_max_days`, `unit`, `basis`, `storage_condition`,
`explanation`, `shelf_life_status`). The frontend never recalculates shelf
life; the backend remains the single source of truth.

* `estimated` + numeric `remaining_days` → "~N days" plus the typical range.
* `expired` → explicit "Expired" banner (backend semantics).
* `remaining_days: null` → status text ("Not estimated"/"Not available") —
  never "0 days".
* Missing shelf_life object → "Remaining shelf life cannot be reliably
  estimated."

> Shelf life is a deterministic heuristic over typical metadata ranges,
> freshness class and user-provided storage context. It is NOT a validated
> expiry prediction for the individual item.

## 8. Nutrition data semantics

`src/data/nutrition.json` holds typical USDA FoodData Central values per
100 g edible portion for all 10 detector classes (apple, grape, kiwi, mango,
orange, strawberry, banana, cherry, chickoo, guava), each with a `source_ref`
provenance. `nutritionService.getNutrition(fruit)` performs a plain keyed
lookup; unknown fruits render "Nutrition data unavailable".

> Nutrition is a database lookup keyed by detected identity — it is NOT
> inferred from the captured pixels. Values are typical reference values for
> the raw edible portion, not per-item measurements.

## 9. Camera permissions

Requested only after the health probe passes. Denial or absence of hardware
surfaces the messages in §2 with an inline Retry that re-runs `getUserMedia`
without unmounting the scanner. Nothing is uploaded until a stable lock
triggers capture.

## 10. Local development

```bash
# backend
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# frontend
cd frontend
npm install
npm run dev          # http://localhost:5173 (binds 0.0.0.0)
npm test             # vitest suite
npm run build        # production bundle -> dist/
```

## 11. Phone testing

`127.0.0.1` on the PHONE is the phone itself. To test on a physical device:

1. Find the dev machine's LAN IP (`ipconfig`).
2. Run uvicorn bound to `0.0.0.0`.
3. Create `frontend/.env.local`: `VITE_API_BASE_URL=http://<LAN-IP>:8000`.
4. Allow ports 8000/5173 through Windows Firewall.
5. Open `http://<LAN-IP>:5173` on the phone. Camera access needs HTTPS or
   localhost; on Android use
   `chrome://flags#unsafely-treat-insecure-origin-as-secure` for LAN HTTP,
   or serve the built bundle over HTTPS.

Manual verification checklist (real device):

| # | Case         | Expected                                                     |
|---|--------------|--------------------------------------------------------------|
| 1 | Fresh apple  | Apple detected → fresh → shelf life estimated → apple row    |
| 2 | Rotten apple | Rotten → Expired banner                                      |
| 3 | Orange       | ML-supported freshness → shelf life → nutrition              |
| 4 | Banana       | same as above                                                |
| 5 | Grape        | freshness unsupported → honest text, nutrition still shown   |
| 6 | Two fruits   | "Please place one fruit in the frame."                       |
| 7 | Empty scene  | "No fruit detected. Move the fruit into the frame."          |
| 8 | Scan Again   | camera restarts cleanly, all counters reset                  |

Unit tests cover the logic around the camera (stability gate, API client,
result rendering); actual camera hardware behaviour must be verified manually
as above — a jsdom test cannot prove a webcam works.

## 12. Known limitations

* Live preview requires HTTPS or localhost (browser security).
* The stability gate is label+bbox-based, not a full visual tracker.
* One scan analyses one dominant fruit; multi-fruit selection is out of scope.
* Shelf-life numbers are heuristic estimates, not measurements.
* Nutrition is reference-table data, not per-item analysis.


