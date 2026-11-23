Demo (runs in the browser, offline)

1. Open your Colab project folder (the one with the lab folder). Drag export_model.py into it.
2. Run:   !python export_model.py --cap 800
   (about 1-2 minutes; it trains the detectors on your real HC3 data and writes model.js)
3. In the Colab file panel, right-click model.js and Download it.
4. On your laptop put detector.html and the NEW model.js in the same folder.
   Replace the demo model.js that came in this zip.
5. Double-click detector.html (Chrome or Edge). No install, no internet needed.

The yellow "demo model" banner disappears once the real model.js is in place.
