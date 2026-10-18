# Demo video script (`submissions/demo.mp4`, v2)

1920×1080, 60 fps, voice-over (ElevenLabs TTS) with burned-in captions. The guiding analogy is a rally co-driver with a pace-note book
(slide 3, called back at the switch demo). Every number is from `submissions/README.md`, the experiment log or a run shown on screen.
Ground truth appears for display and scoring only. No camera images are shown (dataset licence: no redistribution).

| time | shot | voice-over |
|---|---|---|
| 0:00.0 | Slide 1 · problem and result | Where is the race car, using only its cameras? camloc estimates position and heading ten times a second around Yas Marina, with no GNSS, no LiDAR, and no starting pose. On held-out night laps, where the map and every calibration exclude the scored lap, the position error is zero point five one metres, lateral zero point three, and ninety-five percent of poses are within one metre. These are held-out training laps, not a hidden-test result. |
| 0:30.1 | Slide 2 · why not odometry | Why not just odometry? Wheel speeds and steering alone drift more than a hundred metres, even from the true start. Three of the seven cameras, front left, front right and rear, carry the map fixes. |
| 0:44.6 | Slide 3 · the co-driver analogy mapped to the pipeline | Think of a rally co-driver with a pace-note book. Between landmarks, they count distance and turns from the car's own speed and steering. That is our odometry: smooth, but it drifts. When they recognise a landmark from the notes, they snap back to the map. That is our geo-referenced feature map, solved by PnP on each camera. Their judgement decides which sightings to trust and rejects look-alikes: cross-camera consensus, a statistical gate, and a Kalman filter. And dropped mid-stage with no position, they look around until sightings agree, commit provisionally, and change their mind if the evidence says so. |
| 1:27.3 | Slide 4 · methodology: why each choice + evidence | Every block earns its place with a measurement. We chose classical methods, not deep learning: we have one night of data, so there is nothing to overfit, and it runs deterministically in about twenty-five to forty milliseconds on a laptop CPU, against a hundred millisecond budget. The map gives absolute metres: it takes the error from over a hundred metres to under one. Odometry and the filter bridge the gaps: thirty seconds without any fix costs about two point seven metres, and it recovers. The gate makes wrong landmarks harmless: injecting two percent of fixes ten metres wrong leaves lap four unchanged. And strict validation is why these numbers can be trusted: the map and every calibration are rebuilt without the scored lap, and the official scorer does the scoring. |
| 2:19.4 | Terminal · live held-out lap 4 run + official scorer (s 0.5–11.5) | A live run on held-out lap four. The map and the calibration were rebuilt without that lap, and no start pose is given. |
| 2:30.4 | Terminal · live held-out lap 4 run + official scorer (s 34.6–55.6) | About a thousand frames, at around twenty-five milliseconds each, locked on the first frame. The official scorer reports zero point four six metres, and ninety-six point seven percent of poses within one metre on this lap. |
| 2:51.4 | Animation · lap 4 replay ×4 (GT display only) | Here is that lap replayed at four times speed. The blue estimate stays on the ground-truth ring, which is drawn for display only. The error stays well under a metre, with short spikes above it, and the running error ends at zero point four six metres. |
| 3:18.1 | Slide 5 · results | Across the three held-out laps: zero point six, zero point four six, and zero point four two metres. Each development step was kept only if the held-out error went down, from one point zero six to zero point five one metres. |
| 3:35.3 | Animation · no start pose, lap 6 @ 70 % (co-driver callback) | No start pose. At seventy percent of lap six, two cameras agree on a look-alike place, thirteen and a half metres off. Like a co-driver who realises they misread the landmark: within a third of a second, three rejected fixes agree with each other elsewhere, and the filter switches to them. Two accepted fixes then confirm it. |
| 3:59.0 | Video sheet · 16 mid-lap starts | Over sixteen mid-lap starts, every run locks within two seconds, and all sixteen end correct. |
| 4:06.8 | Slide 6 · limits and reproducibility | We know exactly where it degrades, and by how much. Without map fixes, the error grows to about two point seven metres over thirty seconds, and five over a minute. On the organisers' daylight sample, which has no ground truth, we only checked structure: it locks at zero point six seconds and covers every second, with few fixes. |
| 4:28.3 | Slide 6 · limits and reproducibility | And it all runs with one command, on both topic layouts, on a CPU. Code, experiment log and validation protocol are in the repository. |

Total: 4:39.0.

## Rebuild
```bash
.venv/bin/python -W ignore submissions/media/figs.py            # figures (needs the strict-fold runs in cache/)
node submissions/media/render_slides.mjs                          # slides.pdf + slide PNGs (DEMO_CHROME=<chrome-headless-shell>)
vhs submissions/media/terminal.tape                               # terminal take (runs submissions/media/heldout_lap4.sh)
.venv/bin/python -W ignore submissions/media/anim.py lap4 && .venv/bin/python -W ignore submissions/media/anim.py nostart
# then the demo-video cut (prep.py -> render.mjs -> build.py) with submissions/media/shots.json and theme.css
```
