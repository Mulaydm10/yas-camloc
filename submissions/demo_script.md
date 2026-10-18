# Demo video script (`submissions/demo.mp4`)

1920×1080, 60 fps, voice-over (ElevenLabs TTS) with burned-in captions. Every number is from `submissions/README.md`, the experiment log
or a run shown on screen. Ground truth appears for display and scoring only. No camera images are shown (dataset licence: no redistribution).

| time | shot | voice-over |
|---|---|---|
| 0:00.0 | Slide 1 · problem and result | Where is the race car, using only its cameras? camloc estimates position and heading ten times a second around Yas Marina, with no GNSS, no LiDAR, and no starting pose. On held-out night laps, where the map and every calibration exclude the scored lap, the position error is zero point five one metres, lateral zero point three, and ninety-five percent of poses are within one metre. These are held-out training laps, not a hidden-test result. |
| 0:30.1 | Slide 2 · why not odometry | Why not just odometry? Wheel speeds and steering alone drift more than a hundred metres, even from the true start. Seven cameras watch the track at night. Three of them, front left, front right and rear, carry the map fixes. |
| 0:48.7 | Slide 3 · pipeline | The pipeline is classical and causal. Odometry predicts, and keyframe optical flow gives the relative heading. ORB features, matched against a geo-referenced map built from the training laps, give absolute fixes by PnP and RANSAC on each camera. Cross-camera consensus and a statistical gate decide what reaches the Kalman filter, which also learns the car's sideslip online. With no start pose, it searches the whole map, takes a provisional lock, and then confirms it, or switches to a consistent alternative. |
| 1:26.3 | Terminal · live held-out lap 4 run + official scorer (s 0.5–11.5) | A live run on held-out lap four. The map and the calibration were rebuilt without that lap, and no start pose is given. |
| 1:37.3 | Terminal · live held-out lap 4 run + official scorer (s 34.6–55.6) | About a thousand frames, at around twenty-five milliseconds each, locked on the first frame. The official scorer reports zero point four six metres, and ninety-six point seven percent of poses within one metre on this lap. |
| 1:58.3 | Animation · lap 4 replay ×4 (GT display only) | Here is that lap replayed at four times speed. The blue estimate stays on the ground-truth ring, which is drawn for display only. The error stays well under a metre, with short spikes above it, and the running error ends at zero point four six metres. |
| 2:25.0 | Slide 4 · results | Across the three held-out laps: zero point six, zero point four six, and zero point four two metres. Each development step was kept only if the held-out error went down, from one point zero six to zero point five one metres. |
| 2:42.3 | Animation · no start pose, lap 6 @ 70 % | No start pose. At seventy percent of lap six, two cameras agree on a look-alike place, thirteen and a half metres off. The lock is only provisional. Within a third of a second, three rejected fixes agree with each other elsewhere, and the filter switches to them. Two accepted fixes then confirm it. |
| 3:03.5 | Video sheet · 16 mid-lap starts | Over sixteen mid-lap starts, every run locks within two seconds, and all sixteen end correct. |
| 3:11.3 | Slide 5 · robustness and limits | Honest limits. Injected wrong fixes barely matter, but without map fixes the error grows: about two point seven metres over thirty seconds, and five over a minute. On the organisers' daylight sample, which has no ground truth, we only checked structure. It locks at zero point six seconds and covers every second, but accepts few fixes, so it relies more on dead reckoning. |
| 3:37.9 | Slide 6 · reproducibility | It all runs with one command, on both topic layouts, on a laptop CPU. The code, the experiment log and the strict validation protocol are in the repository. |

Total: 3:50.1.

## Rebuild
```bash
.venv/bin/python -W ignore submissions/media/figs.py            # figures (needs the strict-fold runs in cache/)
node submissions/media/render_slides.mjs                          # slides.pdf + slide PNGs (DEMO_CHROME=<chrome-headless-shell>)
vhs submissions/media/terminal.tape                               # terminal take (runs submissions/media/heldout_lap4.sh)
.venv/bin/python -W ignore submissions/media/anim.py lap4 && .venv/bin/python -W ignore submissions/media/anim.py nostart
# then the demo-video cut (prep.py -> render.mjs -> build.py) with submissions/media/shots.json and theme.css
```
