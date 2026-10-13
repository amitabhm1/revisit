# Recording your own room

Revisit reads recordings from [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662), a free
app for iPhones and iPads with LiDAR (Pro models). A recording is a folder with `rgb.mp4`, `depth/`,
`confidence/`, `odometry.csv` (the ARKit camera path), `imu.csv` and `camera_matrix.csv`. Copy the folders
from the Files app (On My iPhone > Stray Scanner) to your machine.

## Six rules

1. Use the same device and hold it the same way (sideways, same side up) for every scan.
2. Keep lighting the same: lights on, curtains as they were, screens off.
3. Start and end each scan with a two-second hold on something that will not move.
4. Walk slowly at chest height, then add a lower lap around low furniture. About a minute or two is enough;
   Revisit keeps 28 steady keyframes per scan, spread over the whole recording.
5. Let the second scan see the same places as the first. A place it never sees cannot be checked: objects
   there come back as UNVERIFIED with a suggested spot to look from, or not at all if they produce no proposal.
6. To resolve them, record an extra scan that covers those spots and pass it with `--relook`.

## Running

```bash
python scripts/run_room.py --before scans/monday --after scans/friday              # two scans
python scripts/run_room.py --before scans/monday --after scans/friday --relook scans/corner
python scripts/run_room.py --before scans/monday --after scans/friday --verifier sensenova
```

Two plain videos also work: put both in one folder and pass `--scene-dir <folder>`. Frames are then taken
every 30 frames, and without LiDAR or ARKit there is no metric scale (distances stay in Pi3 units).

## Outputs

Everything is written to `outputs/room/<before>_vs_<after>[_relook_<relook>]/`:

| File | Contents |
|:--|:--|
| `change_report.md` | one row per change: verdict, label, p(change), displacement in metres, the verifier's one-line summary; rejected proposals below |
| `change_report.json` | the same plus every proposal's evidence: 3D centroids and boxes in both scans, Stage-1 scores, visibility, look-from pose |
| `scene_diff.rrd` | open with `rerun scene_diff.rrd`: camera paths, coloured point clouds and boxes per change, arrows for moves, the evidence images |
| `timing.json` | seconds per stage, peak host and GPU memory |

Positions are in the joint Pi3 reconstruction of all scans, scaled to metres with the ARKit trajectory.
