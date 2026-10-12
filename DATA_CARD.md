# Applicant data and provenance

20 phone videos, 6446 frames; 12 training, 3 validation, 5 final test videos.
The applicant recorded fixed-view tabletop pushing around a brown notebook
(22 x 13 cm). Five test goals were independently placed black markers, confirmed
by the applicant. Raw source SHA256 hashes and calibration metadata are recorded
in outputs/dataset/manifest.json. Prepared coordinates are included so simulation
and training can run without access to the original videos. A tracking overlay
from one training video is included as an example of the collected data.

Raw recordings remain local. Re-running extraction requires all original files
under data/ with their recorded filenames, including the reference video.
Coordinate extraction is scene-specific and approximate: all frames were detected,
but this does not establish coordinate accuracy. The first 15 videos have no
independent target labels; their endpoints define training/validation tasks.

Test policy evaluation uses only the initial object position and independent goal
marker, never its subsequent demonstration trajectory. Full test trajectories are
included for audit only. Notebook coordinates normalize its camera rotation.
The state world model targets are simulated transitions; it learns no measured
real-world masses or friction. See README.md for limitations and results.
