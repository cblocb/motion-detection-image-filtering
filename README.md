# motion-detection-image-filtering

this was my first project for eece 5639 (computer vision) at northeastern

the goal was to detect motion in a fixed-camera image sequence by looking at temporal grayscale changes frame to frame

i compared:

- simple central-difference temporal derivatives
- derivative-of-gaussian temporal filters at different temporal scales
- spatial smoothing before differencing using box filters and gaussian blur
- an adaptive threshold based on the per-frame std of the temporal-gradient magnitudes

the dataset used here was the redchair sequence

the repo includes:

- `project1.py` for the main experiments
- `make_comparison_grids.py` for stitching method outputs into comparison grids
- `project1_report.pdf` for the submitted writeup
- a couple sample output images from frame 63 / the report

main takeaway:

spatial smoothing mattered more than fancy temporal filtering on this dataset. gaussian smoothing gave the cleanest motion masks, and the best-looking results in the report came from stronger spatial blur.

if you want the exact project framing/results, the pdf has the full writeup.
