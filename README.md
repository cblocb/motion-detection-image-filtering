# motion-detection-image-filtering

motion detection from a fixed-camera image sequence

this was my first computer vision project

i tried a few different ways of doing temporal differencing / filtering to detect moving pixels, then thresholded the magnitude to get binary motion masks

i also compared what happens if you smooth spatially before doing the temporal derivative

main stuff in here:

- simple central-difference temporal derivative
- derivative-of-gaussian in time
- box / gaussian smoothing in space
- adaptive thresholding from the temporal-gradient magnitudes

the dataset was the redchair sequence

the best-looking result ended up coming more from stronger spatial gaussian smoothing than from the fancier temporal filter

there are a couple sample output images here and the full submitted pdf writeup is `project1_report.pdf`
