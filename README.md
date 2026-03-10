## Kalman Video denoiser

This is the source code of a simple streamlit app that denoise and enhance a video from a camera in real time.
The app is here.
This tool is especially useful for small cameras mounted on FPV drones that have to be denoised in real time and that can be readed from computer or phone.
This tool allowes to:

- connect any camera that can be connected to the browser

- denoise a video using the Adaptive Kalman Filter (optional, suggested for noisy videos)

- enhance each frame of the video using the Clahe HSV algorithm (optional, suggested for dark environments or backlit images with high contrast)

- see the denoised video in full screen (thanks to streamlit functionality)

- record the denoised video (thanks to streamlit functionality)

