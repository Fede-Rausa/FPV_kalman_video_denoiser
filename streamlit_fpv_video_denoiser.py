import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from kalman_filter_class import kalman_video_denoiser
import numpy as np
from clahe_opencv import clahe_free

class Processor(VideoProcessorBase):
    def __init__(self):
        self.setuptodo = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24") # Shape: (H, W, 3)

        if self.setuptodo:
            # Capture (H, W, C)
            self.shape = img.shape
            # Initialize with the full 3D shape
            self.kf = kalman_video_denoiser(shape=self.shape, R=400)
            self.clahe = clahe_free(mode='hsv')

            self.setuptodo = False

        denoised_img = img
        if use_akf:
            denoised_img = self.kf.update(denoised_img)
        if use_clahe:
            denoised_img = self.clahe.apply(denoised_img)


        return av.VideoFrame.from_ndarray(denoised_img, format="bgr24")



# 2. Configure the Streamer
st.title("Real-Time FPV video denoiser")
st.markdown(''' A fast real time video denoiser useful for webcams and FPV cameras that can be used on simple cpu.''')
use_akf = st.checkbox('use Adaptive Kalman Filter', value=True)
use_clahe = st.checkbox('use Clahe', value=False)


# This allows the user to pick which camera to use in the browser
webrtc_streamer(
    key="example",
    video_processor_factory=Processor,
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }),
    media_stream_constraints={"video": True, "audio": False}
)