import streamlit as st
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import parselmouth
from parselmouth.praat import call
import librosa
import os

# Define constants
SAMPLE_RATE = 44100
DURATION = 5
AUDIO_FILENAME = "temp_user_audio.wav"

def analyze_speech(audio_file):
    try:
        snd = parselmouth.Sound(audio_file)
        
        # 1. Pitch (Fundamental Frequency)
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        
        # 2. Jitter (Frequency variation)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        
        # 3. Shimmer (Amplitude variation)
        shimmer = call(point_process, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        
        # 4. Harmonics-to-Noise Ratio (HNR)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # 5. DDK Rate (pa-ta-ka)
        y, sr = librosa.load(audio_file)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        ddk_rate = (len(onsets) -1) / (onsets[-1] - onsets[0]) if len(onsets) > 1 else 0
        
        return {
            "Mean Pitch (F0)": f"{mean_pitch:.2f} Hz",
            "Jitter (Local)": f"{jitter:.3f} %",
            "Shimmer (Local)": f"{shimmer:.3f} %",
            "Harmonics-to-Noise (HNR)": f"{hnr:.2f} dB",
            "DDK Rate": f"{ddk_rate:.2f} syllables/sec"
        }
        
    except Exception as e:
        if "Pitch" in str(e):
             return {"Error": "Could not detect a clear pitch. Please record again, speaking clearly and avoiding background noise."}
        return {"Error": f"An error occurred: {e}"}

def run():
    st.header("Neurological Dysarthria Detector")
    
    with st.expander("ℹ️ Instructions & Explanation", expanded=True):
        st.markdown(f"""
        This tool analyzes your speech for signs of dysarthria (slurred, slow, or difficult speech) by measuring voice stability.
        
        **How to Use:**
        1.  Find a quiet place.
        2.  Click the **"Start {DURATION}-Second Recording"** button.
        3.  When prompted, **repeatedly say "pa-ta-ka"** as quickly and clearly as you can for the full {DURATION} seconds.
        4.  The recording will stop automatically.
        5.  Click the **"Analyze Speech"** button.
        
        **Metrics Explained:**
        * **Jitter:** Measures the variation in voice pitch. Higher jitter can sound "rough" or "hoarse".
        * **Shimmer:** Measures the variation in voice loudness. Higher shimmer is also a sign of instability.
        * **HNR:** The ratio of "tonal" sound to "noise". A lower HNR can indicate a breathy or harsh voice.
        * **DDK Rate:** The speed at which you can repeat syllables, a key test of motor-speech coordination.
        """)

    if st.button(f"Start {DURATION}-Second Recording"):
        try:
            st.info("Recording... Speak 'pa-ta-ka' repeatedly now!")
            
            # Record audio
            my_recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()  # Wait for recording to finish
            
            # Save as WAV file
            wavfile.write(AUDIO_FILENAME, SAMPLE_RATE, my_recording)
            
            st.success("Recording complete!")
            st.audio(AUDIO_FILENAME)
            st.session_state.audio_recorded = True
            
        except Exception as e:
            st.error(f"Error during recording: {e}. Please ensure your microphone is enabled in the browser.")
            st.session_state.audio_recorded = False

    if st.button("Analyze Speech"):
        if 'audio_recorded' in st.session_state and st.session_state.audio_recorded:
            with st.spinner("Analyzing audio..."):
                results = analyze_speech(AUDIO_FILENAME)
                
                st.subheader("Acoustic Analysis Results")
                if "Error" in results:
                    st.error(results["Error"])
                else:
                    for key, value in results.items():
                        st.metric(label=key, value=value)
            # Clean up
            if os.path.exists(AUDIO_FILENAME):
                os.remove(AUDIO_FILENAME)
            st.session_state.audio_recorded = False
        else:
            st.warning("You must record your audio first!")