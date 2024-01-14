import librosa
import numpy as np
import streamlit as st
from pydub import AudioSegment
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('F:\\class\\DeepLearning\\Spech Reaction Classification\\audio.h5')

# Define your class names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'ps']

# Define custom messages for each emotion
custom_messages = {
    'angry': "Your voice sounds like it might express anger.",
    'disgust': "Your voice sounds like it might express disgust.",
    'fear': "Your voice sounds like it might express fear.",
    'happy': "Your voice sounds like it might express happiness.",
    'neutral': "Your voice sounds neutral.",
    'sad': "Your voice sounds like it might express sadness.",
    'ps': "Your voice sounds like a pleasant surprise."
}


# Function to extract audio features using librosa
def extract_feature(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Function for preprocessing and making predictions
def preprocess_and_predict(file_path):
    # Preprocess the audio file
    feature = extract_feature(file_path)
    x = np.expand_dims(feature, axis=0)
    x = np.expand_dims(x, axis=-1)

    # Make predictions
    predictions = model.predict(x)
    predicted_label = class_names[np.argmax(predictions)]

    return predicted_label

# Main function for Streamlit app
def main():
    # Streamlit app title
    st.title("Speech Emotion Recognition App")
    st.write(
        "This app predicts the emotion conveyed in speech audio. "
        "You can either upload an MP3 or WAV file audio to analyze."
    )
    st.write(
        "For More Information About This Project Check This Link:  "
        "https://github.com/DreamIsMl/Speech-Emotion-Classification"
    )
    
    # File uploader for audio files (WAV or MP3)
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    # If a file is uploaded
    if uploaded_file is not None:
        # Convert MP3 to WAV format if uploaded file is MP3
        if uploaded_file.type == "audio/mp3":
            audio = AudioSegment.from_mp3(uploaded_file)
            audio.export("temp_audio.wav", format="wav")
        else:
            # Save the uploaded file temporarily as WAV
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getvalue())

        # Display the audio file details
        st.audio("temp_audio.wav", format='audio/wav')
        st.write("File details:")
        st.write(uploaded_file.type)
        st.write(uploaded_file.size)

        # Preprocess and predict emotion
        predicted_label = preprocess_and_predict("temp_audio.wav")

        # Display custom messages based on predicted emotion
        if predicted_label in custom_messages:
            st.success(f"Prediction: {predicted_label}")
            st.write(custom_messages[predicted_label])
        else:
            st.success(f"Prediction: {predicted_label}")

        # Force Streamlit to refresh for displaying developer information
        st.text("")  # Add an empty line to trigger a refresh



# Run the Streamlit app if the script is executed directly
if __name__ == "__main__":
    main()
