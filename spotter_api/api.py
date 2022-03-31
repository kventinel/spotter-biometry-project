import os
import glob
import time
import streamlit as st
from audiorecoder import AudioRecoder
from config import *

from service import Keyword_Spotting_Service

def record_sample(path, seconds):
    with st.spinner(f'Говорите в течении {seconds} секунд ....'):
        st.session_state.audio_recoder.record(path, seconds)


def main():
    kss = Keyword_Spotting_Service()
    path = os.path.join(DIRECTORY_SAMPELS, TEMPLATE_NAME_RECORD)
    
    st.header("Споттер")
    st.write(os.getcwd())

    if st.button('Записать голосовое сообщение'):
        # Удаляем старое сообщение
        try:
            os.remove(path)
        except OSError:
            pass
        # пишем новое сообщение
        record_sample(path, DURATIONS)
        st.session_state.speach_recorded = True
        st.write(f"Сообщение записано.")

    if st.button('Запустить споттер'):
        if not st.session_state.speach_recorded:
            st.write(f"Вначале запишите сообщение.")
        else: 
            predict = kss.predict(path)
            st.write(f"Spotted '{predict}'.")


if __name__ == "__main__":
    if 'audio_recoder' not in st.session_state:
        st.session_state.audio_recoder = AudioRecoder()
        
    if 'speach_recorded' not in st.session_state:
        st.session_state.speach_recorded = False

    main()