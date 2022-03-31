import os
import glob
import time
import streamlit as st
from audiorecoder import AudioRecoder
from service import BiometryService
from config import *


def record_sample(path, seconds):
    with st.spinner(f'Говорите в течении {seconds} секунд ....'):
        st.session_state.audio_recoder.record(path, seconds)


def main():
    model = load_model()

    st.header("Голосовая биометрия")
    st.write(os.getcwd())

    if not st.session_state.start_record_example:
        st.session_state.start_record_example = st.button('Записать образец голоса')

    if st.session_state.start_record_example:

        if st.session_state.count_record_example == 0:
            old_records = glob.glob(f'{os.path.join(DIRECTORY_SAMPELS, TEMPLATE_NAME_EXAMPLES_VOICE.format("*"))}')

            for record in old_records:
                # st.write(record)
                os.remove(record)

        if st.button('Начать запись'):
            path = os.path.join(DIRECTORY_SAMPELS, TEMPLATE_NAME_EXAMPLES_VOICE.format(st.session_state.count_record_example))
            record_sample(path, DURATIONS)
            st.session_state.count_record_example += 1

        if st.button('Очистить'):
            st.session_state.count_record_example = 0

        st.write(f"Записано {st.session_state.count_record_example}/{COUNT_EXAMPLES_VOICE} образцов.")

        if st.session_state.count_record_example == COUNT_EXAMPLES_VOICE:
            st.session_state.start_record_example = False
            st.session_state.count_record_example = 0

    if st.button('Провести идентификацию'):
        if len(glob.glob(f'{os.path.join(DIRECTORY_SAMPELS, TEMPLATE_NAME_EXAMPLES_VOICE.format("*"))}')) != COUNT_EXAMPLES_VOICE:
            st.write(f"Вначале запишите все образцы.")
        else:
            path = os.path.join(DIRECTORY_SAMPELS, TEMPLATE_REC_VOICE)
            model.set_specimen_d_vector()
            record_sample(path, DURATIONS_RECOG)
            if model.predict():
                st.write(f"Пользователь совпадает")
            else:
                st.write(f"Пользователь не совпадает")


@st.cache(allow_output_mutation=True)
def load_model():
    return BiometryService(PATH_MODEL, -1.9)


if __name__ == "__main__":
    if 'audio_recoder' not in st.session_state:
        st.session_state.audio_recoder = AudioRecoder()

    if 'start_record_example' not in st.session_state:
        st.session_state.start_record_example = False

    if 'start_recognition' not in st.session_state:
        st.session_state.start_recognition = False

    if 'count_record_example' not in st.session_state:
        st.session_state.count_record_example = 0

    main()
