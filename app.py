import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import os
import tempfile
import time
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import streamlit as st
from docx import Document
from docx.shared import Pt

# =======================
# Инициализация модели
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
processor = AutoProcessor.from_pretrained(model_id)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

# =======================
# Функции
# =======================
def convert_mp4_to_mp3(video_path, output_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path, codec='mp3')
    return output_path

def process_audio(file_path, language="ru", progress_callback=None):
    if file_path.endswith(".mp4"):
        temp_mp3_path = file_path.replace(".mp4", ".mp3")
        file_path = convert_mp4_to_mp3(file_path, temp_mp3_path)

    audio_data, sr = sf.read(file_path)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Разбиваем аудио на сегменты
    segment_duration_sec = 30  # 30 секунд на сегмент
    total_samples = len(audio_data)
    samples_per_segment = sr * segment_duration_sec
    num_segments = int(np.ceil(total_samples / samples_per_segment))

    full_text = ""

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = min((i + 1) * samples_per_segment, total_samples)
        segment = audio_data[start_sample:end_sample]

        result = asr_pipeline(
            {"array": segment, "sampling_rate": sr},
            generate_kwargs={"language": language}
        )
        full_text += result["text"] + "\n\n"

        if progress_callback:
            progress_callback(i + 1, num_segments)

    return full_text.strip()

def generate_docx(text: str) -> bytes:
    doc = Document()

    # Заголовок
    heading = doc.add_heading("Транскрибированный текст", level=1)
    heading.alignment = 0  # по левому краю

    # Основной текст
    paragraph = doc.add_paragraph(text)
    run = paragraph.runs[0]
    font = run.font
    font.name = "Times New Roman"
    font.size = Pt(14)

    # Сохраняем в память
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        tmp.seek(0)
        docx_bytes = tmp.read()

    return docx_bytes

# =======================
# Streamlit UI
# =======================

st.set_page_config(page_title="Whisper Transcriber", layout="centered")
st.title("🎧 Whisper: Транскрибация аудио/видео")
st.markdown("Загрузите `.mp3`, `.wav` или `.mp4` и получите транскрибацию в формате Word (.docx).")

uploaded_file = st.file_uploader("Загрузите файл", type=["mp3", "wav", "mp4"])

language = st.selectbox("Выберите язык речи", ["ru", "en", "de", "fr", "es"], index=0)

if uploaded_file is not None:
    st.info("Файл успешно загружен. Начинаем обработку...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    progress_bar = st.progress(0, text="Обработка сегментов...")

    def update_progress(current, total):
        progress = int((current / total) * 100)
        progress_bar.progress(progress, text=f"Обработка сегмента {current} из {total}")

    try:
        start_time = time.time()
        with st.spinner("Транскрибация..."):
            transcribed_text = process_audio(temp_path, language, progress_callback=update_progress)

        elapsed = time.time() - start_time
        st.success(f"Обработка завершена за {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

        st.text_area("Предпросмотр текста", transcribed_text, height=300)

        docx_data = generate_docx(transcribed_text)

        st.download_button(
            label="📄 Скачать результат (.docx)",
            data=docx_data,
            file_name="transcription.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except Exception as e:
        st.error(f"Произошла ошибка: {str(e)}")

    finally:
        os.remove(temp_path)
