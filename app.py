from __future__ import annotations
from typing import Iterable
import gradio as gr
import time
import os

import pytesseract
from PIL import Image


rtl_css = """
body {
    direction: rtl;
    text-align: right;
}
textarea {
    direction: rtl;
    text-align: right;
}
"""

os.environ["WATSONX_APIKEY"] = "<your watsonx api key here>"


parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 200,
    "min_new_tokens": 1,
    "temperature": 0.1,
    # "top_k": 50,
    # "top_p": 1,
}


model_id = "sdaia/allam-1-13b-instruct"
url = "https://eu-de.ml.cloud.ibm.com"
project_id = "<your project_id here>"


from langchain_ibm import WatsonxLLM


watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url=url,
    project_id=project_id,
    params=parameters,
)


from langchain_core.prompts import PromptTemplate

template = """
اعرب الجملة التالية:
{sentence}

"""
prompt = PromptTemplate.from_template(template)

llm_chain = prompt | watsonx_llm


# for using whisper only
# we can use whisper locally since it's open-source
# but it will take a long time for inference, since
# the code is running inside my laptop 

from openai import OpenAI
client = OpenAI()

def transcribe_and_respond(audio):
    with open(audio, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ar",
        )
    return transcription.text


def process(message, history, audio):
    files = message["files"]
    message = message["text"]
    if audio:
        message = transcribe_and_respond(audio)

    elif files:
        file = files[0]
        if any(file.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
            img = Image.open(file)
            message = pytesseract.image_to_string(img, lang="ara")

    partial_message = ""
    for chunk in llm_chain.stream(message):
        partial_message += chunk
        yield partial_message


def check_input(input_text):
    # a function to check if input is correct, return either "ص", "خ"
    template = """
    أنت معلم لغة عربية وتصحح الأخطاء الإملائية والنحوية فقط.
    قل رأيك بكلمة واحدة فقط، لو كانت صحيحة اكتب "ص" أو خاطئة فاكتب "خ".
    الجملة: {sentence}.
    رأيك: 
    """

    prompt = PromptTemplate(template=template)

    llm_chain = prompt | watsonx_llm
    ans = llm_chain.invoke({"sentence": input_text}).strip()
    print(f"({ans}), {input_text}")
    if ans == "ص":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    else:
        corrected_text = get_corrected_text(input_text)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=corrected_text, visible=True),
        )


def get_corrected_text(sentence):
    wrong_template = """
    هذه الجملة خاطئة، صححها.
    الجملة: {sentence}.
    الجملة الصحيحة: 
    """

    wrong_prompt = PromptTemplate(template=wrong_template)
    wrong_chain = wrong_prompt | watsonx_llm
    return wrong_chain.invoke({"sentence": sentence})


correct_me_interface = gr.Interface(
    fn=check_input,
    inputs=gr.Textbox(label="الجملة", lines=5),
    # additional_inputs_accordion=gr.Accordion("الصوت", open=True),
    outputs=[
        gr.Markdown(
            label="Correct",
            visible=False,
            value="<span style='color:green; font-size:20px'>صحيح، أحسنت</span>",
        ),
        gr.Markdown(
            label="Incorrect",
            visible=False,
            value="<span style='color:red; font-size:20px'> غير صحيح :( </span>",
        ),
        gr.Textbox(label="التصحيح", visible=False),
    ],
    description='<div dir="rtl">ضع نصًا بالكتابة أو بالنطق وسنصحح الأخطاء إن وجدناها</div>',
    flagging_mode="manual",
    examples=[
        "جاء محمد راكضٌ",
        "جاء محمد راكضًا",
        "اللغة أساس الهوية",
        "اللغه اساس الهوية.",
    ],
)

grammer_chat = gr.ChatInterface(
    fn=process,
    type="messages",
    textbox=gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="اكتب رسالة او ارفع ملف...",
        show_label=False,
        rtl=True,
    ),
    additional_inputs=gr.Audio(type="filepath"),
    additional_inputs_accordion=gr.Accordion("الصوت", open=True),
    description='تستطيع إدخال الجملة المراد إعرابها عن طريق الصوت أو الصورة أو النص',
    multimodal=True,
    css=rtl_css,
    # chatbot=gr.Chatbot(
    #         elem_id="chatbot", 
    #         bubble_full_width=False, 
    #         type="messages",
    #         rtl=True,
    #         placeholder="ALLAM",
    #         ),
)

demo = gr.TabbedInterface(
    [grammer_chat, correct_me_interface],
    ["الإعراب", "التصحيح"],
    title="سند",
    css=rtl_css,
    theme="earneleh/paris",
)

demo.launch(share=True)
