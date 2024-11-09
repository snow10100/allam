# Allam Challenge 
You only need an IBM WatsonX API KEY and a project ID to run the code.

## Usage 
There are two tabs, one for arabic language grammer, the other for syntax checking. 

The tabs are intuitive and simple to understand, you have a chat interface where you can upload
images, audio, and chat using text. Allam will respond by grammatizing your input. 
Arabic text will be extracted from images using tesseract, and audio will be transcribed using whisper. 

The syntax checking and correction tab, is quite simple aswell. You put text and submit.  
Allam will check and fix errors for you. 

## Installation 
Add your api key and project_id into the code. 

Then, 
```python
pip install requirements.txt
```

then run the code by 
```python
python app.py
```
