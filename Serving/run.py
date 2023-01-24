from flask import Flask, render_template, request
from transformers import pipeline
from preprocessor import preprocess_sentence
from time_check import do_something
import math
import time


app = Flask(__name__)
model_name = "jx7789/kobart_summary_v2"
gen_kwargs = {"length_penalty": 1.2, "num_beams":8, "max_length": 128}


@app.route('/', methods=['GET','POST'])

def home():
    
    do_something()
    result = []
    text_output = ""
    text_input = str(request.form.get('size'))
    if text_input:
        text_input = preprocess_sentence(text_input)
        for i in text_input:
            result.append(preprocess_sentence(i, v2=True).lower())
        
        pipe = pipeline("summarization", model=model_name)
        start = time.time()
        text_output = pipe('[sep]'.join(result), **gen_kwargs)[0]["summary_text"]
        end = time.time()

        print(end - start)
    return render_template('index.html', text_output=text_output)

if __name__ == '__main__':
    app.run(debug=True)