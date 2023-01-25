from flask import Flask, render_template, request
from transformers import pipeline
from preprocessor import preprocess_sentence, preprocess_result
from time_check import do_something
import math
import time


app = Flask(__name__)
model_name = "jx7789/kobart_summary_v2"
gen_kwargs = {"length_penalty": 1.2, "num_beams":8, "max_length": 128}


@app.route('/', methods=['GET','POST'])

def home():
    
    do_something()
    text_output = ""
    text_input = str(request.form.get('size'))
    if text_input:
        
        text_input = preprocess_sentence(text_input)
        result = preprocess_result(text_input)
        del result[0]
        print(result)
        start = time.time()
        pipe = pipeline("summarization", model=model_name)
        
        text_output = pipe('[sep]'.join(result), **gen_kwargs)[0]["summary_text"]
        end = time.time()

        print("걸린시간",end - start)
    return render_template('index.html', text_output=text_output)

if __name__ == '__main__':
    app.run(debug=True)