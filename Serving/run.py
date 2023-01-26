from flask import Flask, render_template, request
from transformers import pipeline
from preprocessor import preprocess_sentence, preprocess_result, remove_empty_pattern
#from time_check import do_something
import math
import time


app = Flask(__name__)
model_name = "jx7789/kobart_summary_v3"
gen_kwargs = {"length_penalty": 1.2, "num_beams":8, "max_length": 128}


@app.route('/', methods=['GET','POST'])

def home():
    #do_something()
    text_input = False
    text_input = str(request.form.get('size'))
    if text_input: 
        result = preprocess_sentence(text_input)
        result = remove_empty_pattern(result)
        result = preprocess_result(result) #내 생각과는 다르게 사용을 하지 않아도 [이름][시간]다 제거되었음, 
        #start = time.time()
        #print(result)
        pipe = pipeline("summarization", model=model_name)
        text_output = pipe('[sep]'.join(result), **gen_kwargs)[0]["summary_text"]
        #end = time.time()
        #print("걸린시간",end - start)
        #print("test하기",text_output)
    return render_template('index.html', text_output=text_output)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(3000), debug=True)