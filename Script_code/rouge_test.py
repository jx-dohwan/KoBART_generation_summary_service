from rouge import Rouge
from transformers import AutoTokenizer
rouge = Rouge()
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

def compute_metrics(pred):
    
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    return rouge.get_scores(pred_str, label_str, avg=True) 

""" 이거는 그냥 밖에서 사용할 수 있도록 해도 될듯
def generate_summary(test_samples, model, config):

    inputs = tokenizer(
        test_samples["Text"],
        padding="max_length",
        truncation=True,
        max_length=config.max_len,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)

    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, num_beams=5, no_repeat_ngram_size=3,
                            attention_mask=attention_mask, max_length=128
                            pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str
""" 