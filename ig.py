# BERT model with Integrated Gradients (IG)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import captum

#bert model finetuned on mnli
#swap Adam optimizer with M-FAC
model_name = "M-FAC/bert-tiny-finetuned-mnli" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# 0:entailment, 1:neutral, 2:contradiction
target_label = int(1)

# forward
def forward_func(input_ids, attention_mask):
    input_ids = input_ids.long()
    logits = model(input_ids=input_ids, attention_mask=attention_mask,return_dict=True).logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs
    
#take an example
premise = "Your gift is appreciated by each and every student who will benefit from your generosity."
hypothesis = "Hundreds of students will benefit from your generosity."
encoded = tokenizer(premise,hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding="max_length")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

emb_out = model.bert.embeddings.word_embeddings(input_ids).detach()
emb_out.requires_grad_(True)  
baseline_ids  = torch.zeros_like(emb_out).long()

# IntegratedGradients
ig = captum.attr.IntegratedGradients(forward_func)

# attribution
attributions, delta = ig.attribute(
    inputs=emb_out, 
    additional_forward_args=attention_mask,
    baselines=baseline_ids,
    target=target_label,
    return_convergence_delta=True,
    n_steps=20)

word_attributions = attributions[0].squeeze(0)

# token-level IG value
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
attr_scores = word_attributions.detach().cpu().numpy().tolist()

print("Token-level attributions for label", target_label_idx, "probability:")
for token, score in zip(tokens, attr_scores):
    if token not in ("[PAD]", tokenizer.pad_token):
        print(f"{token:>10} : {score:.4f}")
