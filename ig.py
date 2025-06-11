# BERT model with Integrated Gradients (IG)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import captum
import numpy as np

#bert model finetuned on mnli
#swap Adam optimizer with M-FAC
model_name = "M-FAC/bert-tiny-finetuned-mnli" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# 0:entailment, 1:neutral, 2:contradiction
target_label = int(1)

# predict_and_gradients
def calculate_outputs_and_gradients(inputs_embeds, attention_mask, model, target_label_idx):
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    if target_label_idx is None:
        target_label_idx = torch.argmax(probs, 1).item()
    
    # clear grad
    score = probs[0, target_label_idx]
    model.zero_grad()
    score.backward()
    gradients = inputs_embeds.grad.detach().cpu().numpy()[0]

    return gradients
    
#take an example
premise = "Your gift is appreciated by each and every student who will benefit from your generosity."
hypothesis = "Hundreds of students will benefit from your generosity."
encoded = tokenizer(premise,hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding="max_length")

input_ids = encoded["input_ids"]
input_ids = input_ids.long()
attention_mask = encoded["attention_mask"]


# IntegratedGradients
def integrated_gradients(inputs, attention_mask, model, target_label_idx, predict_and_gradients, steps=50):
    embed_layer   = model.get_input_embeddings()
    orig_embeds   = embed_layer(inputs)
    inputs_embeds = orig_embeds.detach().requires_grad_(True)
    baseline  = torch.zeros_like(orig_embeds)
    diff = orig_embeds - baseline

    grads = predict_and_gradients(inputs_embeds, attention_mask, model, target_label_idx)

    total_grads = torch.zeros_like(diff)
    for i in np.linspace(0, 1, steps):
        embeds = (baseline + i * diff).detach().requires_grad_(True)
        grads = predict_and_gradients(inputs_embeds, attention_mask, model, target_label_idx)
        total_grads += grads

    avg_grads = total_grads / steps
    integrated_grad = diff * avg_grads
    integrated_grad = integrated_grad.detach().squeeze(0).cpu().numpy()
    return integrated_grad

# attribution
attributions = integrated_gradients(input_ids, attention_mask, model, target_label, 
                                    calculate_outputs_and_gradients, steps=50)

# token-level IG value
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
scores = np.linalg.norm(attributions, axis=1)
for tok, sc in zip(tokens, scores):
    print(f"{tok:>12} : {sc.item(): .4f}")