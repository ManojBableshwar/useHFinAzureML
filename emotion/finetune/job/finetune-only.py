
import argparse
import os
parser = argparse.ArgumentParser("train")
parser.add_argument("--base_model_name", type=str, help="model id from huggingface.co")
parser.add_argument("--epochs", type=str, help="epochs count")
parser.add_argument("--model_output", type=str, help="Path of output model")


args = parser.parse_args()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import mlflow
mlflow.pytorch.autolog()

model_name = args.base_model_name

from datasets import load_dataset
emotions = load_dataset("emotion")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_encoded["train"].features

from transformers import AutoModelForSequenceClassification
num_labels = 6
id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label2id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id2label, label2id=label2id).to(device))

import numpy as np
import evaluate

#metric = evaluate.load("accuracy")

#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    return metric.compute(predictions=predictions, references=labels)

log_dir=os.path.join(args.model_output, "metrics")
os.mkdir(log_dir)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(output_dir=args.model_output,
                                  num_train_epochs=int(args.epochs),
                                  learning_rate=2e-5,
                                  evaluation_strategy="steps",
                                  logging_dir = log_dir)

trainer = Trainer(model=model, args=training_args,
#                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"])
train_result = trainer.train()

# save train results
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)

# compute evaluation results
#metrics = trainer.evaluate()
#trainer.log_metrics("eval", metrics)
#trainer.save_metrics("eval", metrics)

model.save_pretrained(args.model_output)
tokenizer.save_pretrained(args.model_output)




