```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```


注意这里可以直接对tokenizer的输出batch设置label：
batch["labels"] = torch.tensor([1, 1])


## Processing the data

### Load dataset

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

可以直接通过index查看数据
![](Pasted%20image%2020210630003945.png)

通过raw_train_dataset.features查看详细信息：
![](Pasted%20image%2020210630004119.png)


tokenizer可以同时接收多组sequence：利用`token_type_ids`
![](Pasted%20image%2020210630011241.png)

Note that if you select a different checkpoint, you won’t necessarily have the `token_type_ids` in your tokenized inputs (for instance, they’re not returned if you use a DistilBERT model). They are only returned when the model will know what to do with them, because it has seen them during its pretraining.


```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

但是这样必须把所有data都加载到memory中
To keep the data as a dataset, we will use the [`Dataset.map`] method. This also allows us some extra flexibility, if we need more preprocessing done than just tokenization. The `map` method works by applying a function on each element of the dataset, so let’s define a function that tokenizes our inputs:

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

```

注意这里没有padding，因为padding最好是在建立batch的时候再建立，这样句子不会padding得太长

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```
这里返回的dataset就是在原本的dataset中多加了几维信息：
![](Pasted%20image%2020210630013715.png)
You can even use multiprocessing when applying your preprocessing function with `Dataset.map` by passing along a `num_proc` argument. We didn’t do this here because the 🤗 Tokenizers library already uses multiple threads to tokenize our samples faster, but if you are not using a fast tokenizer backed by this library, this could speed up your preprocessing.


### Dynamic padding

To do this in practice, we have to define a collate function that will apply the correct amount of padding to the items of the dataset we want to batch together. Fortunately, the 🤗 Transformers library provides us with such a function via `DataCollatorWithPadding`. It takes a tokenizer when you instantiate it (to know which padding token to use, and whether the model expects padding to be on the left or on the right of the inputs) and will do everything you need:
```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```


### Trainer API

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```


The only argument you have to provide is a directory where the trained model will be saved
```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```
注意当我们定义的任务与pretrain不相符的时候会报warning，这时head部分会被换成带有random weight的对应的head，因此要重新训练。





Once we have our model, we can define a `Trainer` by passing it all the objects constructed up to now — the `model`, the `training_args`, the training and validation datasets, our `data_collator`, and our `tokenizer`:
```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

然后trainer.train()就能训练，但此时evaluation是没有定义的。

#### evaluation
首先对validation set进行predict：
```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)



```

The output of the `predict` method is another named tuple with three fields: `predictions`, `label_ids`, and `metrics`. The `metrics` field will just contain the loss on the dataset passed, as well as some time metrics (how long it took to predict, in total and on average). Once we complete our `compute_metrics` function and pass it to the `Trainer`, that field will also contain the metrics returned by `compute_metrics`.

因为prediction还是logot，因此还是要处理一下：
```python
import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)
```


然后用load_metric和compute计算metric
```python
from datasets import load_metric

metric = load_metric("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```





Wrapping everything together, we get our `compute_metrics` function:
```python
def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```


然后我们可以设置成每个epoch计算一次：
```python
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

The `Trainer` will work out of the box on multiple GPUs or TPUs and provides lots of options, like mixed-precision training (use `fp16 = True` in your training arguments). We will go over everything it supports in Chapter 10.


















