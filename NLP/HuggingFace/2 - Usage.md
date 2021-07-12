## Introduction

At their core, all models are simple PyTorch `nn.Module` or TensorFlow `tf.keras.Model` classes and can be handled like any other models in their respective machine learning (ML) frameworks.


## Behind the pipeline
![](Pasted%20image%2020210629143959.png)


### tokenizer
作用：
* 分词，得到token
* 把token映射到integer
* Adding additional inputs that may be useful to the model

对token的处理要和pretrain model一致，因此要查pretrain的信息。
 [Model Hub](https://huggingface.co/models).


To do this, we use the `AutoTokenizer` class and its `from_pretrained` method.

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

代码第一次运行的时候，利用checkpoint的名字下载model

经过tokenizer，把文本转成id

transformer model接收的是tensor，要指定backend是keras还是pytorch，要在tokenizer中设定`return_tensors`.
```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```


Here’s what the results look like as PyTorch tensors:
```python
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```
The output itself is a dictionary containing two keys, `input_ids` and `attention_mask`.


### Going through the model


同样，通过`from_pretrained`来下载model
```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

Transformer的输出通常有3维：Batch size, Sequence length, Hidden size.



```python

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```
Note that the outputs of 🤗 Transformers models behave like `namedtuple`s or dictionaries. You can access the elements by attributes (like we did) or by key (`outputs["last_hidden_state"]`), or even by index if you know exactly where the thing you are looking for is (`outputs[0]`).


### Model heads: Making sense out of numbers

model head是对前面的model输出的hidden state作处理的部分

![](Pasted%20image%2020210629151822.png)



There are many different architectures available in 🤗 Transformers, with each one designed around tackling a specific task. Here is a non-exhaustive list:

-   `*Model` (retrieve the hidden states)
-   `*ForCausalLM`
-   `*ForMaskedLM`
-   `*ForMultipleChoice`
-   `*ForQuestionAnswering`
-   `*ForSequenceClassification`
-   `*ForTokenClassification`
-   and others 🤗


假设我们做文本分类，那么可以直接用`AutoModelForSequenceClassification`而不是`AutoModel` class。


```python

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)


```


这时原本的N*batch_size*768的hidden state就经过head layer变成了N*2的tensor (因为是classification的one hot所以是2)



### Postprocessing the output


(**all 🤗 Transformers models output the logits**, as the loss function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function, such as cross entropy)

因此使用时要把logits经过softmax

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

```

To get the labels corresponding to each position, we can inspect the `id2label` attribute of the model config (more on this in the next section):
```python
model.config.id2label

{0: 'NEGATIVE', 1: 'POSITIVE'}
```


## Models
pretrained model加载的过程：
![](Pasted%20image%2020210629153157.png)



可以用下面的方法调参，然后train from scratch：
![](Pasted%20image%2020210629153638.png)

save和load的方法：
![](Pasted%20image%2020210629153709.png)



之前用的AutoModel是利用传入的checkpoint判断用哪个model，然后自动加载config
其实也可以手动指定：
```python
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

```
注意此时model的weight是随机生成的，没有加载pretrained参数

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")

```

最好是用AutoModel直接传checkpoint name

下载下来的model的路径：
The weights have been downloaded and cached (so future calls to the `from_pretrained` method won’t re-download them) in the cache folder, which defaults to _~/.cache/huggingface/transformers_. You can customize your cache folder by setting the `HF_HOME` environment variable.

保存model：
model.save_pretrained("directory_on_my_computer")

### Inference

```python
sequences = [
  "Hello!",
  "Cool.",
  "Nice!"
]

encoded_sequences = [
  [ 101, 7592,  999,  102],
  [ 101, 4658, 1012,  102],
  [ 101, 3835,  999,  102]
]

import torch

model_inputs = torch.tensor(encoded_sequences)

output = model(model_inputs)





```




## Tokenizers

### Word-based
we need a custom token to represent words that are not in our vocabulary. This is known as the “unknown” token, often represented as ”[UNK]” or ””. It’s generally a bad sign if you see that the tokenizer is producing a lot of these tokens

One way to reduce the amount of unknown tokens is to go one level deeper, using a character-based tokenizer.


### Character-based

* 在有些语言中，character意义很小
* 文本变长

### Subword tokenization

**frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.**

### And more!

Unsurprisingly, there are many more techniques out there. To name a few:

-   Byte-level BPE, as used in GPT-2
-   WordPiece, as used in BERT
-   SentencePiece or Unigram, as used in several multilingual models

### Loading and saving
`from_pretrained` and `save_pretrained`

Similar to `AutoModel`, the `AutoTokenizer` class will grab the proper tokenizer class in the library based on the checkpoint name, and can be used directly with any checkpoint:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenizer("Using a Transformer network is simple")

{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

```

### Encoding
分词:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)




```
转id:
```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

```



### Decoding

```python
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)

```

Note that the `decode` method not only converts the indices back to tokens, but also groups together the tokens that were part of the same words to produce a readable sentence.



## Handling multiple sequences

model默认接收batch作为输入。

The padding token ID can be found in `tokenizer.pad_token_id`.



注意:单独用tokenizer.pad_token_id，会对结果有影响，因此要结合mask
```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [[200, 200, 200], [200, 200, tokenizer.pad_token_id]]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)


tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
tensor([[ 1.5694, -1.3895],
        [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)



```


```python

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id]
]

attention_mask = [
  [1, 1, 1],
  [1, 1, 0]
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)

tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)


```

With Transformer models, there is a limit to the lengths of the sequences we can pass the models. Most models handle sequences of up to 512 or 1024 tokens

Models have different supported sequence lengths, and some specialize in handling very long sequences. [Longformer](https://huggingface.co/transformers/model_doc/longformer.html) is one example, and another is [LED](https://huggingface.co/transformers/model_doc/led.html).

Otherwise, we recommend you truncate your sequences by specifying the `max_sequence_length` parameter:

sequence = sequence[:max_sequence_length]



## Putting it all together


可以把sentence的batch直接传入tokenizer，这个method很强大。
可以处理单个句子，也可以自动判断，处理多个句子。
```python
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

model_inputs = tokenizer(sequences)

```


可以padding：
```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

可以truncate：
```python
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)

```
 规定返回的tensor类型：
 ```python
 sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```


自动的tokenizer要比手动的多加BEGIN和END

总结：
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)




```







