## Intro

Transformer可以分为3类:

-   GPT-like (also called _auto-regressive_ Transformer models)
-   BERT-like (also called _auto-encoding_ Transformer models)
-   BART/T5-like (also called _sequence-to-sequence_ Transformer models)

两个任务的例子：
causal language modeling
![](Pasted%20image%2020210628223728.png)
_masked language modeling_
![](Pasted%20image%2020210628223747.png)



Encoder和Decoder有时可以单独使用, depending on the task:

-   **Encoder-only models**: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
-   **Decoder-only models**: Good for generative tasks such as text generation.
-   **Encoder-decoder models** or **sequence-to-sequence models**: Good for generative tasks that require an input, such as translation or summarization.

## Encoder


Encoder models use only the encoder of a Transformer model. At each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having “bi-directional” attention, and are often called **auto-encoding models**.

The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence.

**Encoder models are best suited for tasks requiring an understanding of the full sentence, such as sentence classification, named entity recognition (and more generally word classification), and extractive question answering.**

Representatives of this family of models include:

-   [ALBERT](https://huggingface.co/transformers/model_doc/albert.html)
-   [BERT](https://huggingface.co/transformers/model_doc/bert.html)
-   [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)
-   [ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)
-   [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)


![](Pasted%20image%2020210628235445.png)
Encoder的输入是word sequence，输出是一个vector sequence。

vector的维数由model结构决定，比如based-BERT是768维。
每个vector不光由一个word决定，还受到context的影响。(通过self-attention做到的)

使用Encoder的场景：
![](Pasted%20image%2020210629000116.png)
可以用encoder抽取出vectors，然后接到其他网络中做后续处理。

## Decoder
Decoder models use only the decoder of a Transformer model. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. These models are often called **auto-regressive models**.

The pretraining of decoder models usually revolves around predicting the next word in the sentence.

These models are best suited for tasks involving text generation.

Representatives of this family of models include:

-   [CTRL](https://huggingface.co/transformers/model_doc/ctrl.html)
-   [GPT](https://huggingface.co/transformers/model_doc/gpt.html)
-   [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)
-   [Transformer XL](https://huggingface.co/transformers/model_doc/transformerxl.html)

![](Pasted%20image%2020210629001719.png)




## Sequence-to-sequence models


The pretraining of these models can be done using the objectives of encoder or decoder models, but usually involves something a bit more complex. For instance, [T5](https://huggingface.co/t5-base) is pretrained by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is then to predict the text that this mask word replaces.

Sequence-to-sequence models are best suited for tasks revolving around generating new sentences depending on a given input, such as summarization, translation, or generative question answering.

Representatives of this family of models include:

-   [BART](https://huggingface.co/transformers/model_doc/bart.html)
-   [mBART](https://huggingface.co/transformers/model_doc/mbart.html)
-   [Marian](https://huggingface.co/transformers/model_doc/marian.html)
-   [T5](https://huggingface.co/transformers/model_doc/t5.html)


![](Pasted%20image%2020210629002730.png)





## Bias and limitations
When you use these tools, you therefore need to keep in the back of your mind that the original model you are using could very easily generate sexist, racist, or homophobic content. Fine-tuning the model on your data won’t make this intrinsic bias disappear.



