# 🧠 Fine-Tuning T5 Transformer for Text Summarization

!\[Fine-Tuned T5 Banner](Images/ChatGPT Image Nov 2, 2025, 10\_02\_31 PM.png)



> \*\*Project Type:\*\* NLP | Deep Learning | Sequence-to-Sequence  

> \*\*Model:\*\* T5-small (Text-to-Text Transfer Transformer)  

> \*\*Framework:\*\* Hugging Face Transformers  

> \*\*Objective:\*\* Abstractive text summarization using fine-tuned Transformer



---



## 🏁 Overview



This project demonstrates how to \*\*fine-tune the pre-trained T5 model\*\* (`t5-small`) for \*\*abstractive text summarization\*\* using the Hugging Face `transformers` and `datasets` libraries.  

It walks through every stage — from preprocessing to evaluation and deployment — for an end-to-end summarization system.



Unlike extractive summarization, where sentences are selected from the original text, \*\*abstractive summarization\*\* \*generates new sentences\* that capture the essence of the text.



---



## ⚙️ Core Idea: Text-to-Text Framework



The \*\*T5 (Text-to-Text Transfer Transformer)\*\* model, proposed by Google Research, treats every NLP task as a \*\*text-to-text\*\* problem:

| Task | Input Format | Output Format |

|------|---------------|----------------|

| Translation | `"translate English to German: Hello"` | `"Hallo"` |

| Summarization | `"summarize: The article says ..."` | `"The article discusses ..."` |

| QA | `"question: Who invented AI? context: ..."` | `"John McCarthy"` |

This unified approach allows a single model architecture to handle diverse NLP tasks.



---



## 🧩 Project Features

✅ Fine-tunes `t5-small` on a summarization dataset  

✅ Implements preprocessing, tokenization, and data batching  

✅ Includes \*\*ROUGE evaluation\*\* for summary quality  

✅ Handles \*\*training resumption, checkpoints, and saving\*\*  

✅ Compatible with \*\*Google Colab + Drive\*\* for cloud training  

✅ Clean modular code with \*\*configurable parameters\*\* and \*\*logging\*\*

---

## 📁 Directory Structure

> \*\*Note:\*\* Keep your screenshots inside the `Images/` folder.
...
Fine-Tuning--T5-Transformer-for-Text-Summarization/ │ ├── Images/ │ ├── ChatGPT Image Nov 2, 2025, 10\_02\_31 PM.png │ ├── app\_demo\_1.png │ ├── app\_demo\_2.png │ ├── app\_demo\_3.png │ ├── training\_log\_1.png │ ├── training\_log\_2.png │ ├── training\_log\_3.png │ └── transformer\_architecture.png │ ├── training\_script.ipynb ├── eval\_results.json (optional) ├── .gitignore └── README.md
...

## 🧠 Model Architecture

\*\*T5 (Text-to-Text Transfer Transformer)\*\* is based on the Transformer encoder-decoder structure.

\- \*\*Encoder:\*\* Converts the input text into contextual embeddings  

\- \*\*Decoder:\*\* Generates the target summary token by token  

\- \*\*Objective:\*\* Minimize the cross-entropy loss between predicted and target summaries  

(Optional diagram)

!\[Transformer Architecture](Images/transformer\_architecture.png)

---

## 🧮 Dataset Preparation

Any dataset containing pairs of \*text → summary\* can be used.  

The input data should contain at least two columns:

\- `article` (or `text`) — source text to summarize  

\- `highlights` (or `summary`) — reference summary  



Example (from CNN/DailyMail or custom dataset):


```python

{

&nbsp; "article": "The T5 model was introduced by Google Research...",

&nbsp; "summary": "T5 is a transformer model treating all NLP tasks as text-to-text."

}

---

## 🧰 Installation \& Setup

Run the following in your Colab or local environment:

pip install transformers datasets evaluate sentencepiece accelerate

Mount Google Drive if using Colab:

from google.colab import drive

drive.mount('/content/drive')

---

## Configuration (config.py)

class Config:

&nbsp;   MODEL\_NAME = "t5-small"

&nbsp;   MAX\_INPUT\_LENGTH = 512

&nbsp;   MAX\_TARGET\_LENGTH = 150

&nbsp;   TRAIN\_BATCH\_SIZE = 8

&nbsp;   EVAL\_BATCH\_SIZE = 8

&nbsp;   LEARNING\_RATE = 3e-4

&nbsp;   NUM\_EPOCHS = 3

&nbsp;   OUTPUT\_DIR = "/content/t5\_summarizer"

&nbsp;   GRADIENT\_ACCUMULATION\_STEPS = 4

---

## 🧹 Data Preprocessing \& Tokenization

Each article-summary pair is preprocessed and tokenized:

def preprocess\_function(examples):

&nbsp;   inputs = \["summarize: " + doc for doc in examples\["article"]]

&nbsp;   model\_inputs = tokenizer(inputs, max\_length=512, truncation=True)

&nbsp;   labels = tokenizer(text\_target=examples\["summary"], max\_length=150, truncation=True)

&nbsp;   model\_inputs\["labels"] = labels\["input\_ids"]

&nbsp;   return model\_inputs


Then processed using:

tokenized\_train = dataset\["train"].map(preprocess\_function, batched=True)

tokenized\_val = dataset\["validation"].map(preprocess\_function, batched=True)

---

🏋️‍♀️ Training Setup

training\_args = TrainingArguments(

&nbsp;   output\_dir=config.OUTPUT\_DIR,

&nbsp;   evaluation\_strategy="steps",

&nbsp;   save\_strategy="steps",

&nbsp;   per\_device\_train\_batch\_size=config.TRAIN\_BATCH\_SIZE,

&nbsp;   per\_device\_eval\_batch\_size=config.EVAL\_BATCH\_SIZE,

&nbsp;   learning\_rate=config.LEARNING\_RATE,

&nbsp;   num\_train\_epochs=config.NUM\_EPOCHS,

&nbsp;   predict\_with\_generate=True,

&nbsp;   logging\_steps=500,

&nbsp;   save\_steps=500,

&nbsp;   eval\_steps=500,

&nbsp;   gradient\_accumulation\_steps=config.GRADIENT\_ACCUMULATION\_STEPS,

&nbsp;   report\_to="none"

)

---

## 📏 Evaluation Metrics (ROUGE)

We use ROUGE (Recall-Oriented Understudy for Gisting Evaluation) — the standard summarization metric. It measures the overlap between model-generated and reference summaries.

rouge = evaluate.load("rouge")

def compute\_metrics(eval\_pred):

&nbsp;   predictions, labels = eval\_pred

&nbsp;   decoded\_preds = tokenizer.batch\_decode(predictions, skip\_special\_tokens=True)

&nbsp;   labels = np.where(labels != -100, labels, tokenizer.pad\_token\_id)

&nbsp;   decoded\_labels = tokenizer.batch\_decode(labels, skip\_special\_tokens=True)



&nbsp;   result = rouge.compute(predictions=decoded\_preds, references=decoded\_labels, use\_stemmer=True)

&nbsp;   return {k: v \* 100 for k, v in result.items()}

---


## 🚀 Training the Model

trainer = Trainer(

&nbsp;   model=model,

&nbsp;   args=training\_args,

&nbsp;   train\_dataset=tokenized\_train,

&nbsp;   eval\_dataset=tokenized\_val,

&nbsp;   tokenizer=tokenizer,

&nbsp;   compute\_metrics=compute\_metrics,

)

train\_result = trainer.train()

trainer.save\_model(config.OUTPUT\_DIR)

tokenizer.save\_pretrained(config.OUTPUT\_DIR)

---

### Sample Output During Training

> (You can replace this table with your own results)

| Step | Training Loss | Validation Loss | ROUGE-1 | ROUGE-2 | ROUGE-L |

| ---- | ------------- | --------------- | ------- | ------- | ------- |

| 500  | 1.744         | 1.821           | 42.85   | 20.58   | 30.30   |

| 1000 | 1.964         | 1.779           | 42.49   | 20.26   | 29.86   |

| 1500 | 1.866         | 1.780           | 42.40   | 20.12   | 29.97   |

---

## 💾 Saving \& Uploading to Google Drive

After training:

```python

import shutil, os

source = "/content/t5\_summarizer"

destination = "/content/drive/MyDrive/t5\_summarizer\_model"

os.makedirs(destination, exist\_ok=True)

shutil.copytree(source, destination, dirs\_exist\_ok=True)


---

## 🔍 Model Evaluation \& Results



\## Interpretation

\* \*\*Training Loss\*\* → model fits data well  

\* \*\*Validation Loss\*\* → generalizes properly  

\* \*\*ROUGE-1\*\* → strong unigram overlap  

\* \*\*ROUGE-2\*\* → moderate bigram coherence  

\* \*\*ROUGE-L\*\* → consistent structural summarization  

---

\*\*Output:\*\*


> “T5 unifies NLP tasks into a single text-to-text model framework.”


---



## 🖥️ App Demo Screenshots (Input → Generated Summary)



Put your UI screenshots here (from your app demo).


!\[App Demo 1](Images/1.png)

!\[App Demo 2](Images/2.png)

!\[App Demo 3](Images/3.png)

(Your UI screenshots in your context)  

\- App Demo screenshot 1: https://www.genspark.ai/api/files/s/9FBQ3Ctc  

\- App Demo screenshot 2: https://www.genspark.ai/api/files/s/Z7QE3cSg  

\- App Demo screenshot 3: https://www.genspark.ai/api/files/s/OeMyxA75  

---

## 🖼️ Output Images (Training Logs / Metrics)


Separate section for your training output screenshots.


!\[Training Output a](Images/a.png)

!\[Training Output b](Images/b.png)

!\[Training Output c](Images/c.png)

---


## 🧰 Troubleshooting



| Issue                                  | Cause                             | Fix                                                                |

| -------------------------------------- | --------------------------------- | ------------------------------------------------------------------ |

| `IndexError: piece id is out of range` | Invalid token IDs during decoding | Clip token IDs to vocab size or ensure correct tokenizer alignment |

| `CUDA out of memory`                   | GPU memory overflow               | Reduce batch size or sequence length                               |

| `Missing weights in checkpoint`        | Tokenizer mismatch                | Reload tokenizer and model from the same checkpoint                |

| `Low ROUGE scores`                     | Model undertrained                | Increase epochs or try T5-base                                     |

---

## 📈 Future Improvements



\* 🔹 Use \*\*T5-base\*\* or \*\*Flan-T5\*\* for better summarization quality  

\* 🔹 Implement \*\*mixed precision training (fp16)\*\* for speed  

\* 🔹 Fine-tune on \*\*domain-specific datasets\*\* (medical, news, legal)  

\* 🔹 Deploy via \*\*Streamlit\*\* or \*\*Gradio demo\*\*  

---

## 📚 References

\* \[T5 Paper: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)  

\* \[Hugging Face Transformers](https://huggingface.co/transformers/)  

\* \[Evaluate: ROUGE Metric](https://huggingface.co/spaces/evaluate-metric/rouge)  

---
