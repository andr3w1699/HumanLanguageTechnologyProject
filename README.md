# HumanLanguageTechnologyProject
This is the repo for the Human Language Technology course final project, group 4

# Dataset (draft)
To download the dataset go to this link: 
https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ and look for file:  amazon_review_full_csv.tar.gz

# Binary vs. Multi-Class Sentiment Classification (draft)
Binary: 1-2 stars = negative, 4-5 stars = positive (3 = neutral or drop) 
Multi-class: Use all 5 ratings (1 to 5) which introduces finer granularity
We 'll need models that can handle nuanced language cues for multi-class tasks

# Analyzing product reviews using transformers(draft)
This problem requires us to train a sentiment classification model using transformer-based encoders like BERT (Bidirectional Encoder Representations from Transformers), introduced [here](https://arxiv.org/abs/1810.04805). Specifically, we will parse product reviews and classify their ratings (according to whether they are 1, 2, 3, 4 or 5.)

We can consider the [Huggingface transformers library](https://github.com/huggingface/transformers) to load a pre-trained BERT model to compute text embeddings, and append this with an fully-connected neural network model or any other classification model to perform sentiment classification.

One possible choice is the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model from Hugging Face, which can be loaded through the `transformers` library.

Other possibilities to consider among the transformer models are: 
- [roberta-base](https://huggingface.co/FacebookAI/roberta-base): more robust variant of BERT
- [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased): lighter & faster, 97% of BERT’s accuracy
- [albert-base-v2](https://huggingface.co/albert/albert-base-v2): smaller and more efficient
- [deberta-base](https://huggingface.co/microsoft/deberta-base): 	Better handling of word order and syntax
- [ELECTRA-small](https://huggingface.co/google/electra-small-discriminator): Efficient trained with replaced token detection

# Model(draft)
In this part, I briefly describe the model architecture to address the sentiment classification task. There are several popular pre-trained language model encoders to solve this problem, such as BERTbase, DistilBERT, and Roberta. Encoders are a great advantage as they allow us to compress the text into a latent space vector, valid for several tasks.

I developed a general architecture of the model below:
![model architecture](Img/ModelArchitecture.png)

# Text Processing 

I uploaded a notebook where i've done some text processing on the dataset: Lowercasing, Removing Punctuation, Removing stopwords, Removing special characters. I also used SpaCy for try some lemmatization.  
