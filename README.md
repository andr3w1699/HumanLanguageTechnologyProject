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

# Classical + Deep Representation Models 
## Word Embeddings (Static Representation) 
Word2Vec / GloVe are methods to map words into fixed-length dense vectors. These vectors are static: the same word always has the same vector, no matter the context but they capture semantic similarity. We can use libraries like spacy, nltk or torchtext. On the top of it, we can use RNNs, models that process sequences word by word mantaining a hidden state. Among them we can use:
- LSTM (Long Short Term Memory): that are ables to handle long-term dependencies
- BiLSTM (Bidirectional LSTM): processes the sequence forward and backward this gives more context, looks at both left and right of a word
- GRU (Gated Recurrent Unit): a simpler, faster alternative to LSTM

The workflow followed by this approach should be: 
We start from the raw sentence (Text Data) and we have also its label. Then we perform a text preprocessing step (lowercase, remove punctuation if needed, tokenize into words). You can use libraries like `nltk`, `spaCy`, or `torchtext` for this step. Then we have to convert Words to Vectors (Embeddings): for such task we can use use a pretrained embedding matrix like GloVe or Word2Vec. Each word gets mapped to a fixed vector, e.g. 100 or 300 dimensions and words not in the embedding are marked using <UNK> (unknown token). Then for each sequence of text in the dataset we can create a 2D Matrix: (sequence length) × (embedding dim). Then you have to feed into a RNN model (LSTM, BiLSTM or GRU), this reads each word vector sequentially, updating its hidden state. Instead of using just the final hidden state, you can apply attention over all hidden state at each time step or pooling. Last a classification layer: take the output (from attention or final LSTM state), and feed it into a fully connected layer. You can train the whole model using cross-entropy loss on your labels (Labels = sentiment classes e.g., 1–5 stars or positive/negative, Optimizer = Adam or SGD, Loss = CrossEntropyLoss). To implement we can use PyTorch. 

# Parse-Tree-Based Models (Syntactic-Aware Models)
These models leverage syntax trees (like constituency or dependency trees) to understand sentence structure and capture things like negation, sarcasm, or subtle sentiment shifts more accurately than flat models.
TreeLSTM is a variant of LSTM that follows the structure of a syntactic parse tree instead of a flat sequence. It’s designed to process hierarchical input, where each node (e.g., noun phrase, verb phrase) aggregates the information from its children.

Key Paper:
"Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
📎 Tai, Socher, Manning (Stanford) – 2015
📄 https://arxiv.org/abs/1503.00075

libraries and tools: 
- Parse Trees (constituency/dependency):	nltk, stanza, benepar
- TreeLSTM Implementations: treelstm-pytorch, torch-struct, custom PyTorch

# Parameter-Efficient Fine-Tuning Techniques (PEFT)
Instead of fine-tuning all parameters of large transformer models, PEFT techniques allow training only a small subset, making them faster and lighter — ideal for low-resource scenarios.

Techniques Overview
Adapters: add small trainable layers between transformer blocks, it is modular, reusable	but has a slight overhead
LoRA:	Inject low-rank matrices into attention layers.	Very memory efficient	but slightly harder to implement
Prompt Tuning: Learn special prompt tokens, freeze the base model. Minimal parameter changes	but needs large data for good results
BitFit: Train only bias terms in the model: Super lightweight, easy to apply	but slightly less accurate in some tasks
Tools for PEFT
peft	--> HuggingFace's library for Adapters, LoRA, Prompt Tuning, etc.

# Text Processing 

I uploaded a notebook where i've done some text processing on the dataset: Lowercasing, Removing Punctuation, Removing stopwords, Removing special characters. I also used SpaCy for try some lemmatization.  
