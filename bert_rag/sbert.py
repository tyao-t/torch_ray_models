      
from datasets import load_dataset
# Load MNLI dataset from GLUE
# 0 = entailment, 1 = neutral, 2 = contradiction
train_dataset = load_dataset(
"glue", "mnli", split="train"
).select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")

print(train_dataset[2])
"""{'premise': 'One of our number will carry out your instructions minutely.',
'hypothesis': 'A member of my team will execute your orders with immense
precision.',
'label': 0}"""

from sentence_transformers import SentenceTransformer
# Use a base model
embedding_model = SentenceTransformer('bert-base-uncased')

from sentence_transformers import losses
# Define the loss function. In softmax loss, we will also need to explicitly
# set the number of labels.

train_loss = losses.SoftmaxLoss(model=embedding_model,
    sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
    num_labels=3
)

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine",
)

# Start training
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(
    output_dir="base_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

from sentence_transformers.trainer import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

evaluator(embedding_model)

trainer.accelerator.clear()
del trainer, embedding_model

# Garbage collection and empty cache
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

from mteb import MTEB

evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(embedding_model)
results

# Cosine Similarity Loss
from datasets import Dataset, load_dataset

train_dataset = load_dataset("glue", "mnli", split="train").select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")

mapping = {2: 0, 1: 0, 0:1}
train_dataset = Dataset.from_dict({
    "sentence1": train_dataset["premise"],
    "sentence2": train_dataset["hypothesis"],
    "label": [float(mapping[label]) for label in train_dataset["label"]]
})

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Create an embedding similarity evaluator for stsb
val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine"
)

from sentence_transformers import losses, SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

embedding_model = SentenceTransformer('bert-base-uncased')

train_loss = losses.CosineSimilarityLoss(model=embedding_model)

args = SentenceTransformerTrainingArguments(
    output_dir="cosineloss_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

# MNR Loss
import random
from tqdm import tqdm
from datasets import Dataset, load_dataset

mnli = load_dataset("glue", "mnli", split="train").select(range(50_000))
mnli = mnli.remove_columns("idx")
mnli = mnli.filter(lambda x: True if x['label'] == 0 else False)

train_dataset = {"anchor": [], "positive": [], "negative": []}
soft_negatives = mnli["hypothesis"]
random.shuffle(soft_negatives)
for row, soft_negative in tqdm(zip(mnli, soft_negatives)):
    train_dataset["anchor"].append(row["premise"])
    train_dataset["positive"].append(row["hypothesis"])
    train_dataset["negative"].append(soft_negative)
train_dataset = Dataset.from_dict(train_dataset)
len(train_dataset)

# Supervised fune tuning, just change the model from bert-base-uncased to this
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Augmented SBERT

"""
Augmented SBERT Implementation
This script demonstrates the Augmented SBERT technique which uses a cross-encoder
to label additional training data for improving bi-encoder (SBERT) performance.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.datasets import NoDuplicatesDataLoader
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Step 1: Fine-tune a cross-encoder
# ------------------------------------------------------------------------------

print("Step 1: Fine-tuning cross-encoder on gold dataset")

# Load and prepare a small set of 10000 documents for the cross-encoder (gold dataset)
dataset = load_dataset("glue", "mnli", split="train").select(range(10_000))

# Map MNLI labels to binary classification (0: not similar, 1: similar)
# MNLI labels: 0=entailment, 1=neutral, 2=contradiction
# We map entailment to 1 (similar), neutral and contradiction to 0 (not similar)
mapping = {2: 0, 1: 0, 0: 1}

# Create DataLoader for cross-encoder training
gold_examples = [
    InputExample(texts=[row["premise"], row["hypothesis"]], label=mapping[row["label"]])
    for row in tqdm(dataset, desc="Preparing gold examples")
]
gold_dataloader = NoDuplicatesDataLoader(gold_examples, batch_size=32)

# Create Pandas DataFrame for easier data handling
gold = pd.DataFrame({
    'sentence1': dataset['premise'],
    'sentence2': dataset['hypothesis'],
    'label': [mapping[label] for label in dataset['label']]
})

# Initialize and train cross-encoder
cross_encoder = CrossEncoder('bert-base-uncased', num_labels=2)
cross_encoder.fit(
    train_dataloader=gold_dataloader,
    epochs=1,
    show_progress_bar=True,
    warmup_steps=100,
    use_amp=False  # Set to True if you have CUDA and want mixed precision
)

print("Cross-encoder training completed")

# Step 2: Create new sentence pairs
# ------------------------------------------------------------------------------

print("\nStep 2: Preparing silver dataset")

# Load additional data for silver dataset (40,000 examples)
silver = load_dataset("glue", "mnli", split="train").select(range(10_000, 50_000))
pairs = list(zip(silver['premise'], silver['hypothesis']))

# Step 3: Label new sentence pairs with the fine-tuned cross-encoder
# ------------------------------------------------------------------------------

print("Step 3: Labeling silver dataset with cross-encoder")

# Predict labels for the silver dataset using our fine-tuned cross-encoder
output = cross_encoder.predict(pairs, apply_softmax=True, show_progress_bar=True)

# Create silver dataset DataFrame with predicted labels
silver = pd.DataFrame({
    "sentence1": silver["premise"],
    "sentence2": silver["hypothesis"],
    "label": np.argmax(output, axis=1)  # Convert softmax outputs to class labels
})

print(f"Silver dataset created with {len(silver)} examples")

# Step 4: Train a bi-encoder (SBERT) on the extended dataset (gold + silver)
# ------------------------------------------------------------------------------

print("\nStep 4: Training bi-encoder on combined gold + silver dataset")

# Combine gold and silver datasets
data = pd.concat([gold, silver], ignore_index=True, axis=0)
data = data.drop_duplicates(subset=['sentence1', 'sentence2'], keep="first")
train_dataset = Dataset.from_pandas(data, preserve_index=False)

print(f"Combined dataset size: {len(data)} examples")

# Prepare evaluator using STS benchmark validation set
val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],  # Normalize scores to 0-1 range
    main_similarity="cosine",
    name="sts-validation"
)

# Initialize bi-encoder model
embedding_model = SentenceTransformer('bert-base-uncased')

# Define loss function (CosineSimilarityLoss for similarity learning)
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

# Configure training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="augmented_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,  # Use mixed precision training if supported
    eval_steps=100,
    logging_steps=100,
    save_steps=500,
)

# Initialize trainer
trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)

# Train the model
trainer.train()

# Evaluate the trained model
print("Evaluating augmented model:")
eval_score = evaluator(embedding_model)
print(f"Augmented model evaluation score: {eval_score}")

# Clean up accelerator state
trainer.accelerator.clear()

# Step 5: Evaluate without silver dataset (baseline comparison)
# ------------------------------------------------------------------------------

print("\nStep 5: Training and evaluating baseline model (gold dataset only)")

# Use only gold dataset for baseline comparison
data_gold_only = pd.concat([gold], ignore_index=True, axis=0)
data_gold_only = data_gold_only.drop_duplicates(subset=['sentence1', 'sentence2'], keep="first")
train_dataset_gold = Dataset.from_pandas(data_gold_only, preserve_index=False)

print(f"Gold-only dataset size: {len(data_gold_only)} examples")

# Initialize fresh model for baseline
embedding_model_baseline = SentenceTransformer('bert-base-uncased')

# Use same loss function
train_loss_baseline = losses.CosineSimilarityLoss(model=embedding_model_baseline)

# Configure training arguments (same as before)
args_baseline = SentenceTransformerTrainingArguments(
    output_dir="gold_only_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
    save_steps=500,
)

# Initialize trainer for baseline
trainer_baseline = SentenceTransformerTrainer(
    model=embedding_model_baseline,
    args=args_baseline,
    train_dataset=train_dataset_gold,
    loss=train_loss_baseline,
    evaluator=evaluator
)

# Train baseline model
trainer_baseline.train()

# Evaluate baseline model
print("Evaluating baseline model:")
eval_score_baseline = evaluator(embedding_model_baseline)
print(f"Baseline model evaluation score: {eval_score_baseline}")

# Print comparison results
print("\n" + "="*50)
print("COMPARISON RESULTS:")
print(f"Augmented model (gold + silver): {eval_score:.4f}")
print(f"Baseline model (gold only):     {eval_score_baseline:.4f}")
print(f"Improvement:                   {eval_score - eval_score_baseline:.4f}")
print("="*50)

# Clean up
trainer_baseline.accelerator.clear()

print("\nTraining completed! Models saved in 'augmented_embedding_model' and 'gold_only_embedding_model' directories")



# Unsupervised learning
"""
Transformer-based Denoising AutoEncoder (TSDAE) Implementation
Unsupervised learning approach for sentence embeddings using denoising autoencoders.
"""

import nltk
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers import models, SentenceTransformer, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Download required NLTK data for tokenization
print("Downloading NLTK punkt tokenizer...")
nltk.download('punkt', quiet=True)
print("Download completed!")

# Step 1: Prepare the dataset
# ------------------------------------------------------------------------------

print("\nStep 1: Preparing dataset for TSDAE training")

# Load MNLI dataset and select 25,000 examples
mnli = load_dataset("glue", "mnli", split="train").select(range(25_000))

# Create a flat list of sentences from premises and hypotheses
flat_sentences = mnli["premise"] + mnli["hypothesis"]

# Remove duplicates and create unique sentence list
unique_sentences = list(set(flat_sentences))
print(f"Created dataset with {len(unique_sentences)} unique sentences")

# Step 2: Add noise to create damaged sentences
# ------------------------------------------------------------------------------

print("\nStep 2: Creating damaged sentences for denoising task")

# Create damaged data using default noise function (delete ratio = 0.6)
damaged_data = DenoisingAutoEncoderDataset(unique_sentences)

# Alternative: Custom noise function with different deletion ratio
# damaged_data = DenoisingAutoEncoderDataset(
#     unique_sentences,
#     noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s, del_ratio=0.6)
# )

# Convert to Hugging Face Dataset format
train_dataset = {"damaged_sentence": [], "original_sentence": []}

for data in tqdm(damaged_data, desc="Creating damaged sentences"):
    train_dataset["damaged_sentence"].append(data.texts[0])  # Noisy/damaged sentence
    train_dataset["original_sentence"].append(data.texts[1])  # Original sentence

train_dataset = Dataset.from_dict(train_dataset)

# Display sample of the damaged data
print("\nSample of damaged vs original sentences:")
print(f"Damaged: {train_dataset[0]['damaged_sentence']}")
print(f"Original: {train_dataset[0]['original_sentence']}")

# Step 3: Prepare evaluator for model performance tracking
# ------------------------------------------------------------------------------

print("\nStep 3: Setting up evaluation metrics")

# Create evaluator using STS benchmark validation set
val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],  # Normalize scores to 0-1 range
    main_similarity="cosine",
    name="sts-validation"
)

print("Evaluator ready - will use STS benchmark for validation")

# Step 4: Initialize the sentence transformer model
# ------------------------------------------------------------------------------

print("\nStep 4: Initializing Sentence Transformer model")

# Create the embedding model architecture
word_embedding_model = models.Transformer('bert-base-uncased')
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), 
    'cls'  # Use CLS token pooling for TSDAE
)
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print(f"Model initialized with {embedding_model.get_sentence_embedding_dimension()} dimensions")

# Step 5: Define the TSDAE loss function
# ------------------------------------------------------------------------------

print("\nStep 5: Setting up Denoising AutoEncoder loss")

# Use the denoising auto-encoder loss with tied encoder-decoder weights
train_loss = losses.DenoisingAutoEncoderLoss(
    embedding_model, 
    tie_encoder_decoder=True  # Share weights between encoder and decoder
)

# Move decoder to GPU if available (for faster training)
try:
    train_loss.decoder = train_loss.decoder.to("cuda")
    print("Decoder moved to GPU for faster training")
except:
    print("GPU not available, using CPU for training")

print("TSDAE loss function configured")

# Step 6: Configure training arguments
# ------------------------------------------------------------------------------

print("\nStep 6: Configuring training parameters")

args = SentenceTransformerTrainingArguments(
    output_dir="tsdae_embedding_model",  # Directory to save model
    num_train_epochs=1,                  # Number of training epochs
    per_device_train_batch_size=16,      # Batch size for training
    per_device_eval_batch_size=16,       # Batch size for evaluation
    warmup_steps=100,                    # Warmup steps for learning rate
    fp16=True,                           # Use mixed precision training
    eval_steps=100,                      # Evaluate every 100 steps
    logging_steps=100,                   # Log metrics every 100 steps
    save_steps=500,                      # Save checkpoint every 500 steps
    learning_rate=2e-5,                  # Learning rate
    weight_decay=0.01,                   # Weight decay for regularization
)

print("Training arguments configured")

# Step 7: Initialize trainer and start training
# ------------------------------------------------------------------------------

print("\nStep 7: Starting TSDAE training")

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator  # Optional: monitor performance during training
)

# Start the training process
print("Beginning training...")
trainer.train()

print("Training completed successfully!")

# Step 8: Evaluate the final model
# ------------------------------------------------------------------------------

print("\nStep 8: Evaluating final model performance")

final_score = evaluator(embedding_model)
print(f"Final model evaluation score on STS benchmark: {final_score:.4f}")

# Step 9: Save the model (optional)
# ------------------------------------------------------------------------------

print("\nStep 9: Saving the trained model")

# Save the final model
embedding_model.save("tsdae_final_model")
print("Model saved to 'tsdae_final_model' directory")

print("\n" + "="*60)
print("TSDAE TRAINING COMPLETE!")
print("="*60)
print(f"Trained on: {len(unique_sentences)} unique sentences")
print(f"Final STS benchmark score: {final_score:.4f}")
print("Model saved for future use")
print("="*60)
