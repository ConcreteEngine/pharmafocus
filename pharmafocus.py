from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline

# Load the BioASQ Dataset from the specified source
dataset = load_dataset("kroshan/BioASQ")

# Inspect the first example in the training set
print("Sample from the dataset:", dataset['train'][0])

# Load the BioGPT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")

# Preprocess and tokenize the dataset
def preprocess_and_tokenize(examples):
    tokenized_inputs = tokenizer(examples['question'], padding="max_length", truncation=True, max_length=128)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # Copy input_ids to labels
    return tokenized_inputs

# Apply preprocessing and tokenization
tokenized_dataset = dataset.map(preprocess_and_tokenize, batched=True)

# Define a Data Collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Load the BioGPT model
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,  # Limited to one epoch due to resource constraints
    weight_decay=0.01,
    no_cuda=True  # Use CPU
)

# Initialize the Trainer with the Data Collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'] if 'validation' in tokenized_dataset else None,
    data_collator=data_collator  # Use the data collator for language modeling
)

# Train the model
trainer.train()

# Generate Answers Using the Trained Model
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# Refined prompt
prompt = "Doxycycline is an antibiotic that works by"

# Generate an answer
generated_text = text_generator(prompt, max_length=100)
print("Generated Answer:", generated_text[0]['generated_text'])

# Optional: Post-processing to clean the generated text
def clean_generated_text(text):
    # Adjust to remove incomplete words but preserve meaningful content
    last_period = text.rfind(".")
    if last_period != -1:
        text = text[:last_period + 1]  # Keep content up to the last full stop
    return text.strip()

# Apply post-processing
cleaned_text = clean_generated_text(generated_text[0]['generated_text'])
print("Cleaned Generated Answer:", cleaned_text)






