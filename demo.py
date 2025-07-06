
import nltk
from nltk.corpus import wordnet as wn
import torch
from transformers import BertTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset


exit(0)

#####################################################
# Скачиваем WordNet
nltk.download('wordnet')

# 1. Загружаем определения из WordNet
definitions = []
for synset in wn.all_synsets():
    def_text = synset.definition()
    if def_text:
        definitions.append(def_text)

print(f"Loaded {len(definitions)} definitions.")

# 2. Создаём датасет HuggingFace
dataset = Dataset.from_dict({"text": definitions})

# 3. Загружаем токенизатор и модель
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Функция для токенизации с маскированием (используем динамическое маскирование)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Создаём функцию подготовки входных данных с маскированием для MLM
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# 4. Настройки обучения
training_args = TrainingArguments(
    output_dir="./bert-wordnet",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
)

# 5. Создаём Trainer и запускаем обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
