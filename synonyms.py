import nltk
from nltk.corpus import wordnet as wn

# Скачай WordNet (если ещё не скачан)
nltk.download('wordnet')

def count_synonyms(words):
    results = {}
    for word in words:
        synsets = wn.synsets(word)
        lemmas = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                lemmas.add(lemma.name())
        results[word] = {
            "synonym_count": len(lemmas),
            "synonyms": sorted(lemmas)
        }
    return results

# Пример слов
words = ["run", "happy", "dog", "big"]

result = count_synonyms(words)

for word, data in result.items():
    print(f"\nСлово: {word}")
    print(f"Количество синонимов: {data['synonym_count']}")
    print(f"Синонимы: {data['synonyms']}")
