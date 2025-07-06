import nltk
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from collections import Counter


def wordnet_stats():
    nltk.download('wordnet', quiet=True)

    synsets = list(wn.all_synsets())
    print(f"All synsets: {len(synsets)}")

    lemmas = set()
    definitions = 0
    pos_counts = Counter()
    example_sentences = 0

    for syn in synsets:
        # Считаем леммы (уникальные слова)
        for lemma in syn.lemmas():
            lemmas.add(lemma.name())

        # Считаем определения
        if syn.definition():
            definitions += 1

        # POS
        pos_counts[syn.pos()] += 1

        # Примеры
        if syn.examples():
            example_sentences += len(syn.examples())

    print(f"Уникальных лемм (слов): {len(lemmas)}")
    print(f"Определений (glosses): {definitions}")
    print(f"Примеров предложений: {example_sentences}")
    print("Распределение частей речи (POS):")
    for pos, count in pos_counts.items():
        pos_full = {
            'n': 'Существительные (noun)',
            'v': 'Глаголы (verb)',
            'a': 'Прилагательные (adj)',
            's': 'Прилагательные (satellite adj)',
            'r': 'Наречия (adv)',
        }.get(pos, pos)
        print(f"  {pos_full}: {count}")


wordnet_stats()
########################################################

def max_definition_length():
    nltk.download('wordnet', quiet=True)

    max_len_chars = 0
    max_len_words = 0
    longest_def = ""
    longest_target = ""

    for syn in wn.all_synsets():
        gloss = syn.definition()
        gloss_len_chars = len(gloss)
        gloss_len_words = len(gloss.split())

        if gloss_len_chars > max_len_chars:
            max_len_chars = gloss_len_chars
            longest_def = gloss
            longest_lemmas = syn.lemmas()

        if gloss_len_words > max_len_words:
            max_len_words = gloss_len_words

    source = ": ".join(lemma.name() for lemma in longest_lemmas)
    source = source.replace("_", " ")
    print(f"Максимальная длина определения (символы): {max_len_chars}")
    print(f"Максимальная длина определения (слова): {max_len_words}")
    print(f"Пример самого длинного определения:\n{source}: {longest_def}")


max_definition_length()
#######################################################################

def stat_distribution():

    # 1. Скачай WordNet один раз:
    nltk.download('wordnet')

    # 2. Собираем длины gloss
    lengths = []

    for syn in wn.all_synsets():
        gloss = syn.definition()
        lengths.append(len(gloss.split()))

    # 3. Статистика
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)

    print(f"Мин. длина (слов): {min_len}")
    print(f"Макс. длина (слов): {max_len}")
    print(f"Средняя длина (слов): {avg_len:.2f}")

    # 4. Гистограмма
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=range(0, max_len + 2, 1), edgecolor='black')
    plt.title("Распределение длин определений WordNet")
    plt.xlabel("Длина (слов)")
    plt.ylabel("Частота")
    plt.grid(axis='y')
    plt.show()


#stat_distribution()
