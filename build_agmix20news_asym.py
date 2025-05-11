import random

train_file = r'train.csv'
ag_file = r'D:ag.csv'
output_file = r'train_agmix20news_asym0.6.csv'
random.seed()

noise_ratio = 0.5
num_classes = 20

confusable_groups = {
    "comp": [1, 2, 3, 4, 5],
    "rec": [7, 8, 9, 10],
    "sci": [11, 12, 13, 14],
    "talk": [16, 17, 18, 19],
    "religion": [0, 15, 19],
}


label_to_confusables = {}
for group in confusable_groups.values():
    for label in group:
        label_to_confusables[label] = [l for l in group if l != label]


with open(train_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

noisy_lines = []
for line in lines:
    if not line.strip():
        continue
    label_str = line.strip().split('\t')[0]
    try:
        sentence = line.strip().split('\t')[1]
    except:
        sentence=''
    label = int(label_str)

    if random.random() < noise_ratio:

        if label in label_to_confusables and random.random() < 0.8:

            noisy_label = random.choice(label_to_confusables[label])
        else:

            noisy_label = random.choice([i for i in range(num_classes) if i != label])
        noisy_lines.append(f"{noisy_label}\t{sentence}\n")
    else:

        noisy_lines.append(line)

with open(ag_file, 'r', encoding='utf-8') as f:
    ag_sentences = f.readlines()

for sentence in ag_sentences:
    if not sentence.strip():
        continue
    random_label = random.randint(0, num_classes - 1)
    noisy_lines.append(f"{random_label}\t{sentence.strip()}\n")


with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(noisy_lines)
