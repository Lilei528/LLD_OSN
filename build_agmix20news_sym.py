import random

train_file = r'train.csv'
ag_file = r'D:ag.csv'
output_file = r'train_agmix20news_sym0.6.csv'
random.seed()

num_classes = 20
noise_ratio = 0.6


with open(train_file, 'r', encoding='utf-8') as f:
    train_lines = f.readlines()

noisy_train_lines = []
for line in train_lines:
    if not line.strip():
        continue
    label_str = line.strip().split('\t')[0]
    try:
        sentence = line.strip().split('\t')[1]
    except:
        sentence=''
    label = int(label_str)


    if random.random() < noise_ratio:
        noisy_label = random.choice([i for i in range(num_classes) if i != label])
        noisy_train_lines.append(f"{noisy_label}\t{sentence}\n")
    else:
        noisy_train_lines.append(line)


with open(ag_file, 'r', encoding='utf-8') as f:
    ag_sentences = f.readlines()

for sentence in ag_sentences:
    if not sentence.strip():
        continue
    random_label = random.randint(0, num_classes - 1)
    noisy_train_lines.append(f"{random_label}\t{sentence.strip()}\n")


with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(noisy_train_lines)
