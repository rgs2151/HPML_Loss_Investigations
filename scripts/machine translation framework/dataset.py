import spacy
spacy_en = spacy.load('en_core_web_md')
spacy_de = spacy.load('de_core_news_md')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

# Source field (German)
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)

# Target field (English)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

train_data.examples = train_data.examples[:len(train_data.examples)//4]
valid_data.examples = valid_data.examples[:len(valid_data.examples)//4]
test_data.examples = test_data.examples[:len(test_data.examples)//4]

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)