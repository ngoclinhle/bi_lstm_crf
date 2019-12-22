import sys
import copy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import torchtext

from bi_lstm_crf import BiLSTM_CRF
from bi_lstm_crf import prepare_sequence
from sejong import SejongDataset, unique_everseen

EMBEDDING_DIM = 100
HIDDEN_DIM = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: %s' % device)

train_file = sys.argv[2]
valid_file = sys.argv[4]

WORD = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)
LEX = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, is_target=True)
POS_TAG = torchtext.data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True,
                               unk_token=None, is_target=True)

train = SejongDataset.splits(fields=(WORD, LEX, POS_TAG),
                                    train=train_file,
                                    validation=None,
                                    test=None,
                                    max_token=50)

# build vocab
WORD.build_vocab(train, specials=["<pad>", "<tok>", "<blank>"], min_freq=0)
LEX.build_vocab(train, specials=["<pad>", "<tok>", "<blank>"], min_freq=0)
POS_TAG.build_vocab(train, specials=["<pad>", "<tok>", "<blank>"])

MERGE = copy.deepcopy(WORD)
for k in LEX.vocab.stoi:
    if k not in MERGE.vocab.stoi:
        MERGE.vocab.stoi[k] = len(MERGE.vocab.stoi)
MERGE.vocab.itos.extend(LEX.vocab.itos)
MERGE.vocab.itos = unique_everseen(MERGE.vocab.itos)
MERGE.is_target = True
MERGE.include_lengths = False
LEX = MERGE

train.fields['lex'] = LEX
print('Size of WORD vocab: %d' % len(WORD.vocab))
print('Size of LEX vocab: %d' % len(LEX.vocab))
print('Size of POS_TAG vocab: %d' % len(POS_TAG.vocab))

batch_size = 2

train_batches = torchtext.data.BucketIterator(train, batch_size=batch_size,
                                           sort_key=lambda x: x.word.__len__(),
                                           sort_within_batch=True,
                                           shuffle=True)

# Make up some training data
# training_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]

# word_to_ix = {}
# for sentence, tags in training_data:
#     for word in sentence:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)

tag_to_ix = POS_TAG.vocab.stoi


model = BiLSTM_CRF(len(LEX.vocab), tag_to_ix, embedding_dim=100, hidden_dim=100, batch_size=batch_size).to(device)
model_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(model_params, lr=0.001)

# Check predictions before training
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
#     print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    train_loss = 0
    for batch_count, batch in enumerate(train_batches):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # sentence_in = prepare_sequence(sentence, word_to_ix)
        # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(batch.lex.to(device), batch.pos.to(device)).to(device)
        train_loss += loss
        # print("Epoch %d: loss=%f" % (epoch, loss))

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
    print('epoch ', epoch, 'train loss ', train_loss)

# Check predictions after training
with torch.no_grad():
    for batch in train_batches:
        expected_pos_seq = batch.pos[0]
        predicted_pos_seq = model(batch.lex.to(device))
        print('expected pos', expected_pos_seq)
        print('predicted pos', predicted_pos_seq[0])
        break
# We got it!