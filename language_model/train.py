# blog: https://machinetalk.org/2019/02/08/text-generation-with-pytorch/
# code: https://github.com/ChunML/NLP

import torch
import torch.nn as nn

import numpy as np
from collections import Counter
from argparse import Namespace
import math

flags = Namespace(
    train_file='oliver.txt',
    seq_size=32,               # define the sequence length
    batch_size=16,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    initial_words=['I', 'am'],
    predict_top_k=5,
    checkpoint_path='checkpoint',
)


def get_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.split()

    # No pre-processing in text (lower case, symbols)
    
    
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)  # a list of zero with same size as in_text
    
    out_text[:-1] = in_text[1:]        # taking all the input words except first one. 
    out_text[-1] = in_text[0]          # setting the last output as first input char.
    # out_text[:-1] means setting all value into 0 to (length-1) index of out
    
    
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)          # they used no dropout
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
        
    #print (output.shape)  # torch.Size([1, 1, voca_length])
    
    _, top_ix = torch.topk(output[0], k=top_k)   #  taking top k words from the voca
    choices = top_ix.tolist()
    
    choice = np.random.choice(choices[0]) # from the top k words, take one randomly
    #print (choice)   # single integer value
    words.append(int_to_vocab[choice])
    #words.append(int_to_vocab[choices[0][0] ])
    #words.append(int_to_vocab[choices[0][1] ])
    

    for _ in range(10):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print(' '.join(words).encode('utf-8'))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
    flags.train_file, flags.batch_size, flags.seq_size)

net = RNNModule(n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size)
net = net.to(device)
criterion, optimizer = get_loss_and_train_op(net, 0.01)

iteration = 0
set_epoch = 20
for e in range( set_epoch ):
    batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
    state_h, state_c = net.zero_state(flags.batch_size)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for x, y in batches:
        iteration += 1
        net.train()

        optimizer.zero_grad()

        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)

        logits, (state_h, state_c) = net(x, (state_h, state_c))
        loss = criterion(logits.transpose(1, 2), y)

        loss_value = loss.item()

        loss.backward()

        state_h = state_h.detach()
        state_c = state_c.detach()

        _ = torch.nn.utils.clip_grad_norm_(
            net.parameters(), flags.gradients_norm)

        optimizer.step()

        if iteration % 100 == 0:
            print('Epoch: {}/{}'.format(e, set_epoch),
                  'Iteration: {}'.format(iteration),
                  'Loss: {}'.format(loss_value),     'ppl: {}'.format( math.exp(loss_value) ) )

        if iteration % 1000 == 0:
            predict(device, net, flags.initial_words , n_vocab,vocab_to_int, int_to_vocab, top_k=5)
            
            torch.save(net.state_dict(),'checkpoint_pt/model-{}.pth'.format(iteration))

