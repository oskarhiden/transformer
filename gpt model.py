import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

perform_training = True
perform_load_model = False

# Hyperparameters
lr = 1e-3
epochs = 10000
batch_size = 32  # Bathces used in training for efficiency
block_size = 8  # Nr of chars to use at a time
n_embed = 32  # Number of embedding dimensions
eval_iter = 10  # How manny iterations when evaluating
n_heads = 4
n_layers = 2
dropout = 0.2   # Dropout rate

# get uniqe tokens from text
tokens = sorted(list(set(text)))
vocab_size = len(tokens)
print("Potential tokens: ", ''.join(tokens))

# create a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(tokens)}
idx2char = np.array(tokens)

def encoder(text):
    return np.array([char2idx.get(c) for c in text])

def decoder(arr):
    # Integer array indexing allows selection of arbitrary items in the array based on their N-dimensional index.
    # Each integer array represents a number of indices into that dimension.
    return ''.join(idx2char[arr])


# encode and store data in transformer
data = torch.tensor(encoder(text), dtype=torch.long)

# Split text into training and test data
n = len(data)*0.9
train_data = data[:int(n)]
test_data = data[int(n):]


def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = test_data

    # Take out the batches
    rand = torch.randint(0, len(data)-block_size, (batch_size,)) # len(data) ≤ rand < block_size, therefore no +1
    x = torch.stack([data[i:i+block_size] for i in rand])
    y = torch.stack([data[i+1:i+block_size+1] for i in rand])

    return x, y

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # Attention
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # Randomly preventing some neurons from updating their weights
        # Weighted aggregation
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ Linear layer folowd by ReLU for non linearity"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ One block of transformer """
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Self-attention
        x = x + self.ff(self.ln2(x))  # Feed-forward
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential( *[Block(n_embed=n_embed, n_heads=n_heads) for _ in range(n_layers)] )
        self.ln_final = nn.LayerNorm(n_embed)
        #self.sa_head = MultiHeadAttention(4, n_embed//4)  # devide by 4 to then get size n_embed when cocatenated
        #self.ff = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C) C = n_embed
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T,C) arange inputs 0,1,2.... to T dvs the pos
        x = tok_emb + pos_emb  # (B,T,C)  # add the two embeddings
        #x = self.sa_head(x)  # (B,T,C)
        #x = self.ff(x)  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_final(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # cross_entropy expects vocab_sixe (C) to be the second input, therefore rearange
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]  # (B,block_size)
            # get predictions
            logits, loss = self(idx_cond)
            # last timestep B, T, C. T=box
            logits = logits[:, -1, :]  # Now (B, C)
            # Softmax
            prob = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(prob, num_samples=1)  # Now (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


@torch.no_grad()
def evaluate_los():
    out = {}
    m.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            X, Y = get_batch(split)
            logit, loss = m(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


m = BigramLanguageModel()

## Training
if perform_training:
    # Optimizer
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        x_t, y_t = get_batch('train')
        logits, loss = m(x_t, y_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0 or epoch == epochs - 1:
            # Evaluate
            losses = evaluate_los()
            print(f"Epoch {epoch}, train loss {losses['train']:.3f}, test loss {losses['test']:.3f}")

    # store model
    torch.save(m, 'model.pt')


# test
if perform_load_model:
    m = torch.load('model.pt')
start = torch.zeros((1, 1), dtype=torch.long) # Jsut a char 0, which is a space
coded_res = m.generate(idx=start, max_new_tokens=1000)
print(decoder(coded_res[0].tolist()))


# Tn this implementation there is only a decoder
# In text prediction decoder is only needed
# Encoder-Decoder structure is used in e.g. translation. where you have an
# full text (to translate) that you condition on. It adds info to your text
# genereation.

# French to English translation example:

# <--------- ENCODE ------------------><--------------- DECODE ----------------->
# les réseaux de neurones sont géniaux! <START> neural networks are awesome!<END>
