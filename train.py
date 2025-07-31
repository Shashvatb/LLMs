import torch
import torch.nn as nn
from torch.nn import functional as F

from models.bigram import BigramLanguageModelBasic, BigramLanguageModel

def read_file(file_name):
    with open(file=file_name, mode='r', encoding='utf-8') as f:
        file = f.read()
    print("length of dataset ", file_name, " in characters: ", len(file))
    return file

def get_encoder_decoder(vocabulary):
    # create a mapping for string to integer (for encoding text to string)
    stoi = {ch:i for i, ch in enumerate(vocabulary)}
    # create a mapping for integer to string (for decoding integers to text)
    itos = {i:ch for i, ch in enumerate(vocabulary)}

    # create encoder
    encoder = lambda s: [stoi[c] for c in s]
    # create decoder
    decoder = lambda i: ''.join([itos[c] for c in i])
    
    return encoder, decoder

def get_batch(data, batch_size, block_size, device):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, training_data, val_data, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'train':
            data = training_data
        else:
            data = val_data
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    # training vars
    model = 'bigram_basic'
    eval_interval = 1000
    max_iters = 10000
    batch_size = 32
    block_size = 8
    n_embd = 32

    # read file
    file = read_file('input.txt')

    # get unique characters
    vocab = sorted(set(file))
    vocab_size = len(vocab)
    print("size of vocabulary: ", vocab_size)

    # tokenize the text -> convert characters to machine readable format. can also be called character mapping in this case
    # this is a very basic way of doing this. in more sophisticated algorithms a subword-tokenization is used (google: sentencepiece, gpt2: tiktoken)
    encoder, decoder = get_encoder_decoder(vocab)
    print("encoded values: ", encoder("hello world"))
    print("decoded values: ", decoder(encoder("hello world")))

    # tokenizing the data
    dataset = torch.tensor(encoder(file), dtype=torch.long)
    
    # splitting dataset into train and val
    split_point = int(len(dataset) * 0.9)
    training_data, val_data = dataset[:split_point], dataset[split_point:]
    print("training and validation data size: ", training_data.shape, val_data.shape)

    # we define block size - it is the size of a single data chunk that will be used to train our model (also called context length)
    print("block of data: ", training_data[:block_size+1])
    # when training we always use data size of block size + 1. we use block size as context and the character after that as the target to be predicted
    # we train the model on context all the way from size 1 to block size. we want the model to be used to the different context sizes

    # we define batch size - we need to have multiple blocks (independently) for our model to train. if the amount of blocks is too little it will be inefficient and take a 
    # really long time. If it is too big, it wont fit in our GPU
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    demo_x, demo_y = get_batch(training_data, batch_size, block_size, device)
    print("context and target values in a single batch: ", demo_x, demo_y)

    # model declaration
    if model == 'bigram_basic':
        m = BigramLanguageModelBasic(vocab_size).to(device)
    elif model == 'bigram':
        m = BigramLanguageModel(vocab_size, n_embd, device=device).to(device)

    # model inference
    logits, loss = m(demo_x, demo_y)
    print('shape of logits: ', logits.shape)
    print('loss: ', loss)

    print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    # training basic char based bigram model
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(m, training_data, val_data, eval_interval)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(training_data, batch_size, block_size, device)

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print('trained')
    print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=300)[0].tolist()))


    # understanding self attention
    print('understanding self attention')
    B,T,C = 4,8,2 # batch, time, channels
    x = torch.randn(B,T,C)
    print('shape of random tensor: ', x.shape)

    # we take an average of the tokens in the context
    xbow = torch.zeros((B,T,C)) #initialize bag of words
    for b in range(B):
        for t in range(T):
            xprev = x[b,:t+1] # (t,C) uptil time t (inclusive)
            xbow[b,t] = torch.mean(xprev, 0) # (C) averaging over time (0 dim)
    print('xbow for 0th element', xbow[0])
    # we can be efficient about this with matrix multiplication
    a = torch.tril(torch.ones(T, T)) # tril: lower triangular matrix -> this can be used to do a cumulative sum (or avg) on the time steps to recreate what we did earlier
    a = a / torch.sum(a, 1, keepdim=True) # this helps us convert it from cumulative sum to avg (our goal) -> this will act as our weights
    c = a @ x # this is equicalent to xbow
    print('effieciently dont version: ', c[0])
    # the weights a, can be calculated without being dependent on anything in advance
    # this means we can calculate weights in any way (for example how self attention works) and use it on x without it being inefficient because of the history
    # for attention we want our weights to show the affinity between different tokens ranging from 0 -> 1
    # due to this we implement our weights slightly differently using softmax
    tril = torch.tril(torch.ones(T, T)) # our lower triangular matrix to show which part is the context and which is not
    wei = torch.zeros((T,T)) # initialzing weights to 0 since we do not know the relationship
    wei = wei.masked_fill(tril == 0, float('-inf')) # adding -inf to the future tokens since they are not of use to us
    wei = F.softmax(wei, dim=-1) # calculating softmax on the matrix will give us the same matrix as we did in the previous version
    xbow = wei @ x
    print('softmax based result of attention trick: ', xbow[0])
    
    # coding self attention 
    # it has 3 different trainable layer: key, query and value (KQV)
    # K and Q are used to calculate the weight matrix
    # we have a hyperparam called head size, which tells us the size of the output
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    k = key(x)   # (B, T, 16)
    q = query(x) # (B, T, 16)
    wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
    tril = torch.tril(torch.ones(T, T))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    # V is the transformation for our x which would be multiplied with the weights
    v = value(x)
    out = wei @ v
    print('shape of output of self attention: ', out.shape)








