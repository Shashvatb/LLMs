import torch
from models.bigram import BigramLanguageModel

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

def get_batch(data, batch_size, block_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

if __name__ == "__main__":
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
    block_size = 8
    print("block of data: ", training_data[:block_size+1])
    # when training we always use data size of block size + 1. we use block size as context and the character after that as the target to be predicted
    # we train the model on context all the way from size 1 to block size. we want the model to be used to the different context sizes

    # we define batch size - we need to have multiple blocks (independently) for our model to train. if the amount of blocks is too little it will be inefficient and take a 
    # really long time. If it is too big, it wont fit in our GPU
    batch_size = 4
    demo_x, demo_y = get_batch(training_data, batch_size, block_size)
    print("context and target values in a single batch: ", demo_x, demo_y)

    m = BigramLanguageModel(vocab_size, vocab_size)
    logits, loss = m(demo_x, demo_y)
    print(logits.shape)
    print(loss)

    print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    for steps in range(100): # increase number of steps for good results...

        # sample a batch of data
        xb, yb = get_batch(training_data, batch_size, block_size)

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())