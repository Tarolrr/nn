import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def read_file(filename):
    with open(filename, 'r') as f:
        return f.read() 
    
text = read_file('tinyshakespeare.txt')
vocab = sorted(set(text))
print(vocab)

char2idx = lambda c: [vocab.index(c_i) for c_i in c]
idx2char = lambda i: "".join([vocab[i_i] for i_i in i])

text_tensor = torch.tensor(char2idx(text), dtype=torch.long)

train_text_tensor = text_tensor[:int(0.9*len(text_tensor))]
val_text_tensor = text_tensor[int(0.9*len(text_tensor)):]

print(len(train_text_tensor), len(val_text_tensor))

def get_batch(block_size, batch_size, mode='train'):
    if mode == 'train':
        text_tensor = train_text_tensor
    elif mode == 'eval':
        text_tensor = val_text_tensor
    else:
        print('mode must be either train or eval')
        raise Exception()
    
    start_indices = torch.randint(0, len(text_tensor) - block_size-1, (batch_size,))
    
    input_seq = [text_tensor[start_idx:start_idx + block_size] for start_idx in start_indices]
    #target_seq = [text_tensor[start_idx + 1:start_idx + 1 + block_size] for start_idx in start_indices]
    target_seq = [text_tensor[start_idx + 1 + block_size] for start_idx in start_indices]
    input_seq = torch.stack(input_seq)
    target_seq = torch.stack(target_seq)
    
    return input_seq, target_seq

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(BigramLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(in_features=self.embedding_dim*block_size, out_features=self.embedding_dim*block_size)
        self.linear = nn.Linear(in_features=self.embedding_dim*block_size, out_features=self.vocab_size)

    def forward(self, x, target_seq):
        #print(x.shape)
        x = self.embedding(x)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        x = self.hidden(x)
        x = self.linear(x)
        #print(x.shape)
        x = x.reshape(-1, self.vocab_size)
        if target_seq is not None:
            # print(x.shape)
            # print(target_seq.shape)
            #raise
            target_seq = target_seq.reshape(-1)
            loss = F.cross_entropy(x, target_seq)
        else:
            loss = None
        # print(x.shape)
        return x, loss
    
    def generate_text(self, context, new_token_count):
        generated_text = context.view(-1)
        for i in range(new_token_count):
            y, _ = self(context, None)
            # print(y.shape)
            y_last = y[-1, :]
            # print(y_last.shape)
            next_token = torch.multinomial(F.softmax(y_last, dim=0), num_samples=1)
            #print(next_token.shape)
            #print(generated_text.shape)
            generated_text = torch.cat([generated_text, next_token], dim=-1)
            next_token = next_token.reshape(-1, 1)
            #print(next_token.shape)
            #print(context[:,1:].shape)
            context = torch.cat([context[:,1:], next_token], dim=-1)
        return generated_text


# perform training of the model
# initialize variables: block_size, batch_size, num_epochs, learning_rate

block_size = 8
batch_size = 32
num_epochs = 50000
learning_rate = 1e-4

# create an instance of the model

model = BigramLanguageModel(vocab_size=len(vocab), embedding_dim=10)

# create an instance of the optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# generate text from the model before training
# supply space as the context by using method char2idx
# generate 100 tokens

context = torch.tensor([char2idx('        ')])
generated_text = model.generate_text(context, new_token_count=100)

# convert the generated text to a string and print it
print("Untrained model generated text:")
generated_text = idx2char(generated_text.view(-1))
print(generated_text)
print("------------------")
# track the loss and the index for graphing purposes

losses = []
indices = []

# set the model to train mode

model.train()

# iterate over the number of epochs

for epoch in range(num_epochs):
    # get batches of input and target sequences
    input_seq, target_seq = get_batch(block_size=block_size, batch_size=batch_size, mode='train')

    # zero the gradients

    optimizer.zero_grad()

    # apply the model to the input sequence

    y, loss = model(input_seq, target_seq)

    # add the loss to the list

    losses.append(loss.item())
    indices.append(epoch)

    # perform backpropagation

    loss.backward()

    # perform gradient descent

    optimizer.step()

    # print the total loss for the epoch every 1000 epochs

    if epoch % 1000 == 0:
        print(f'epoch: {epoch}, loss: {loss}')

# lets smooth the loss curve by averaging over 100 points
# we also need to adjust the indices

smoothed_losses = []
smoothed_indices = []

for i in range(len(losses) // 100):
    smoothed_losses.append(sum(losses[i*100:(i+1)*100]) / 100)
    smoothed_indices.append((i+1)*100)


context = torch.tensor([char2idx('        ')])
print(context.shape)

generated_text = model.generate_text(context, new_token_count=100)

# convert the generated text to a string and print it
print("Trained model generated text:")
generated_text = idx2char(generated_text.view(-1))
print(generated_text)
print("------------------")

# plot the smoothed loss curve

plt.plot(smoothed_indices, smoothed_losses)
plt.show()