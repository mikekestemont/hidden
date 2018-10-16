import torch

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    return data.view(bsz, -1).t().contiguous()

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def to_hidden(test, model, dictionary, device):
    model.eval()
    ints = [dictionary.get(c, 0) for c in text]

    with torch.no_grad():
        ints = torch.LongTensor(ints).to(device)
        states, hidden = [], None

        for i in ints:  
            _, hidden = lm(i.view(1, -1), hid_)
            states.append(hidden[-1][0].squeeze().cpu().numpy())

    return np.array(states)