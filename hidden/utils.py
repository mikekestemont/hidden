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

"""
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
"""

def tuples_from_file(fn):
    return [l.rstrip().split('\t') for l in open(fn)]

def string_from_tuples(tuples):
    chars, _ = zip(*(tuples))
    return ''.join(chars)

def char_tuples(original, segmented):
    segm_labels = []
    for s in segmented:
        segm_labels.append('<S/>')
        for w_idx, w in enumerate(s):
            if w_idx != 0:
                segm_labels.append('<W/>')
            segm_labels.extend(w.replace(' ', ''))      

    text, labels = list(original), []

    for char in text:
        if char == ' ':
            labels.append(0)
        elif segm_labels[0] == char:
            segm_labels.pop(0)
            labels.append(0)
        elif segm_labels[0] in ('<S/>', '<W/>'):
            if segm_labels[0] == '<S/>':
                labels.append((1))
            elif segm_labels[0] == '<W/>':
                labels.append(2)
            while segm_labels[0] in ('<S/>', '<W/>'):
                segm_labels.pop(0)
            segm_labels.pop(0)
    
    return tuple(zip(text, labels))