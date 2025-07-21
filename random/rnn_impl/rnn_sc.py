import numpy as np

data = open('data/hp_book1.txt', 'r').read()
dt_size = len(data)
chars = list(set("".join(data)))
vocab_size = len(chars)
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}
hddn_size = 40
seq_len = 11
lr = 1e-1

# weight init
Whx, Whh, Why, bh, by = (
    np.random.randn(hddn_size, vocab_size) * 0.01,
    np.random.randn(hddn_size, hddn_size) * 0.01,
    np.random.randn(vocab_size, hddn_size) * 0.01,
    np.zeros((hddn_size, 1)),
    np.zeros((vocab_size, 1)),
)
# initial loss; same random probability of all chars
smooth_loss = -np.log(1.0/vocab_size) * seq_len 

# memory of Adagrad optim
mWhx, mWhh, mWhy, mbh, mby = (
    np.zeros_like(Whx), np.zeros_like(Whh), 
    np.zeros_like(Why), np.zeros_like(bh), np.zeros_like(by)  
)

def sample(h_prev, seed_ix, max_new_tokens):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    outs = []
    h = h_prev.copy()

    for t in range(max_new_tokens):
        # forward through all layer
        h = np.tanh( 
                np.dot(Whx, x ) + np.dot(Whh, h) + bh
            )
        y = np.dot(Why, h) + by
        # compute prob and get next char to follow 
        p = np.exp(y) / np.sum(np.exp(y)) 
        ix = np.random.choice(range(vocab_size), p=p.ravel())

        # overwrite prev char
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        outs.append(ix)
    return outs


def forward(x, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy( hprev ) # put hprev at -1
    loss = 0
    for t in range(seq_len):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][x[t]] = 1 # one-hot
        # ht 
        hs[t] = np.tanh( 
            np.dot(Whx, xs[t] ) + np.dot(Whh, hs[t -1 ]) + bh
            )
        # yt 
        ys[t] = np.dot(Why, hs[t]) + by
        # pt -> prob using softmax
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) 
        # cross entropy loss
        loss += -np.log( ps[t][ targets[t], 0 ] )
    return xs, hs, ys, ps, loss, hs[t]

def backward(xs, hs, ys, ps, hprev, targets):
    """
    xs: t, (vc, 1)
    hs: t, (h, 1)
    ys: t, (vc, 1)
    ps: t, (vc, 1)
    loss: float
    hprev: (h, 1)
    targets: seq
    """
    # zero-grad
    dWhx, dWhh, dWhy, dbh, dby, dhnext = (
        np.zeros_like(Whx), np.zeros_like(Whh), np.zeros_like(Why),
        np.zeros_like(bh), np.zeros_like(by), np.zeros_like(hprev)
    )
    for t in reversed(range(seq_len)):
        # dy, starting of backprob
        dy = ps[t].copy()
        dy[targets[t]] -= 1 # (vc, 1)

        # dWhy & dWb
        dWhy += np.dot(dy, hs[t].T) 
        dby += dy 

        # hidden state backprob
        # dh, we are not adding up, because that is carried by dhnext
        dh = np.dot(Why.T, dy) + dhnext
        # through tanh
        draw = (1 - hs[t] * hs[t]) * dh

        # gradients of the initial eqn
        dbh += draw
        dWhh+= np.dot( draw, hs[t -1 ].T )
        dWhx+= np.dot( draw, xs[t].T )

        # dhnext
        dhnext = np.dot( Whh.T, draw )

    # cliping grads to avoid exploding
    for dparam in [dWhx, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    return dWhx, dWhh, dWhy, dbh, dby

epoch = 1
for _ in range(epoch):
    hprev = np.zeros((hddn_size, 1))
    for pntr in range(0, dt_size - seq_len, seq_len):
        x = [ ctoi[c] for c in data[pntr:pntr+(seq_len)] ] # (vc, seq_len)
        y = [ ctoi[c] for c in data[pntr+1:pntr+(seq_len+1)] ] # (vc, seq_len)
        xs, hs, ys, ps, loss, hprev = forward(x, y, hprev)
        dWhx, dWhh, dWhy, dbh, dby = backward(xs, hs, ys, ps, hprev, y)

        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # log and sample updated model pred
        if pntr % 1000 == 0:
            # loss
            print(f"\n --- iter {pntr}, loss: {smooth_loss}\n")
            # new preds
            preds = sample(hprev, x[0], 100)
            print( '###' + ''.join(itoc[ix] for ix in preds) + '###' )

        # step, update the weights
        for param, dparam, mem in [(Whx, dWhx, mWhx), (Whh, dWhh, mWhh),
                                   (Why, dWhy, mWhy), (bh, dbh, mbh), (by, dby, mby)]:
            mem += dparam * dparam
            param += -lr * dparam / np.sqrt(mem + 1e-8)  # Adagrad update