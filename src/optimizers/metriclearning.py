import torch as th
from torch.utils.data.dataloader import DataLoader
from pytorch_metric_learning import miners, losses
from tqdm import tqdm


def optimize_triplets(loader: DataLoader, model: th.nn.Module, epochs: int=3, cuda: bool=False, verbose: bool=True) -> None:
    margin = 0.05
    embed_loss = losses.TripletMarginLoss(margin=margin)
    miner = miners.TripletMarginMiner(margin=margin, type_of_triplets='all')

    if cuda:
        embed_loss = embed_loss.cuda()
        miner = miner.cuda()

    model.train()
    train_params = [
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 1e-1},
    ]
    optim = th.optim.SGD(train_params, momentum=0.8, weight_decay=5e-4)
    scheduler = th.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.1)
    
    for epoch in range(epochs):
        print('Epoch %d' % (epoch + 1))
        train_loss = 0.0
        count = 0
        p_bar = tqdm(loader)
        if verbose:
            p_bar.set_description('# of triples: %4d Loss: %.4f' % (-1, -1))
        for x, y in p_bar:
            count += 1
            optim.zero_grad()
            if cuda:
                x, y = x.cuda(), y.cuda()
            x = model.forward(x).squeeze_()
            samples = miner(x, y)
            loss = embed_loss(x, y, samples)
            if loss.item() > 0.0:
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss += loss.item()
            if verbose:
                p_bar.set_description('# of triples: %4d prev. loss: %.4f' % (len(samples[0]), loss.item()))
        print('Average loss: %.4f' % (train_loss / count))
        scheduler.step()
