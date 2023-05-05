from evaluate import evaluate
import torch
import torch.nn.functional as F

def train(args,model):
  opt = torch.optim.Adam(model.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])
  best_test_acc = 0
  labels = args['graph'].ndata['label']

  for epoch in range(args['epoches']):
      model.train()
      logits = model(args['graph'],args['graph'].ndata['feat'])
      logp = F.log_softmax(logits, 1)

      if args['loss'] == 'nll_loss':
        loss = F.nll_loss(logp[args['train_mask']], labels[args['train_mask']])
      else:
        loss = F.cross_entropy(logp[args['train_mask']], labels[args['train_mask']])

      opt.zero_grad()
      loss.backward()
      opt.step()
      test_acc = evaluate(model, args['graph'], args['graph'].ndata['feat'], args['graph'].ndata['label'], args['test_mask'])
      
      if best_test_acc < test_acc:
          best_test_acc = test_acc

  return loss.item(),test_acc,best_test_acc, model