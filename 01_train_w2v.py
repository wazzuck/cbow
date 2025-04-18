#
#
#
import tqdm
import wandb
import torch
import dataset
import evaluate
import datetime
import model


#
#
#
torch.manual_seed(42)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


#
#
#
ds = dataset.Wiki()
dl = torch.utils.data.DataLoader(dataset=ds, batch_size=256)


#
#
#
args = (len(ds.int_to_vocab), 128)
mFoo = model.CBOW(*args)
print('mFoo:params', sum(p.numel() for p in mFoo.parameters()))
opFoo = torch.optim.Adam(mFoo.parameters(), lr=0.003)
criterion = torch.nn.CrossEntropyLoss()


#
#
#
wandb.init(project='mlx7-week1-cbow', name=f'{ts}')
mFoo.to(dev)


#
#
#
for epoch in range(5):
  prgs = tqdm.tqdm(dl, desc=f'Epoch {epoch+1}', leave=False)
  for i, (ipt, trg) in enumerate(prgs):
    ipt, trg = ipt.to(dev), trg.to(dev)
    opFoo.zero_grad()
    out = mFoo(ipt)
    loss = criterion(out, trg.squeeze())
    loss.backward()
    opFoo.step()
    wandb.log({'loss': loss.item()})
    if i % 10_000 == 0: evaluate.topk(mFoo)

  # checkpoint
  checkpoint_name = f'{ts}.{epoch + 1}.cbow.pth'
  torch.save(mFoo.state_dict(), f'./checkpoints/{checkpoint_name}')
  artifact = wandb.Artifact('model-weights', type='model')
  artifact.add_file(f'./checkpoints/{checkpoint_name}')
  wandb.log_artifact(artifact)


#
#
#
wandb.finish()
