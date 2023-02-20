from torch import optim
optimizer = optim.Adam(model.parameters(), lr=0.00001)
loss_fn = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from torchmetrics import Accuracy
accuracy = Accuracy(mdmc_reduce = "global").to(device)

epoch_count =0
for epoch in range(10):
  # train configuration ============================================
  epoch_count += 1
  print('epoch :', epoch_count)

  model.train()

  loss_sum = 0
  accuracy_sum = 0
  count = 0

  for x,y in tqdm(train_dataloader):
    x = x.to(device)
    y = y.to(device)

    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    acc =  accuracy(y_pred, y)

    accuracy_sum += acc.item()
    loss_sum += loss.item()
    count += 1

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
  
  avg_loss = loss_sum/count
  avg_accuracy = accuracy_sum/count
  print('\naverage train loss :', avg_loss)
  print('average train accuracy :', avg_accuracy)

  # valiation configuration=====================================================
  model.eval()

  loss_sum = 0
  accuracy_sum = 0
  count = 0
  for x, y in val_dataloader:
    x = x.to(device)
    y = y.to(device)

    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    acc =  accuracy(y_pred, y)

    accuracy_sum += acc.item()
    loss_sum += loss.item()
    count += 1

  avg_loss = loss_sum/count
  avg_accuracy = accuracy_sum/count
  print('average validation loss :', avg_loss)
  print('average validation accuracy :', avg_accuracy)