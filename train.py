def train(model, train_loader, optimizer, criterion, epoch):
    train_loss = 0
    model.train()
    for batch_idx, (data, target, length) in enumerate(train_loader):
        #         data, target = data.to(device), target.to(device)
        #         print(data.shape)
        optimizer.zero_grad()
        output = model(data)

        output = output.transpose(0, 1)

        input_len, batch_size, vocab_size = output.size()
        # encode inputs
        logits = output.log_softmax(2).to(torch.float64)
        #         encoded_texts, text_lens = label_converter.encode(texts)

        #         print(length)

        #         text_lens  = torch.IntTensor(length)

        length = torch.full(size=(batch_size,), fill_value=5, dtype=torch.int32)

        target_lengths = Variable(length)
        logits_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
        # calculate ctc
        loss = criterion(logits, target, logits_lens, target_lengths)

        # #         print (output.shape)
        # # #         ctc_outputs = ctc(inputs,init_hidden)
        # #         ctcloss_inputs = output.transpose(0,1) #SeqLen * BatchSize * Hidden

        # # #         print (ctcloss_inputs.shape)

        # #         label_lens = torch.full(size=(output.shape[0],), fill_value=5, dtype=torch.long)
        # #         act_lens = torch.full(size=(32,), fill_value=5, dtype=torch.long)


        #         log_probs=output.detach().requires_grad_()

        # #         print(log_probs.shape)

        # #         log_probs = log_probs.transpose(1,0)

        # #         print(log_probs.shape)

        #         input_lengths=torch.full((log_probs.shape[0],), log_probs.shape[1],   dtype=torch.long)

        #         input_lengths=Variable(input_lengths)

        #         target_lengths=torch.Tensor(32,).long()
        #         target_lengths=Variable(target_lengths)


        #         loss = criterion(log_probs, target, input_lengths, target_lengths)


        #         loss = criterion(ctcloss_inputs, target, act_lens, label_lens)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 10 == 0 or batch_idx % len(train_loader.dataset) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}\n'.format(train_loss))
    return train_loss



import datetime

dataloaders=idf
criterion=model.criterion_
optimizer=model.optimizer_
num_epochs = 5

train_loss = []
test_loss = []

start_time = datetime.datetime.now()

for epoch in range(1, 5):
        train_loss.append(train(model, trainloader, optimizer, criterion, epoch))
#         test_loss.append(test(model, test_loader, criterion))

learning_time = datetime.datetime.now() - start_time