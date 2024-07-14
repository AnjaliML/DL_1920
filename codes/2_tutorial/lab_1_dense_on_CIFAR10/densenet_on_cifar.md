start = time.time()
	•	Purpose: Records the starting time of the training process to measure elapsed time later.

criterion = nn.CrossEntropyLoss()
	•	Purpose: Defines the loss function. CrossEntropyLoss is used for classification tasks with multiple classes.

lr = 0.01
momentum = 0.5
num_epochs = 20
	•	Purpose: Sets hyperparameters for the training process.
	•	lr: Learning rate, which controls how much to adjust the model's weights in response to the gradient.
	•	momentum: Helps accelerate gradients vectors in the right directions, thus leading to faster converging.
	•	num_epochs: The number of times the entire training dataset will be passed through the network.

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	•	Purpose: Initializes the optimizer SGD (Stochastic Gradient Descent) with model parameters, learning rate, and momentum for weight updates.

for epoch in range(1, num_epochs + 1):
	•	Purpose: Loops over the dataset multiple times (for each epoch).

  for i, (x_batch, y_batch) in enumerate(trainloader):
	•	Purpose: Iterates over batches of data provided by trainloader.

    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
	•	Purpose: Moves the data to the designated device (CPU or GPU).

    optimizer.zero_grad()
	•	Purpose: Clears the old gradients from the previous iteration.

    y_pred = model(x_batch.view(-1, w * h * c))
	•	Purpose: Passes the input batch through the model. view is used to reshape the input data.

    loss = criterion(y_pred, y_batch)
	•	Purpose: Computes the loss between the predicted labels and the true labels.

    loss.backward()
	•	Purpose: Backpropagates the error, computing the gradient of the loss with respect to model parameters.

    optimizer.step()
	•	Purpose: Updates the model parameters based on the computed gradients.

    y_pred_max = torch.argmax(y_pred, dim=1)
	•	Purpose: Finds the indices of the maximum values along dimension 1, which corresponds to the predicted labels.

    correct = torch.sum(torch.eq(y_pred_max, y_batch)).item()
	•	Purpose: Counts the number of correct predictions by comparing predicted and actual labels.

    elapsed = time.time() - start
	•	Purpose: Calculates the elapsed time since training started.

    if not i % 20:
      print(f'epoch: {epoch}, time: {elapsed:.3f}s, loss: {loss.item():.3f}, train accuracy: {correct / batch_size:.3f}')
	•	Purpose: Prints the training progress every 20 batches including epoch number, elapsed time, loss, and training accuracy.

  correct_total = 0
	•	Purpose: Initializes the correct predictions counter for the test set.

  for i, (x_batch, y_batch) in enumerate(testloader):
	•	Purpose: Iterates over batches of data provided by testloader.

    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
	•	Purpose: Moves the data to the designated device.

    y_pred = model(x_batch.view(-1, w * h * c))
    y_pred_max = torch.argmax(y_pred, dim=1)
	•	Purpose: Passes the input batch through the model and finds the predicted labels.

  correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()
	•	Purpose: Counts the correct predictions and accumulates them.

  print(f'Test accuracy: {correct_total / len(testset.data):.3f}')
	•	Purpose: Computes and prints the test accuracy after every epoch.