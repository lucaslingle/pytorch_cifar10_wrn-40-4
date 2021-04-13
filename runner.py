import torch as tc


class Runner:
    def __init__(self, max_epochs, verbose=True):
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.global_step = 0 # would be nice if we could checkpoint this like in tensorflow; look into later

    def train_epoch(self, model, train_dataloader, optimizer, scheduler, device, loss_fn):
        for batch_idx, (X, y) in enumerate(train_dataloader, 1):
            X, y = X.to(device), y.to(device)

            # Forward
            logits = model(X)
            loss = loss_fn(logits, y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update global step
            self.global_step += 1

            if self.verbose and batch_idx % 100 == 0:
                loss = loss.item()
                print(f"loss: {loss:>7f}  [{batch_idx:>5d}/{len(train_dataloader):>5d}]")

        scheduler.step() # for zagoruyko learning rate schedule on CIFAR-10, we step this once per epoch
        return

    def evaluate_epoch(self, model, dataloader, device, loss_fn):
        num_test_examples = len(dataloader.dataset)
        test_loss, correct = 0, 0
        with tc.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += len(X) * loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(tc.float).sum().item()
        test_loss /= num_test_examples
        correct /= num_test_examples
        return {
            "accuracy": correct,
            "loss": test_loss
        }

    def run(self, model, train_dataloader, test_dataloader, device, loss_fn, optimizer, scheduler):

        for epoch in range(1, self.max_epochs+1):
            if self.verbose:
                print(f"Epoch {epoch}\n-------------------------------")

            model.train()  # turn batchnorm, dropout, etc. to train mode.
            self.train_epoch(model, train_dataloader, optimizer, scheduler, device, loss_fn)

            # after every epoch, print stats for test set. should be validation set. fix later.
            model.eval()  # turn batchnorm, dropout, etc. to eval mode.
            test_eval_dict = self.evaluate_epoch(model, test_dataloader, device, loss_fn)
            test_accuracy = test_eval_dict['accuracy'] * 100
            test_loss = test_eval_dict['loss']
            if self.verbose:
                print(f"Test Error: \n Accuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f}\n")

            if epoch % 10 == 0:
                tc.save(model.state_dict(), "model.pth")
                tc.save(optimizer.state_dict(), "optimizer.pth")
                tc.save(scheduler.state_dict(), "scheduler.pth")