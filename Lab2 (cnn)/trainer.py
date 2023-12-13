import os
import torch
from matplotlib import pyplot as plt
# from torch_lr_finder import LRFinder


class Trainer:
    def __init__(self, model, train_ld, val_ld):
        self.device = torch.device('mps')
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), weight_decay=1e-1)

        self.train_ld = train_ld
        self.val_ld = val_ld

        self.history = {
            'train': {
                'losses': [],
                'accs': []
            },
            'valid': {
                'losses': [],
                'accs': []
            },
            'lr': []
        }

        self.best_acc = 0

        self.log_period = 3

        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    def start(self, epochs, clear=True):
        if clear:
            self.clear_history()
        # self.find_lr()
        for epoch in range(epochs):
            train_loss, train_acc = self.train(self.train_ld)
            valid_loss, valid_acc = self.validate(self.val_ld)

            self.update_history(train_loss, train_acc, valid_loss, valid_acc)
            self.log(epoch, epochs, train_loss, valid_loss, train_acc, valid_acc)
            self.checkpoint(valid_acc)

    def clear_history(self):
        self.history = {'train': {'losses': [], 'accs': []}, 'valid': {'losses': [], 'accs': []}, 'lr': []}
        self.best_acc = 0

    def update_history(self, train_loss, train_acc, valid_loss, valid_acc):
        self.history['train']['losses'].append(train_loss)
        self.history['valid']['losses'].append(valid_loss)
        self.history['train']['accs'].append(train_acc)
        self.history['valid']['accs'].append(valid_acc)

    # def find_lr(self):
    #     lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=self.device)
    #     lr_finder.range_test(self.train_ld, start_lr=1e-10, end_lr=1e-2, num_iter=500, step_mode='linear')
    #     learning_rate = lr_finder.get_optimal()
    #     lr_finder.reset()
    #
    #     print(f'Optimal lr: {learning_rate}')
    #     self.history['lr'].append(learning_rate)
    #     self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=learning_rate)

    def log(self, epoch, epochs, train_loss, valid_loss, train_acc, valid_acc):
        if epoch % self.log_period == 0:
            print(
                "Epoch: {}/{}.. ".format(epoch + 1, epochs),
                "Train Loss: {:.3f}".format(train_loss),
                "Valid Loss: {:.3f}".format(valid_loss),
                "Train Accuracy: {:.3f}".format(train_acc),
                "Valid Accuracy: {:.3f}".format(valid_acc)
            )

    def checkpoint(self, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.model.state_dict(), './checkpoint.pth')
        # else:
            # self.find_lr()
            # self.model.load_state_dict(torch.load('./checkpoint.pth'))

    def train(self, train_ld):
        self.model.train()
        total_loss, total_acc = 0.0, 0.0
        num_batches = len(train_ld)

        for batch in train_ld:
            batch_loss, batch_acc = self.train_batch(batch)
            total_loss += batch_loss
            total_acc += batch_acc

        return total_loss / num_batches, total_acc / num_batches

    def validate(self, valid_ld):
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0
        num_batches = len(valid_ld)

        for batch in valid_ld:
            batch_loss, batch_acc = self.valid_batch(batch)
            total_loss += batch_loss
            total_acc += batch_acc

        return total_loss / num_batches, total_acc / num_batches

    def train_batch(self, batch):
        images = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        batch_loss = loss.item()
        loss.backward()
        self.optimizer.step()
        batch_acc = self.calc_accuracy(outputs, labels)

        return batch_loss, batch_acc

    def valid_batch(self, batch):
        with torch.no_grad():
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            batch_loss = loss.item()
            batch_acc = self.calc_accuracy(outputs, labels)

        return batch_loss, batch_acc

    @staticmethod
    def calc_accuracy(outputs, targets):
        ps = torch.exp(outputs)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == targets.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        return float(accuracy)

    def plot_history(self):
        train_losses = self.history['train']['losses']
        train_accs = self.history['train']['accs']
        valid_losses = self.history['valid']['losses']
        valid_accs = self.history['valid']['accs']

        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b', label='Train')
        plt.plot(epochs, valid_losses, 'g', label='Val')
        plt.title('Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b', label='Train')
        plt.plot(epochs, valid_accs, 'g', label='Val')
        plt.title('Accs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
