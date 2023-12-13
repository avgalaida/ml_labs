import torch
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, model, train_ld, val_ld):
        self.device = torch.device('mps')
        self.model = model.to(self.device)
        self.criterion = torch.nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-3)

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
            }
        }

        self.best_acc = 0

        self.log_period = 1

    def start(self, epochs, clear=True):
        if clear:
            self.clear_history()

        for epoch in range(epochs):
            train_loss, train_acc = self.train(self.train_ld)
            valid_loss, valid_acc = self.validate(self.val_ld)

            self.update_history(train_loss, train_acc, valid_loss, valid_acc)
            self.log(epoch, epochs, train_loss, valid_loss, train_acc, valid_acc)
            self.checkpoint(valid_acc)

    def clear_history(self):
        self.history = {'train': {'losses': [], 'accs': []}, 'valid': {'losses': [], 'accs': []}}
        self.best_acc = 0

    def update_history(self, train_loss, train_acc, valid_loss, valid_acc):
        self.history['train']['losses'].append(train_loss)
        self.history['valid']['losses'].append(valid_loss)
        self.history['train']['accs'].append(train_acc)
        self.history['valid']['accs'].append(valid_acc)

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
            torch.save(self.model.state_dict(), './checkpoints/{}.pth'.format(self.model.__class__.__name__))

    def load_checkpoint(self):
        self.model.load_state_dict(torch.load('./checkpoints/{}.pth'.format(self.model.__class__.__name__)))
        return self.model.to('mps')

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
        specs, labels = batch['specgram'].to(self.device), batch['label'].to(self.device).float()

        self.optimizer.zero_grad()
        outputs = self.model(specs)
        loss = self.criterion(outputs, labels)
        batch_loss = loss.item()
        loss.backward()
        self.optimizer.step()
        batch_acc = self.compute_accuracy(outputs, labels)

        return batch_loss, batch_acc

    def valid_batch(self, batch):
        with torch.no_grad():
            specs, labels = batch['specgram'].to(self.device), batch['label'].to(self.device).float()

            outputs = self.model(specs)
            loss = self.criterion(outputs, labels)
            batch_loss = loss.item()
            batch_acc = self.compute_accuracy(outputs, labels)

        return batch_loss, batch_acc

    @staticmethod
    def compute_accuracy(model_output, label):
        predictions = (model_output > 0.5).float()
        correct = (predictions == label).sum().item()
        total = label.size(0)
        return correct / total

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
