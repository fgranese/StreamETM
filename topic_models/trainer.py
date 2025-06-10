import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from logging import getLogger

logger = getLogger(__name__)


class BasicTrainer:
    def __init__(self,
                 model,
                 dataset,
                 num_top_words=15,
                 epochs=200,
                 learning_rate=0.1,
                 batch_size=1000,
                 weight_decay=0.0,
                 use_lr_scheduler=None,
                 lr_step_size=125,
                 log_interval=50,
                 device='cpu',
                 verbose=False):

        self.model = model
        self.dataset = dataset
        self.num_top_words = num_top_words
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.device = device
        self.verbose = verbose

        if verbose:
            logger.setLevel("DEBUG")
        else:
            logger.setLevel("WARNING")

    def create_optimizer(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def create_lr_scheduler(self,
                            optimizer):
        # lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.lr_step_size)
        return lr_scheduler

    def get_batch(self,
                  doc_term_matrix,
                  indices):
        data_batch = doc_term_matrix[indices, :]
        data_batch = torch.from_numpy(data_batch.toarray()).float().to(self.device)
        return data_batch

    def train(self,
              refine=False,
              verbose=True):

        if refine:
            self.model.topic_embeddings.requires_grad = False
            self.model.word_embeddings.requires_grad = False

        optimizer = self.create_optimizer()

        lr_scheduler = self.create_lr_scheduler(optimizer) if self.use_lr_scheduler else None

        self.model.to(self.device)

        dataloader = DataLoader(
            TensorDataset(torch.tensor(self.dataset.toarray(), dtype=torch.float32, device=self.device)),
            batch_size=self.batch_size, shuffle=True, )

        for epoch in range(1, self.epochs + 1):

            self.model.train()

            for idx, data_batch in enumerate(dataloader):
                optimizer.zero_grad()

                data_batch = data_batch[0]
                loss, recon_loss, kl_loss = self.model.forward(data_batch)

                loss.backward()

                optimizer.step()

                if lr_scheduler:
                    lr_scheduler.step()

                if verbose and epoch % self.log_interval == 0:
                    logger.info(
                        f"Epoch: {epoch} .. LR: {optimizer.param_groups[0]['lr']} .. "
                        f"KL: {kl_loss.item():.3f} .. recon: {recon_loss.item():.3f} .. Total_loss: {loss.item():.3f}"
                    )

        if refine:
            self.model.topic_embeddings.requires_grad = True
            self.model.word_embeddings.requires_grad = True  ## TODO: Check if this is necessary

        return self.model

    def predict(self,
                chunk):
        self.model.eval()
        data_batch = self.get_batch(chunk, torch.arange(chunk.shape[0]))  # TODO: Legacy, use Dataloader instead
        theta = self.model.get_theta(data_batch, numpy=True)
        beta = self.model.get_beta(numpy=True)
        return theta, beta
