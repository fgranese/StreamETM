import torch
import torch.nn.functional as F
from torch import nn


class ETM(nn.Module):
    """
    Embedded Topic Model (ETM)

    Args:
        num_topics (int): Number of topics.
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Size of the hidden layer in the encoder.
        embed_size (int): Size of the word embeddings.
        embeddings (ndarray, optional): Pretrained word embeddings. Default is None.
        train_embeddings (bool, optional): Whether to train word embeddings. Default is False.
        enc_drop (float, optional): Dropout rate for the encoder. Default is 0.2.
        vocab (list, optional): Vocabulary list. Default is None.
        alpha (ndarray, optional): Initial topic embeddings. Default is None.
    """

    def __init__(
            self,
            num_topics,
            vocab_size,
            hidden_size,
            embed_size,
            embeddings=None,
            train_embeddings=False,
            enc_drop=0.2,
            vocab=None,
            alpha=None,
    ):
        super(ETM, self).__init__()

        # Set the number of topics
        self.num_topics = alpha.shape[0] if alpha is not None else num_topics
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.enc_drop = enc_drop
        self.vocab = vocab

        # Define the word embedding matrix rho
        if embeddings is not None:
            # Use provided embeddings
            self.word_embeddings = nn.Parameter(
                torch.from_numpy(embeddings).float(), requires_grad=train_embeddings
            )
        else:
            # Initialize word embeddings randomly
            # self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))
            self.word_embeddings = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(vocab_size, embed_size)
                ),
                requires_grad=train_embeddings,
            )

        # Initialize topic embeddings alpha
        if alpha is not None:
            self.topic_embeddings = nn.Parameter(alpha)
        else:
            # self.topic_embeddings = nn.Parameter(torch.randn((num_topics, self.word_embeddings.shape[1])))
            self.topic_embeddings = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(self.num_topics, self.embed_size)
                )
            )

        # Define the encoder network with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Softplus(),
            nn.Dropout(enc_drop),
        )

        # Define layers for mean and log variance of latent variable z
        self.fc_mu = nn.Linear(hidden_size, self.num_topics)
        self.fc_logvar = nn.Linear(hidden_size, self.num_topics)

        # https://github.com/BobXWu/CFDTM/blob/master/CFDTM/models/networks/Encoder.py
        self.mean_bn = nn.BatchNorm1d(self.num_topics)
        self.mean_bn.weight.requires_grad = True  # Different from original paper
        self.logvar_bn = nn.BatchNorm1d(self.num_topics)
        self.logvar_bn.weight.requires_grad = True
        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = True

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def encode(self, x):
    #     hidden = self.encoder(x)
    #     mu = self.fc_mu(hidden)
    #     logvar = self.fc_logvar(hidden)
    #     return mu, logvar

    def encode(self, x):
        e1 = self.encoder(x)
        mu = self.mean_bn(self.fc_mu(e1))
        logvar = self.logvar_bn(self.fc_logvar(e1))
        return mu, logvar

    def get_theta(self, x, numpy=False):
        x = self.normalize_input(x, norm_type="l1") # scale in the other, not present in original

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # theta = F.softmax(z, dim=-1)
        temperature = 1.  # help sharpen the distribution
        theta = F.softmax(z / temperature, dim=-1)
        if numpy:
            return theta.cpu().detach().numpy()
        elif self.training:
            return theta, mu, logvar
        else:
            return theta

    def get_beta(self, numpy=False):
        beta = F.softmax(
            torch.matmul(self.topic_embeddings, self.word_embeddings.T), dim=1
        )
        if numpy:
            return beta.cpu().detach().numpy()
        return beta

    def normalize_input(self, x, norm_type="l1"):
        if norm_type == "l1":
            return self.l1_normalize(x)
        elif norm_type == "l2":
            return self.l2_normalize(x)
        elif norm_type == "minmax":
            return self.row_min_max_scale(x)
        elif norm_type == "scale":
            return x / x.sum(dim=1, keepdim=True)
        else:
            return x  # No normalization

    def l1_normalize(self, x):
        norm = x.norm(p=1, dim=1, keepdim=True)
        norm_x = x / (norm + 1e-8)
        return norm_x

    def l2_normalize(self, x):
        norm = x.norm(p=2, dim=1, keepdim=True)
        norm_x = x / (norm + 1e-8)
        return norm_x

    def row_min_max_scale(self, x):
        min_vals = x.min(dim=1, keepdim=True).values
        max_vals = x.max(dim=1, keepdim=True).values
        norm_x = (x - min_vals) / (max_vals - min_vals + 1e-8)
        return norm_x

    def get_topic_embeddings(self, numpy=False):
        if numpy:
            return self.topic_embeddings.cpu().detach().numpy()
        return self.topic_embeddings

    def decode(self, theta, beta):
        return F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)

    def forward(self, bows):
        if self.training:
            theta, mu, logvar = self.get_theta(bows)
            beta = self.get_beta()
            preds = self.decode(theta, beta)
            loss, recon_loss, kld_loss = self.loss_function(bows, preds, mu, logvar)
            return loss, recon_loss, kld_loss
        else:
            theta, mu, logvar = self.get_theta(bows)
            beta = self.get_beta()
            preds = self.decode(theta, beta)
            return preds

    def loss_function(self, bows, preds, mu, logvar):
        preds = torch.clamp(preds, min=1e-10)
        # Reconstruction loss
        recon_loss = -(bows * preds.log()).sum(1)

        # KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        loss = recon_loss.mean() + KLD.mean()
        # Mean losses
        return loss, recon_loss.mean(), KLD.mean()

