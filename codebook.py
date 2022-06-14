import torch
import torch.nn as nn

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        # Embedding Matrix
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)



    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        # latent vectors
        z_flattened = z.view(-1, self.latent_dim)

        # expanded version of L2 Loss
        # (a - b)^2 = a^2 -2ab +b^2
        # this is the distance between all of the latent and embedding vectors
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t())) + \
            torch.sum(self.embedding.weight**2, dim=1)

        # find which ones are the closest
        # get indexes of embedding and latent vectors which are closest
        min_encoding_indices = torch.argmin(d, dim=1)
        # codebook vectors
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # get codebook loss with stop gradient/detach function
        # remove quantized latent vectors from the gradient flow and then subtracting it from the original latent vectors
        # 11:44, you should read your paper
        # preserving gradients for backward flow?
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3 , 1, 2)

        return z_q, min_encoding_indices, loss