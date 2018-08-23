from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch

import pyro
import dmm.polyphonic_data_loader as poly
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


def model(sequences, sequence_lengths, trans_prior, emit_prior):
    assert len(sequences) == len(sequence_lengths)
    trans = pyro.sample("trans", trans_prior)
    emit = pyro.sample("emit", emit_prior)
    tones_iarange = pyro.iarange("tones", sequences.shape[-1], dim=-1)
    with pyro.iarange("sequences", len(sequences), dim=-2):
        x = 0
        for t in range(sequence_lengths.max()):
            with poutine.scale(scale=(sequence_lengths > t).float().unsqueeze(-1)):
                x = pyro.sample("x_{}".format(t), dist.Categorical(trans[x]),
                                infer={"enumerate": "parallel", "expand": False})
                with tones_iarange:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(emit[x]),
                                obs=sequences[:, t])


def main(args):
    logging.info('Loading data')
    data = poly.load_data()
    sequences = torch.tensor(data['train']['sequences'], dtype=torch.float32)
    sequence_lengths = torch.tensor(data['train']['sequence_lengths'], dtype=torch.long)
    data_dim = sequences.shape[-1]

    logging.info('Training')
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)
    trans_prior = dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1).independent(1)
    emit_prior = dist.Dirichlet((4. / 88.) * torch.ones(args.hidden_dim, data_dim)).independent(1)
    guide = AutoDelta(poutine.block(model, expose=["trans", "emit"]))
    optim = Adam({'lr': 0.1})
    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    svi = SVI(model, guide, optim, elbo)

    logging.info('Epoch\tLoss')
    for epoch in range(args.num_epochs):
        loss = svi.step(sequences[:args.num_sequences],
                        sequence_lengths[:args.num_sequences],
                        trans_prior,
                        emit_prior)
        logging.info('{: >5d}\t{}'.format(epoch, loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseball batting average using HMC")
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-s", "--num-sequences", default=10, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    args = parser.parse_args()
    main(args)
