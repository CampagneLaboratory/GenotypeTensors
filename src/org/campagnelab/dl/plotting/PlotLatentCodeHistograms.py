import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy.stats import norm


def plot_histogram(histogram, dimension, epoch, output_basename):
    plot_path = "{}_dimension_{}_epoch_{}.png".format(output_basename, dimension, epoch)
    x = np.linspace(-3, 3, len(histogram))
    norm_pdf = norm.pdf(x, 0, 1)
    plt.plot(x, norm_pdf, x, histogram)
    plt.title("Gaussian pdf vs histogram for dimension {} in epoch {}".format(dimension, epoch))
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        "Plot latent code histograms generated during semisupervised autoencoder training"
    )
    arg_parser.add_argument("-i", "--histogram-basename", type=str, required=True,
                            help="Basename for latent code histogram files.")
    arg_parser.add_argument("-o", "--output-basename", type=str, default="latent_histogram_output",
                            help="Basename for latent code histogram outputs.")
    args = arg_parser.parse_args()

    output_dir = os.path.dirname(args.output_basename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    curr_epoch = 0
    while True:
        histogram_path = "{}_{}.pt".format(args.histogram_basename, curr_epoch)
        if not os.path.exists(histogram_path):
            break
        histogram_list = torch.load(histogram_path)
        for dim_idx, dim_histogram in enumerate(histogram_list):
            plot_histogram(dim_histogram, dim_idx, curr_epoch, args.output_basename)
        curr_epoch += 1
