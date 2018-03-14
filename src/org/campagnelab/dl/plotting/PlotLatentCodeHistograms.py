import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy.stats import norm


def plot_histogram(dimension, epoch, output_basename, latent_histogram, gaussian_histogram):
    if len(latent_histogram) != len(gaussian_histogram):
        raise ValueError("Lengths of histograms are unequal")
    plot_path = "{}_dimension_{}_epoch_{}.png".format(output_basename, dimension, epoch)
    x = np.linspace(-3, 3, len(latent_histogram))
    norm_pdf = norm.pdf(x, 0, 1)
    plt.plot(x, norm_pdf, label="Gaussian reference distribution")
    plt.plot(x, latent_histogram, label="Latent code histogram")
    plt.plot(x, gaussian_histogram, label="Gaussian drawn histogram")
    plt.legend(loc="best")
    plt.title("Histograms for dimension {} in epoch {}".format(dimension, epoch))
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
        histogram_dict = torch.load(histogram_path)
        for dim_idx in range(len(histogram_dict["latent"])):
            plot_histogram(dim_idx, curr_epoch, args.output_basename, histogram_dict["latent"][dim_idx],
                           histogram_dict["gaussian"][dim_idx])
        curr_epoch += 1
