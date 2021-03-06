import argparse
import csv

import os
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Log results from evaluation script")
    parser.add_argument('--checkpoint-key', help='Random key to load a checkpoint model', required=True, type=str)
    parser.add_argument('--model-path', help='Path of where the models directory is located.', required=True, type=str)
    parser.add_argument('--dataset', help='Dataset the model was evaluated on: validation, test or training.', required=True, type=str)
    parser.add_argument('--model-label', help='Model label: best or latest.', default="best", type=str)
    parser.add_argument("--vcf-path", help="Path to base output directory for rtg vcfeval", type=str, required=True)
    parser.add_argument("--output-path", help="Location of log file; if file not present, creates a new CSV file",
                        type=str, required=True)
    args = parser.parse_args()

    print('==> Loading model from checkpoint..')
    assert os.path.isdir('{}/models/'.format(args.model_path)), 'Error: no models directory found!'
    checkpoint = None
    try:
        checkpoint_filename = '{}/models/pytorch_{}_{}.t7'.format(args.model_path, args.checkpoint_key,
                                                                  args.model_label)
        checkpoint = torch.load(checkpoint_filename)
    except FileNotFoundError:
        print("Unable to load model {} from checkpoint".format(args.checkpoint_key))
        exit(1)
    epoch = checkpoint['epoch']

    # Formatted to extract output from RTG vcfeval summary files
    full_summary_path = os.path.join(args.vcf_path, "full", "summary.txt")
    full_exists = os.path.exists(full_summary_path)
    snp_summary_path = os.path.join(args.vcf_path, "snp", "summary.txt")
    indel_summary_path = os.path.join(args.vcf_path, "indel", "summary.txt")
    with open(snp_summary_path, "r") as snp_summary_f, open(indel_summary_path, "r") as indel_summary_f:
        snp_stats = list(map(float, snp_summary_f.readlines()[2].strip().split()))
        indel_stats = list(map(float, indel_summary_f.readlines()[2].strip().split()))
        snp_precision = snp_stats[-3]
        snp_recall = snp_stats[-2]
        snp_f1 = snp_stats[-1]
        indel_precision = indel_stats[-3]
        indel_recall = indel_stats[-2]
        indel_f1 = indel_stats[-1]

    full_precision = None
    full_recall = None
    full_f1 = None
    if full_exists:
        with open(full_summary_path, "r") as full_summary_f:
            full_stats = list(map(float, full_summary_f.readlines()[2].strip().split()))
            full_precision = full_stats[-3]
            full_recall = full_stats[-2]
            full_f1 = full_stats[-1]

    file_exists = os.path.exists(args.output_path)
    fieldnames = ["path", "checkpoint", "label", "dataset", "epoch"]
    if full_exists:
        fieldnames += ["Precision_Full", "Recall_Full", "F1_Full"]
    fieldnames += ["Precision_SNPs", "Recall_SNPs", "F1_SNPs", "Precision_Indels", "Recall_Indels", "F1_Indels"]
    with open(args.output_path, "a") as output_f:
        output_writer = csv.DictWriter(output_f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            output_writer.writeheader()
        output_row = {
            "path": args.model_path,
            "checkpoint": args.checkpoint_key,
            "label": args.model_label,
            "dataset": args.dataset,
            "epoch": epoch,
            "Precision_SNPs": snp_precision,
            "Recall_SNPs": snp_recall,
            "F1_SNPs": snp_f1,
            "Precision_Indels": indel_precision,
            "Recall_Indels": indel_recall,
            "F1_Indels": indel_f1
        }
        if full_exists:
            output_row.update({
                "Precision_Full": full_precision,
                "Recall_Full": full_recall,
                "F1_Full": full_f1,
            })
        output_writer.writerow(output_row)
