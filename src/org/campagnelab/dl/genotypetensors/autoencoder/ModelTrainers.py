import argparse
import random
import string

import sys

from org.campagnelab.dl.genotypetensors.autoencoder.adversarial_autoencoder_trainer import AdversarialAutoencoderTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.adversarial_crossencoder_trainer import \
    AdversarialCrossencoderTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.autoencoder import create_autoencoder_model
from org.campagnelab.dl.genotypetensors.autoencoder.genotype_softmax_classifier import \
    create_genotype_funnel_classifier_model
from org.campagnelab.dl.genotypetensors.autoencoder.genotyping_adda_trainer import GenotypingADDATrainer
from org.campagnelab.dl.genotypetensors.autoencoder.genotyping_semisup_trainer import GenotypingSemiSupTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.genotyping_semisupervised_mixup_trainer import \
    GenotypingSemisupervisedMixupTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.genotyping_supervised_mixup_trainer import \
    GenotypingSupervisedMixupTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.genotyping_supervised_trainer import GenotypingSupervisedTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.genotyping_trainer import GenotypingAutoEncoderTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.sbi_classifier import create_classifier_model
from org.campagnelab.dl.genotypetensors.autoencoder.semisup_adversarial_autoencoder import \
    create_semisup_adv_autoencoder_model
from org.campagnelab.dl.genotypetensors.autoencoder.somatic_trainer import SomaticTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import StructGenotypingModel, \
    StructGenotypingSupervisedTrainer


def define_train_auto_encoder_parser():
    parser = argparse.ArgumentParser(description='Train an auto-encoder for .vec files.')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate.')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint.')

    parser.add_argument('--constant-learning-rates', action='store_true',
                        help='Use constant learning rates, not schedules.')
    parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=128)
    parser.add_argument('--encoded-size', type=int,
                        help='Size the auto-encoder compresses the input to. The number of floats used '
                             'to represent the encoded data.', default=32)
    parser.add_argument('--num-epochs', '--max-epochs', type=int,
                        help='Number of epochs to run before stopping. Additional epochs when --resume.', default=200)
    parser.add_argument('--num-training', '-n', type=int, help='Maximum number of training examples to use.',
                        default=sys.maxsize)
    parser.add_argument('--num-validation', '-x', type=int, help='Maximum number of training examples to use.',
                        default=sys.maxsize)
    parser.add_argument('--num-unlabeled', '-u', type=int, help='Maximum number of unlabeled examples to use.',
                        default=sys.maxsize)
    parser.add_argument('--num-layers', type=int, help='Number of layers in the classifier.', default=3)
    parser.add_argument('--indel-weight-factor', type=float,
                        help='Weight multiplying factor (with respect to SNPs) to increase loss contribution'
                             ' for indels.',
                        default=10.0)
    parser.add_argument('--max-examples-per-epoch', type=int, help='Maximum number of examples scanned in an epoch'
                                                                   '(e.g., for ureg model training). By default, equal '
                                                                   'to the number of examples in the training set.',
                        default=None)
    parser.add_argument('--momentum', type=float, help='Momentum for SGD.', default=0.9)
    parser.add_argument('--gamma', type=float, help='Float used to control the effect of reconstruction loss.',
                        default=1.0)
    parser.add_argument('--L2', type=float, help='L2 regularization.', default=1E-4)
    parser.add_argument('--dropout-probability', type=float, help='Probability of droping activations (drop-out P).',
                        default=0)
    parser.add_argument('--seed', type=int,
                        help='Random seed', default=random.randint(0, sys.maxsize))
    parser.add_argument('--checkpoint-key', help='random key to save/load checkpoint',
                        default=''.join(random.choices(string.ascii_uppercase, k=5)))
    parser.add_argument('--lr-patience', default=10, type=int,
                        help='number of epochs to wait before applying LR schedule when loss does not improve.')
    parser.add_argument('--model', default="PreActResNet18", type=str,
                        help='The model to instantiate. One of VGG16, ResNet18, ResNet50, ResNet101,ResNeXt29, '
                             'ResNeXt29, DenseNet121, PreActResNet18, DPN92')
    parser.add_argument('--problem', default="genotyping:basename", type=str,
                        help='The genotyping problem dataset name. basename is used to locate file named '
                             'basename-train.vec, basename-validation.vec, basename-unlabeled.vec')
    parser.add_argument('--mode', help='Training mode: autoencoder, supervised_somatic',
                        default="supervised",
                        choices=["autoencoder", "supervised_somatic", "semisupervised_genotypes",
                                 "supervised_genotypes", "semisupervised_autoencoder", "supervised_crossencoder",
                                 "supervised_mixup_genotypes", "semisupervised_mixup_genotypes",
                                 "supervised_funnel_genotypes", "supervised_mixup_funnel_genotypes",
                                 "semisupervised_mixup_funnel_genotypes", "ADDA","struct_genotyping"])
    parser.add_argument('--epoch-min-accuracy', default=0, type=float,
                        help='Stop training early if test accuracy is below this value after an epoch of training.'
                             ' This option helps prune poor hyperparameter values. You could set the value to 10 to '
                             'stop if the model did not reach 10% accuracy in the first epoch. ')

    parser.add_argument("--reset-lr-every-n-epochs", type=int,
                        help='Reset learning rate to initial value every n epochs.')

    parser.add_argument("--num-gpus", type=int, default=1, help='Number of GPUs to run on.')
    parser.add_argument("--num-workers", type=int, default=0, help='Number of workers to feed data to the GPUs.')
    parser.add_argument("--autoencoder-type", type=int, default=1, help='Choose a variant of auto-encoder. Type 1 or 2 '
                                                                        'are currently available.',
                        choices=[1, 2])

    parser.add_argument('--abort-when-failed-to-improve', default=sys.maxsize, type=int,
                        help='Abort training if performance fails to improve for more than the specified number of '
                             'epochs.')
    parser.add_argument('--test-every-n-epochs', type=int,
                        help='Estimate performance on the test set every n epochs. '
                             'Note that when test is skipped, the previous test '
                             'performances are reported in the log until new ones are available.'
                             'This parameter does not affect testing for the last 10 epochs of a run, each test is '
                             'performed for these epochs.', default=1)
    parser.add_argument('--n-dim', type=int, default=500,
                        help='Number of units in hidden layers for semisupervised adversarial autoencoders')
    parser.add_argument('--epsilon-label-smoothing', type=float, default=0.0,
                        help='Epsilon value to use for label smoothing.')

    parser.add_argument('--normalize', action='store_true', help='Normalize input by mean and standard deviation.')
    parser.add_argument('--no-progress', action='store_true', help='Disable the progress bar.')
    parser.add_argument('--use-selu', action='store_true', help='Use SELU non-linearity, otherwise, use RELU.')
    parser.add_argument('--reweight-by-validation-error', action='store_true',
                        help='Use validation errors to focus reweight loss in the next training epoch.')
    parser.add_argument('--use-density-weights', action='store_true',
                        help='Weight loss by the abundance of each minibatch example in the unlabled set.')
    parser.add_argument("--latent-code-output", type=str, help="Basename of file to save latent code histograms in")
    parser.add_argument("--latent-code-n-per-minibatch", type=int, default=10,
                        help="Number of examples to save latent codes for")
    parser.add_argument("--latent-code-bins", type=int, default=100,
                        help="Number of bins in histogram for latent code distributions")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="Alpha for supervised genotype with mixup")
    parser.add_argument("--reduction-rate", type=float, default=0.8,
                        help="The amount of reduction in hidden nodes to apply per layer")
    parser.add_argument("--model-capacity", type=float, default=1.0,
                        help="A floating number that controls model capacity (i.e., number of hidden nodes in the "
                             "neural network). Use a c >=1 to control how many hidden nodes are created "
                             "(#hiddenNodes=c*#inputs).")

    parser.add_argument('--label-strategy', help='Strategy to dream up labels for the unsupervised set (mixup mode). ',
                        choices=["UNIFORM", "SAMPLING", "VAL_CONFUSION", "VAL_CONFUSION_SAMPLING"],
                        default="SAMPLING")
    parser.add_argument("--skip-batch-norm", action="store_true",
                        help="If set, don't add batch normalization for softmax supervised genotype classifier")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "adagrad"], default="SGD",
                        help="Optimizer to use. Only supervised_genotypes_softmax supports adagrad currently.")
    parser.add_argument('--adda-source-model', help='Checkpoint key for the model to adapt.',
                        default=None)
    parser.add_argument("--no-cuda", action="store_true",help="Do not use CUDA.")
    parser.add_argument("--use-batching", action="store_true",help="Use manual batching when mapping sbi instances to tensors.")
    parser.add_argument("--adda-pass-through", action="store_true",
                        help="If set, train the ADDA encoder to pass-through examples from the training set as unperturbed as possible.")
    parser.add_argument('--num-estimate-class-frequencies', type=int, help='Number of examples to look at to estimate '
                                                                           'class frequencies.', default=100000)
    parser.add_argument('--struct-count-dim', type=int, default=64, help="Dimensionality of the reduced count tensor (used with struct_genotyping only).")
    parser.add_argument('--struct-sample-dim', type=int, default=64, help="Dimensionality of the reduced sample tensor (used with struct_genotyping only).")
    return parser

def configure_model_trainer(train_args, train_problem,train_use_cuda,class_frequencies=None):
    args=train_args
    if train_args.mode == "struct_genotyping":
        model_trainer = StructGenotypingSupervisedTrainer(args=train_args, problem=train_problem,
                                                     use_cuda=train_use_cuda)
        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: model_trainer.create_struct_model(
                problem=train_problem, args=train_args)))

        training_loop_method = model_trainer.train_supervised
        testing_loop_method = model_trainer.test_supervised

    elif train_args.mode == "ADDA":
        model_trainer = GenotypingADDATrainer(args=train_args, problem=train_problem,
                                                     use_cuda=train_use_cuda)
        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: model_trainer.create_ADDA_model(
                problem=train_problem, args=train_args)))

        training_loop_method = model_trainer.train_adda
        testing_loop_method = model_trainer.test_adda

    elif train_args.mode == "autoencoder":
        model_trainer = GenotypingAutoEncoderTrainer(args=train_args, problem=train_problem,
                                                     use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_autoencoder_model(
                model_name,
                problem_type,
                encoded_size=train_args.encoded_size,
                ngpus=train_args.num_gpus,
                dropout_p=train_args.dropout_probability,
                autoencoder_type=train_args.autoencoder_type,
                use_selu=args.use_selu
            )),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_autoencoder
        testing_loop_method = model_trainer.test_autoencoder

    elif train_args.mode == "supervised_somatic":
        model_trainer = SomaticTrainer(args=train_args, problem=train_problem,
                                       use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_classifier_model(
                model_name,
                problem_type,
                encoded_size=train_args.encoded_size,
                ngpus=train_args.num_gpus,
                num_layers=train_args.num_layers,
                autoencoder_type=train_args.autoencoder_type,
                use_selu=train_args.use_selu
            )),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.supervised_somatic
        testing_loop_method = model_trainer.test_somatic_classifer

    elif train_args.mode == "semisupervised_genotypes":
        model_trainer = GenotypingSemiSupTrainer(args=train_args, problem=train_problem,
                                                 use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_classifier_model(
                model_name,
                problem_type,
                encoded_size=train_args.encoded_size,
                somatic=False,
                ngpus=train_args.num_gpus,
                dropout_p=train_args.dropout_probability,
                num_layers=train_args.num_layers,
                autoencoder_type=train_args.autoencoder_type,
                use_selu=args.use_selu
            )),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_semisup
        testing_loop_method = model_trainer.test_semi_sup

    elif train_args.mode == "supervised_genotypes":
        model_trainer = GenotypingSupervisedTrainer(args=train_args, problem=train_problem,
                                                    use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_classifier_model(
                model_name,
                problem_type,
                encoded_size=train_args.encoded_size,
                somatic=False,
                ngpus=train_args.num_gpus,
                dropout_p=train_args.dropout_probability,
                num_layers=train_args.num_layers,
                autoencoder_type=train_args.autoencoder_type,
                drop_decoder=True,
                prenormalized_inputs=args.normalize,
                use_selu=args.use_selu,
            )),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_supervised
        testing_loop_method = model_trainer.test_supervised

    elif train_args.mode == "semisupervised_autoencoder":
        model_trainer = AdversarialAutoencoderTrainer(args=train_args, problem=train_problem,
                                                      use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_semisup_adv_autoencoder_model(
                model_name,
                problem_type,
                encoded_size=train_args.encoded_size,
                dropout_p=train_args.dropout_probability,
                num_hidden_layers=train_args.num_layers,
                n_dim=train_args.n_dim,
                prenormalized_inputs=args.normalize,
                use_selu=args.use_selu
            )),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_semisup_aae
        testing_loop_method = model_trainer.test_semisup_aae

    elif train_args.mode == "supervised_crossencoder":
        model_trainer = AdversarialCrossencoderTrainer(args=train_args, problem=train_problem,
                                                       use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_semisup_adv_autoencoder_model(
                model_name,
                problem_type,
                encoded_size=train_args.encoded_size,
                dropout_p=train_args.dropout_probability,
                num_hidden_layers=train_args.num_layers,
                n_dim=train_args.n_dim,
                prenormalized_inputs=args.normalize,
                use_selu=args.use_selu
            )),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_semisup_aae
        testing_loop_method = model_trainer.test_semisup_aae

    elif train_args.mode == "supervised_mixup_genotypes":
        model_trainer = GenotypingSupervisedMixupTrainer(args=train_args, problem=train_problem,
                                                         use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_classifier_model(
                model_name,
                problem_type,
                encoded_size=train_args.encoded_size,
                somatic=False,
                ngpus=train_args.num_gpus,
                dropout_p=train_args.dropout_probability,
                num_layers=train_args.num_layers,
                autoencoder_type=train_args.autoencoder_type,
                drop_decoder=True,
                prenormalized_inputs=args.normalize,
                use_selu=args.use_selu
            )),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_supervised_mixup
        testing_loop_method = model_trainer.test_supervised_mixup

    elif train_args.mode == "semisupervised_mixup_genotypes":
        model_trainer = GenotypingSemisupervisedMixupTrainer(args=train_args, problem=train_problem,
                                                             use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_classifier_model(
                model_name,
                problem_type,
                encoded_size=train_args.encoded_size,
                somatic=False,
                ngpus=train_args.num_gpus,
                dropout_p=train_args.dropout_probability,
                num_layers=train_args.num_layers,
                autoencoder_type=train_args.autoencoder_type,
                drop_decoder=True,
                prenormalized_inputs=args.normalize,
                use_selu=args.use_selu
            )
        ),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_semisupervised_mixup
        testing_loop_method = model_trainer.test_semisupervised_mixup

    elif train_args.mode == "supervised_funnel_genotypes":
        model_trainer = GenotypingSupervisedTrainer(args=train_args, problem=train_problem,
                                                    use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_genotype_funnel_classifier_model(
                model_name,
                problem_type,
                ngpus=train_args.num_gpus,
                num_layers=train_args.num_layers,
                reduction_rate=train_args.reduction_rate,
                model_capacity=train_args.model_capacity,
                dropout_p=train_args.dropout_probability,
                use_selu=args.use_selu,
                skip_batch_norm=args.skip_batch_norm,
            )
        ),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_supervised
        testing_loop_method = model_trainer.test_supervised
    elif train_args.mode == "supervised_mixup_funnel_genotypes":
        model_trainer = GenotypingSupervisedMixupTrainer(args=train_args, problem=train_problem,
                                                         use_cuda=train_use_cuda)
        if class_frequencies is not None:
            model_trainer.class_frequencies=class_frequencies
        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_genotype_funnel_classifier_model(
                model_name,
                problem_type,
                ngpus=train_args.num_gpus,
                num_layers=train_args.num_layers,
                reduction_rate=train_args.reduction_rate,
                model_capacity=train_args.model_capacity,
                dropout_p=train_args.dropout_probability,
                use_selu=args.use_selu,
                skip_batch_norm=args.skip_batch_norm,
            )
        ), class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_supervised_mixup
        testing_loop_method = model_trainer.test_supervised_mixup
    elif train_args.mode == "semisupervised_mixup_funnel_genotypes":
        model_trainer = GenotypingSemisupervisedMixupTrainer(args=train_args, problem=train_problem,
                                                             use_cuda=train_use_cuda)

        model_trainer.init_model(create_model_function=(
            lambda model_name, problem_type: create_genotype_funnel_classifier_model(
                model_name,
                problem_type,
                ngpus=train_args.num_gpus,
                num_layers=train_args.num_layers,
                reduction_rate=train_args.reduction_rate,
                model_capacity=train_args.model_capacity,
                dropout_p=train_args.dropout_probability,
                use_selu=args.use_selu,
                skip_batch_norm=args.skip_batch_norm,
            )
        ),class_frequencies=class_frequencies)
        training_loop_method = model_trainer.train_semisupervised_mixup
        testing_loop_method = model_trainer.test_semisupervised_mixup
    else:
        model_trainer = None
        print("unknown mode specified: " + train_args.mode)
        exit(1)

    return model_trainer, training_loop_method,testing_loop_method
