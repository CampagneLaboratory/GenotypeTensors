import torch
from torch.autograd import Variable
from torchnet.meter import ConfusionMeter

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.genotype_softmax_classifier import GenotypeSoftmaxClassifer
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar


class Critic(torch.nn.Module):
    """ The critic predicts if inputs are from the training or unlabeled set. """
    def __init__(self, args, input_size):
        super().__init__()
        self.delegate=GenotypeSoftmaxClassifer(num_inputs=input_size, target_size=2,
                                 num_layers=args.num_layers,
                                 reduction_rate=args.reduction_rate,
                                 model_capacity=args.model_capacity,
                                 dropout_p=args.dropout_probability, ngpus=1,
                                 use_selu=args.use_selu,
                                 skip_batch_norm=args.skip_batch_norm)
        self.softmax=torch.nn.LogSoftmax(1)

    def forward(self, input):
        return self.softmax(self.delegate(input))


class TargetEncoder(torch.nn.Module):
    """ The target encoder transforms the inputs form the unlabeled set to make
    them more similar to the inputs from the training set. """

    def __init__(self, args, input_size):
        super().__init__()
        self.delegate = GenotypeSoftmaxClassifer(num_inputs=input_size,
                                                 target_size=input_size,
                                                 num_layers=args.num_layers,
                                                 reduction_rate=0.9,
                                                 model_capacity=args.model_capacity,
                                                 dropout_p=args.dropout_probability,
                                                 ngpus=1,
                                                 use_selu=args.use_selu,
                                                 skip_batch_norm=args.skip_batch_norm)

    def forward(self, input):
        return self.delegate(input)

class ADDA_Model(torch.nn.Module):
    def __init__(self, critic, target_encoder):
        super().__init__()
        self.critic=critic
        self.target_encoder=target_encoder
        self.supervised_model=None

    def install_supervised_model(self,supervised_model):
        """Install a supervised model. After installing a model, a call to forward will use the model with ADDA adapted
        features as input to produce the supervised output. """
        self.supervised_model=supervised_model

    def forward(self, input):
        adapted_features=self.target_encoder(input)
        if self.supervised_model is not None:
            # a supervised model was installed, return the result of supervised classification, with the features
            # adpated with ADDA
            return self.supervised_model(adapted_features)
        else:
            # a supervised model is not installed, return the adapted features.
            return adapted_features

def make_variable(tensor, volatile=False, requires_grad=True):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile,requires_grad=requires_grad)


class GenotypingADDATrainer(CommonTrainer):
    """Train a domain adaptation model with the ADDA aaproach.
    See https://arxiv.org/pdf/1702.05464.pdf and https://github.com/corenel/pytorch-adda
    """

    def __init__(self, args, problem, use_cuda):
        super().__init__(args, problem, use_cuda)
        self.criterion_classifier = torch.nn.CrossEntropyLoss()


        # target encoder:
        self.optimizer_tgt = None
        # critic:
        self.optimizer_critic = None

    def get_test_metric_name(self):
        return "progress"

    def is_better(self, metric, previous_metric):
        return metric < previous_metric

    def create_training_performance_estimators(self):
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("train_critic_loss")]
        performance_estimators += [FloatHelper("train_encoder_loss")]
        performance_estimators += [FloatHelper("train_accuracy")]
        self.training_performance_estimators = performance_estimators
        return performance_estimators

    def create_test_performance_estimators(self):
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("test_critic_loss")]
        performance_estimators += [FloatHelper("test_real_accuracy")]
        performance_estimators += [FloatHelper("test_encoded_accuracy")]
        performance_estimators += [FloatHelper("progress")]
        self.test_performance_estimators = performance_estimators
        return performance_estimators

    def train_adda(self, epoch):
        performance_estimators = self.create_training_performance_estimators()

        print('\nTraining, epoch: %d' % epoch)

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0

        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unlabeled_loader_subset = self.problem.unlabeled_loader()
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset, unlabeled_loader_subset),
            is_cuda=self.use_cuda,
            batch_names=["training", "unlabeled"],
            requires_grad={"training": ["input"], "unlabeled": ["input"]},
            volatile={"training": ["metaData"], "unlabeled": ["metaData"]},
        )

        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s_1 = data_dict["training"]["input"]
                input_u_2 = data_dict["unlabeled"]["input"]

                num_batches += 1
                # allow 10 epochs of pre-training the critic:
                self.do_train_encoder=epoch>1

                self.train_one_batch(performance_estimators, batch_idx, input_s_1, input_u_2)

                if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                    break
        finally:
            data_provider.close()

        return performance_estimators

    def train_one_batch(self, performance_estimators, batch_idx, input_supervised, input_unlabeled):
        self.critic.train()
        self.tgt_encoder.train()
        ###########################
        # 2.1 train discriminator #
        ###########################
        batch_size = input_unlabeled.size(0)
        # zero gradients for optimizer
        self.optimizer_critic.zero_grad()
        self.optimizer_tgt.zero_grad()
        self.tgt_encoder.zero_grad()
        self.critic.zero_grad()
        # extract and concat features
        feat_src = input_supervised
        feat_tgt = input_unlabeled
        feat_concat = torch.cat((feat_src, feat_tgt), 0)

        # predict on discriminator
        pred_concat = self.critic(feat_concat)

        # prepare real and fake label
        source_is_training_set = torch.ones(batch_size).long()
        source_is_unlabeled_set = torch.zeros(batch_size).long()

        label_src = make_variable(source_is_training_set, requires_grad=False)
        label_tgt = make_variable(source_is_unlabeled_set, requires_grad=False)
        label_concat = torch.cat((label_src, label_tgt), 0)

        # compute loss for critic
        loss_critic = self.criterion_classifier(pred_concat, label_concat)
        loss_critic.backward()

        # optimize critic
        self.optimizer_critic.step()

        pred_cls = torch.squeeze(pred_concat.max(1)[1])
        accuracy = (pred_cls == label_concat).float().mean()

        ############################
        # 2.2 train target encoder #
        ############################
        if self.do_train_encoder:
            # train to make unlabeled into training:

            source_is_training_set = source_is_training_set
            self.train_encoder_with(batch_idx, performance_estimators, input_unlabeled, source_is_training_set)
            # train to keep training as training:
            self.train_encoder_with(batch_idx, performance_estimators, input_supervised, source_is_training_set)
        else:
            performance_estimators.set_metric(batch_idx, "train_encoder_loss", -1)
        performance_estimators.set_metric(batch_idx, "train_critic_loss", loss_critic.data[0])
        performance_estimators.set_metric(batch_idx, "train_accuracy",accuracy.data[0])
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         performance_estimators.progress_message(
                             ["train_critic_loss", "train_encoder_loss", "train_accuracy"]))

    def train_encoder_with(self,batch_idx, performance_estimators, features, labels):
        self.tgt_encoder.train()
        self.critic.train()
        # zero gradients for optimizer
        self.optimizer_critic.zero_grad()
        self.optimizer_tgt.zero_grad()
        self.tgt_encoder.zero_grad()
        self.critic.zero_grad()

        # Run the unlabeled examples through the encoder:
        feat_tgt = self.tgt_encoder(features)

        # predict on discriminator
        pred_tgt = self.critic(feat_tgt)

        # prepare fake labels, as if the unlabeled was from the training set:
        label_tgt = make_variable(labels, requires_grad=False)

        # compute loss for target encoder
        loss_tgt = self.criterion_classifier(pred_tgt, label_tgt)
        loss_tgt.backward()

        # Optimize target encoder
        self.optimizer_tgt.step()
        performance_estimators.set_metric(batch_idx, "train_encoder_loss", loss_tgt.data[0])
    def reset_before_test_epoch(self):
        self.cm = ConfusionMeter(self.num_classes, normalized=False)

    def test_one_batch(self, performance_estimators,
                       batch_idx, input_supervised, input_unlabeled):
        self.tgt_encoder.eval()
        self.critic.eval()

        #################################
        #    Evaluate  w/o encoder      #
        #################################
        batch_size = input_supervised.size(0)
        source_is_training_set = torch.ones(batch_size).long()
        source_is_unlabeled_set = torch.zeros(batch_size).long()
        loss_critic, real_accuracy=self.evaluate_test_accuracy(input_supervised,       input_unlabeled,
                                                               source_is_training_set, source_is_unlabeled_set)

        #################################
        #    Evaluate with encoder      #
        #################################
        # extract and target features
        loss_encoded, recoded_accuracy = self.evaluate_test_accuracy(input_supervised,   self.tgt_encoder(input_unlabeled),
                                                                 source_is_training_set, source_is_training_set)

        performance_estimators.set_metric(batch_idx, "test_critic_loss", loss_critic.data[0])
        performance_estimators.set_metric(batch_idx, "test_real_accuracy", real_accuracy.data[0])
        performance_estimators.set_metric(batch_idx, "test_encoded_accuracy", recoded_accuracy.data[0])
        performance_estimators.set_metric(batch_idx, "progress", abs(0.5-recoded_accuracy.data[0]))
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_critic_loss", "test_encoder_loss",
                                                                  "test_real_accuracy","test_encoded_accuracy"]))

    def test_adda(self, epoch):
        print('\nTesting, epoch: %d' % epoch)
        performance_estimators = self.create_test_performance_estimators()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        self.reset_before_test_epoch()
        validation_loader_subset = self.problem.validation_loader_range(0, self.args.num_validation)
        unlabeled_loader_subset = self.problem.unlabeled_loader()
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(validation_loader_subset, unlabeled_loader_subset),
            is_cuda=self.use_cuda,
            batch_names=["validation", "unlabeled"],
            requires_grad={"validation": ["input"], "unlabeled": ["input"]},
            volatile={"validation": ["metaData"],"unlabeled": ["metaData"]},
        )

        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["validation"]["input"]
                input_u = data_dict["unlabeled"]["input"]
                self.net.eval()

                self.test_one_batch(performance_estimators, batch_idx, input_s, input_u)

                if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                    break
            # print()
        finally:
            data_provider.close()

        # Apply learning rate schedule:
        test_metric = performance_estimators.get_metric(self.get_test_metric_name())
        assert test_metric is not None, (self.get_test_metric_name() +
                                         "must be found among estimated performance metrics")
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_metric, epoch)

        self.compute_after_test_epoch()

        return performance_estimators

    def create_ADDA_model(self, problem, args):
        input_size = problem.input_size("input")
        assert len(input_size) == 1, "This ADDA implementation requires 1D input features."

        input_size = problem.input_size("input")[0]
        self.tgt_encoder = TargetEncoder(args=args, input_size=input_size)
        self.critic = Critic(args=args, input_size=input_size)
        model= ADDA_Model(critic=self.critic, target_encoder=self.tgt_encoder)
        beta1 = 0.5
        beta2 = 0.9
        # target encoder:
        self.optimizer_tgt = torch.optim.Adam(self.tgt_encoder.parameters(),
                                              lr=self.args.lr,  # encoder learning rate
                                              betas=(beta1, beta2))
        # critic:
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.args.lr,  # critic learning rate
                                                 betas=(beta1, beta2))

        print("model" + str(model))
        return model

    def evaluate_test_accuracy(self, input_supervised, input_unlabeled, source_is_training_set,
                               source_is_unlabeled_set):
        feat_concat = torch.cat((input_supervised, input_unlabeled), 0)

        # predict on discriminator
        pred_concat = self.critic(feat_concat.detach())
        pred_cls = torch.squeeze(pred_concat.max(1)[1])

        # prepare real and fake label
        label_src = make_variable(source_is_training_set, volatile=True, requires_grad=False)
        label_tgt = make_variable(source_is_unlabeled_set, volatile=True, requires_grad=False)
        label_concat = torch.cat((label_src, label_tgt), 0)
        accuracy = (pred_cls == label_concat).float().mean()
        # compute loss for critic
        loss_critic = self.criterion_classifier(pred_concat, label_concat)
        return loss_critic, accuracy
