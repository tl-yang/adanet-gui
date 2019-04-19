import os
import shutil
from functools import partial

import adanet
import tensorflow as tf

import model
from model import image_preprocessing


def ensemble_architecture(result):
    """Extracts the ensemble architecture from evaluation results."""

    architecture = result["architecture/adanet/ensembles"]
    # The architecture is a serialized Summary proto for TensorBoard.
    summary_proto = tf.summary.Summary.FromString(architecture)
    return summary_proto.value[0].tensor.string_val[0]


class ImageClassificationAdaNet(object):
    _NUM_LAYERS_KEY = "num_layers"
    FEATURES_KEY = "images"
    PreprocessingType = image_preprocessing.PreprocessingType
    Generator = {
        'SimpleCNN': model.SimpleCNNGenerator,
        'SimpleDNN': model.SimpleDNNGenerator,
        'NASNetA': model.NasNetAGenerator,
    }

    def __init__(self, input_data, image_shape, learning_rate=0.01, train_steps=10, batch_size=32,
                 learn_mixture_weights='False', adanet_lambda=0, adanet_iteration=5, num_classes=10,
                 generator='SimpleCNNGenerator'):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = input_data
        self.image_shape = image_shape
        self.label_shape = (self.y_train.shape[1],) if len(self.y_train.shape) > 1 else (1,)
        self.learning_rate = float(learning_rate)
        self.train_steps = int(train_steps)
        self.batch_size = int(batch_size)
        self.learn_mixture_weights = eval(learn_mixture_weights)
        self.adanet_lambda = float(adanet_lambda)
        self.adanet_iterations = int(adanet_iteration)
        if generator not in self.Generator:
            raise RuntimeError('Unrecognized generator: {}'.format(generator))
        self.generator = self.Generator[generator]
        self.num_classes = num_classes
        self.log_dir = '/tmp/models'
        self.estimator = None
        shutil.rmtree(self.log_dir, ignore_errors=True)

    def preprocess_image(self, image, label, partition):
        """Preprocesses an image for an `Estimator`."""
        # First let's scale the pixel values to be between 0 and 1.
        image = tf.image.resize_images(image, self.image_shape[:2])
        image = image / 255.
        # Next we reshape the image so that we can apply a 2D convolution to it.
        # image_height, image_width = self.image_shape[:2]
        image = tf.reshape(image, self.image_shape)
        # image = image_preprocessing.resize_and_normalize(image, image_height, image_width)
        # if partition == 'train':
        #     image = image_preprocessing.basic_augmentation(image, image_height, image_width, self.image_shape[-1])
        # Finally the features need to be supplied as a dictionary.
        features = {self.FEATURES_KEY: image}
        return features, label

    def _input_fn(self, partition, training, batch_size):
        """Generate an input_fn for the Estimator."""

        _input_fn = None

        def _input_fn():
            if partition == "train":
                dataset = tf.data.Dataset.from_generator(
                    self._generator(self.x_train, self.y_train), (tf.float32, tf.int32), (self.image_shape, (1,)))
            elif partition == "predict":
                dataset = tf.data.Dataset.from_generator(
                    self._generator(self.x_test[:10], self.y_test[:10]), (tf.float32, tf.int32),
                    (self.image_shape, (1,)))
            else:
                dataset = tf.data.Dataset.from_generator(
                    self._generator(self.x_test, self.y_test), (tf.float32, tf.int32), (self.image_shape, (1,)))

            # We call repeat after shuffling, rather than before, to prevent separate
            # epochs from blending together.
            if training:
                dataset = dataset.shuffle(10 * batch_size).repeat()
            image_preprocess = partial(self.preprocess_image, partition=partition)
            dataset = dataset.map(image_preprocess).batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels

            _input_fn = _input_fn

        return _input_fn

    def _generator(self, images, labels):
        """Returns a generator that returns image-label pairs."""

        def _gen():
            for image, label in zip(images, labels):
                image = image.reshape(self.image_shape)
                label = label.reshape(self.label_shape)
                yield image, label

        return _gen

    def train_and_evaluate(self, experiment_name):
        """Trains an `adanet.Estimator` to predict housing prices."""

        model_dir = os.path.join(self.log_dir, experiment_name)

        self.estimator = adanet.Estimator(
            # Since we are predicting housing prices, we'll use a regression
            # head that optimizes for MSE.
            head=tf.contrib.estimator.multi_class_head(
                n_classes=self.num_classes,
                loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),

            # Define the generator, which defines our search space of subnetworks
            # to train as candidates to add to the final AdaNet model.
            subnetwork_generator=self.generator(
                learning_rate=self.learning_rate,
                learn_mixture_weights=self.learn_mixture_weights,
                max_iteration_steps=self.train_steps // self.adanet_iterations),

            # Lambda is a the strength of complexity regularization. A larger
            # value will penalize more complex subnetworks.
            adanet_lambda=self.adanet_lambda,

            # The number of train steps per iteration.
            max_iteration_steps=self.train_steps // self.adanet_iterations,

            # The evaluator will evaluate the model on the full training set to
            # compute the overall AdaNet loss (train loss + complexity
            # regularization) to select the best candidate to include in the
            # final AdaNet model.
            evaluator=adanet.Evaluator(
                input_fn=self._input_fn("train", training=False, batch_size=self.batch_size)),

            # Configuration for Estimators.
            config=tf.estimator.RunConfig(
                save_summary_steps=5000,
                save_checkpoints_steps=5000,
                model_dir=model_dir))

        # Train and evaluate using using the tf.estimator tooling.
        train_spec = tf.estimator.TrainSpec(
            input_fn=self._input_fn("train", training=True, batch_size=self.batch_size),
            max_steps=self.train_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=self._input_fn("test", training=False, batch_size=self.batch_size),
            steps=None,
            start_delay_secs=1,
            throttle_secs=1,
        )
        result = tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
        return result

    def predict(self):
        if not self.estimator:
            raise RuntimeError('Not trained yet, predictiong abort')
        return self.estimator.predict(input_fn=self._input_fn("predict", training=False, batch_size=1))

# (x_train, y_train), (x_test, y_test) = (
#     tf.keras.datasets.cifar10.load_data())
# ada = ImageClassificationAdaNet(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, image_shape=(32, 32, 3))
# results, _ = ada.train_and_evaluate("uniform_average_ensemble_baseline")
# pprint(results)
# print("Loss:", results["average_loss"])
# print("Architecture:", ensemble_architecture(results))
