import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.kera import ModelFactory
from sklearn.metrics import roc_auc_score
from utility import get_sample_counts
from augmenter import augmenter


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    data_set_dir = cp["TRAIN"].get("dataset_csv_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    input_weights_name = cp["TRAIN"].get("input_weights_name")
    weights_path = os.path.join(data_set_dir, input_weights_name)
    best_weights_path = os.path.join(data_set_dir, "best_{}".format(input_weights_name))

    # get test sample count
    test_counts, _ = get_sample_counts(data_set_dir, "test", class_names)

    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError("""
                test_steps: {} is invalid,
                please use 'auto' or integer.
                """.format(test_steps))
    print("** test_steps: {} **".format(test_steps))

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)

    print("** load test generator **")
    test_sequence = AugmentedImageSequence(
#         dataset_csv_file=os.path.join(output_dir, "dev.csv"),
        dataset_csv_file=os.path.join(data_set_dir, "test.csv"),
#         dataset_csv_file=os.path.join(data_set_dir, "MIMIC_dataset.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
    )
#     test_sequence.dataset_df.to_csv(os.path.join(output_dir, 'test_data_frame.csv'))

    print("** make prediction **")
    y_hat = model.predict_generator(test_sequence, verbose=1)
    y = test_sequence.get_y_true()
#     np.savetxt(os.path.join(output_dir, 'y_hat_1205_default_weight.txt'), y_hat)
    np.savetxt(os.path.join(output_dir, 'y_0430.txt'), y)

    test_log_path = os.path.join(output_dir, "test.log")
    print("** write log to {} **".format(test_log_path))
    aurocs = []
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
                aurocs.append(score)
            except ValueError:
                score = 0
            f.write("{}: {}\n".format(class_names[i], score))
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write("mean auroc: {}\n".format(mean_auroc))
        print("mean auroc: {}".format(mean_auroc))


if __name__ == "__main__":
    main()