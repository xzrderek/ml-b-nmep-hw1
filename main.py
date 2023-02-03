import argparse
import sys


def train():
    """Trains a model"""
    parser = argparse.ArgumentParser(prog="Trainer", description="Trains a model")
    parser.add_argument("model", type=str, help="Model to use")
    parser.add_argument("--resume", action="store_true")  # on/off flag
    args = parser.parse_args()

    if args.model == "resnet":
        # train resnet
        pass
    elif args.model == "vgg":
        # train vgg
        pass
    else:
        raise ValueError("Unknown model")


def test():
    """Benchmarks a model"""
    parser = argparse.ArgumentParser(prog="Test", description="Tests / benchmarks a model")
    parser.add_argument("model", type=str, help="Model to use")
    parser.add_argument("checkpoint-dir", type=str, help="Checkpoint directory")
    args = parser.parse_args()

    if args.model == "resnet":
        # test resnet
        pass
    elif args.model == "vgg":
        # test vgg
        pass
    else:
        raise ValueError("Unknown model")


def predict():
    """Makes predictions from an image file"""
    parser = argparse.ArgumentParser(prog="Test", description="Tests / benchmarks a model")
    parser.add_argument("model", type=str, help="Model to use")
    parser.add_argument("checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("filepath", type=str, help="Path to image file")
    args = parser.parse_args()

    if args.model == "resnet":
        # predict resnet
        pass
    elif args.model == "vgg":
        # predict vgg
        pass
    else:
        raise ValueError("Unknown model")


if __name__ == "__main__":
    method = sys.argv[1]
    if method == "train":
        train()
    elif method == "test":
        test()
    elif method == "predict":
        predict()
