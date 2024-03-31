from src.train import train_model, test_model
from src.metrics import print_loss_report, plot_loss_evaluation


def train(name, arch):
    train_model(
        name = "TreeNet-v1",
        arch_name = "TreeNetV1",
        epochs = 10,
        lr = 1e-4,
        image_crop = (512, 512),
        # lr_schedule = 0.977,
        batch_size = 50,
        description = "LSTM model trained on IMU acceleration data from the thigh and shank."
    )


if __name__ == "__main__":
    model = "TreeNet-v1"
    arch = "TreeNetV1"
    train(model, arch)
    print_loss_report(model)
    # test_model(model)
    # generate_metrics(model)
    