from src.train import train_model, test_model
from src.metrics import print_loss_report, plot_loss_evaluation


def train(name):
    train_model(
        name = name,
        arch_name = "UNet",
        epochs = 150,
        lr = 1e-4,
        image_crop = (236, 236),
        # lr_schedule = 0.977,
        batch_size = 4,
        description = "Trimmed down U-Net with augmentation."
    )


if __name__ == "__main__":
    model = "U-Net-v1"
    train(model)
    # print_loss_report(model)
    # test_model(model)
    # generate_metrics(model)
    