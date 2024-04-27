from src.train import train_model, test_model
from src.metrics import print_loss_report, plot_loss_evaluation, plot_prediction

# Suitable crop values which dont cause integer flooring can be calculated by
# ((( x * 2 + 4) * 2 + 4) * 2 + 4),
# Where x is the dimension (width and height) of the feature maps at the 
# deepest point of the network.
# (((26 * 2 + 4) * 2 + 4) * 2 + 4) = 236
# (((64 * 2 + 4) * 2 + 4) * 2 + 4) = 540


def train(name):
    train_model(
        name = name,
        arch_name = "UNetV2",
        epochs = 100,
        lr = 1e-3,
        image_crop = (540, 540),
        # image_crop = (236, 236),
        lr_schedule = 0.912,
        batch_size = 4,
        description = "Trimmed down U-Net with augmentation."
    )



if __name__ == "__main__":
    model = "U-Net-v8"
    train(model)
    plot_loss_evaluation(model)
    plot_prediction(model)
    # print_loss_report(model)
    # test_model(model)
    # generate_metrics(model)
    