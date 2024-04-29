from src.train import train_model, test_model
from src.metrics import print_loss_report, plot_loss_evaluation, plot_prediction

# Suitable crop values which dont cause integer flooring can be calculated by
# ((( x * 2 + 4) * 2 + 4) * 2 + 4),
# Where x is the dimension (width and height) of the feature maps at the 
# deepest point of the network.
# (((26 * 2 + 4) * 2 + 4) * 2 + 4) = 236
# (((64 * 2 + 4) * 2 + 4) * 2 + 4) = 540

# lr schedules
# lr_schedule = n_root(factor)
# where factor is the value the lr should be scaled with after n epochs
# for example, lr = 0.001 to lr = 0.0001 results in s scale factor of 0.1,
# and if the lr shoudl be scaled by 0.1 after 10 epochs, lr schedule will be: 
# 10_root(0.1) = 0.794
# 20_root(0.1) = 0.891
# 50_root(0.1) = 0.955

def train(name):
    train_model(
        name = name,
        arch_name = "UNetV2",
        epochs = 100,
        lr = 1e-4,
        # image_crop = (540, 540),
        image_crop = (236, 236),
        # lr_schedule = 0.995,
        lr_schedule = False,
        batch_size = 4,
        description = "Trimmed down U-Net with augmentation."
    )



if __name__ == "__main__":
    model = "U-Net-v8"
    train(model)
    plot_loss_evaluation(model)
    plot_prediction(model, 1, save_img = True, seed = 4)
    # print_loss_report(model)
    # test_model(model)
    # generate_metrics(model)
    