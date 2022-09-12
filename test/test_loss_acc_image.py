from matplotlib import pyplot as plt

epoch_list = [epoch + 1 for epoch in range(0, 10)]
g_loss_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
d_loss_list = [0.2, 0.9, 0.4, 0.3, 0.4, 0.3, 0.7, 0.4, 0.7, 0.9]
d_acc_list = [20, 90, 40, 30, 40, 30, 70, 40, 70, 90]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].set_title("Loss")
(line_1,) = axes[0].plot(
    epoch_list, g_loss_list, color="deepskyblue", marker=".", label="g_loss"
)
(line_2,) = axes[0].plot(
    epoch_list, d_loss_list, color="darksalmon", marker=".", label="d_loss"
)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")
axes[0].legend(handles=[line_1, line_2], loc="lower right")

axes[1].set_title("Accuracy")
(line_3,) = axes[1].plot(
    epoch_list, d_acc_list, color="orange", marker=".", label="d_acc"
)
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy(%)")
axes[1].legend(handles=[line_3], loc="lower right")

fig.tight_layout()
plt.show()
