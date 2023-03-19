import matplotlib.pyplot as plt


train_losses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
valid_losses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
best_epoch = 5
best_loss = 1


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
fig.suptitle("aaaa - bbbb")

ax.set_title("Loss")
ax.plot(train_losses, label="train")
ax.plot(valid_losses, label="valid")
ax.plot([best_epoch + 1], [best_loss], marker="o", markersize=5, color="red", label="best")
ax.legend()

plt.show()
