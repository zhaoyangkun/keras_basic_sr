import matplotlib.pyplot as plt


def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        # ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
        # ax.axis("off")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)


plt.figure(figsize=(7, 2), dpi=300)
# 创建 2x2 的子图，并且布局从第 1 行，第 1 列开始，占据 2 行 1 列
ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 4), (0, 1))
ax3 = plt.subplot2grid((2, 4), (0, 2))
ax4 = plt.subplot2grid((2, 4), (0, 3))
ax5 = plt.subplot2grid((2, 4), (1, 1))
ax6 = plt.subplot2grid((2, 4), (1, 2))
ax7 = plt.subplot2grid((2, 4), (1, 3))

plt.suptitle("subplot2grid")
make_ticklabels_invisible(plt.gcf())
plt.show()
