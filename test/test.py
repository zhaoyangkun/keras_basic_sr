import cv2 as cv
from matplotlib import pyplot as plt


def generate_area_sr_img(generator_list, img_path, downsample_mode="bicubic"):
    """
    生成矩形区域的超分图片
    """
    # 读取图片
    img = cv.imread(img_path)
    cv.imshow("original", img)

    # 选择 ROI
    x, y, w, h = cv.selectROI(windowName="original",
                              img=img,
                              showCrosshair=True,
                              fromCenter=False)
    crop = img[y:y + h, x:x + w]

    # 对 ROI 区域进行退化处理
    crop_lr = cv.resize(crop, (crop.shape[1] // 4, crop.shape[0] // 4),
                        interpolation=cv.INTER_CUBIC)

    # 将图片从 BGR 格式转换为 RGB 格式
    crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
    crop_lr = cv.cvtColor(crop_lr, cv.COLOR_BGR2RGB)

    # 生成超分图像
    sr_img_list = []
    for generator in generator_list:
        if generator["resize_factor"] > 1:
            crop_lr = cv.resize(
                crop_lr, (crop_lr.shape[1] * generator["resize_factor"],
                          crop_lr.shape[0] * generator["resize_factor"]),
                interpolation=cv.INTER_CUBIC)
        sr_img = generate_sr_img_2(generator["model"], crop_lr)
        sr_img_list.append(sr_img)

    # 在原图上绘制边框
    left_top = (x, y + h)
    right_bottom = (x + w, y)
    cv.rectangle(img, left_top, right_bottom, (0, 0, 255), 2)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # 处理子图
    def make_ticklabels_invisible(fig, title_list=[]):
        for i, ax in enumerate(fig.axes):
            # ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
            ax.set_title(title_list[i], y=-0.3, fontsize=6)
            ax.axis("off")
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_visible(False)

    fig = plt.figure(figsize=(6, 2), dpi=300)
    ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2)
    ax1.imshow(img)
    ax2 = plt.subplot2grid((2, 4), (0, 1))
    ax2.imshow(crop_lr)
    ax3 = plt.subplot2grid((2, 4), (0, 2))
    ax3.imshow(sr_img_list[0])
    ax4 = plt.subplot2grid((2, 4), (0, 3))
    ax4.imshow(sr_img_list[1])
    ax5 = plt.subplot2grid((2, 4), (1, 1))
    ax5.imshow(sr_img_list[2])
    ax6 = plt.subplot2grid((2, 4), (1, 2))
    ax6.imshow(sr_img_list[3])
    ax7 = plt.subplot2grid((2, 4), (1, 3))
    ax7.imshow(sr_img_list[4])

    make_ticklabels_invisible(
        fig=plt.gcf(),
        title_list=[
            "",
            downsample_mode,
            "SRCNN",
            "VDSR",
            "SRGAN",
            "ESRGAN",
            "HA-ESRGAN",
        ],
    )
    plt.tight_layout()
    fig.savefig(
        "./image/{}_contract.png".format(downsample_mode),
        dpi=300,
        bbox_inches="tight",
    )
    fig.clear()
    plt.close(fig)

    k = cv.waitKey(0)
    if k == 27:  # 按 esc 键即可退出
        cv.destroyAllWindows()