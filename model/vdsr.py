from tensorflow.keras.layers import Add, Conv2D, Input
from tensorflow.keras.models import Model

from model.srcnn import SRCNN


class VDSR(SRCNN):
    """
    VDSR 模型类
    """

    def __init__(
        self,
        model_name,
        result_path,
        train_resource_path,
        test_resource_path,
        epochs,
        init_epoch=1,
        batch_size=4,
        downsample_mode="bicubic",
        scale_factor=4,
        train_hr_img_height=128,
        train_hr_img_width=128,
        valid_hr_img_height=128,
        valid_hr_img_width=128,
        rdb_num=16,
        max_workers=4,
        data_enhancement_factor=1,
        log_interval=20,
        save_images_interval=10,
        save_models_interval=50,
        save_history_interval=10,
        pretrain_model_path="",
        use_mixed_float=False,
        use_sn=False,
        use_ema=False,
    ):
        super().__init__(
            model_name,
            result_path,
            train_resource_path,
            test_resource_path,
            epochs,
            init_epoch,
            batch_size,
            downsample_mode,
            scale_factor,
            train_hr_img_height,
            train_hr_img_width,
            valid_hr_img_height,
            valid_hr_img_width,
            rdb_num,
            max_workers,
            data_enhancement_factor,
            log_interval,
            save_images_interval,
            save_models_interval,
            save_history_interval,
            pretrain_model_path,
            use_mixed_float,
            use_sn,
            use_ema,
        )

    def build_generator(self):
        """
        构建生成器
        """
        inputs = Input(shape=[None, None, 3])

        x = Conv2D(64, 3, padding="same", activation="relu")(inputs)

        for _ in range(18):
            x = Conv2D(64, 3, padding="same", activation="relu")(x)

        x = Conv2D(3, 3, padding="same")(x)

        outputs = Add(dtype="float32")([inputs, x])

        return Model(inputs=inputs, outputs=outputs)
