import tensorflow as tf

def compute_feature(block):
    """
    计算特征
    """
    bsz = block.shape[0]
    