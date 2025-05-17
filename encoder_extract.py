import sys, types
import tensorflow as tf

# tf.keras 전체를 fake 'keras' 모듈로 매핑
keras_module = types.ModuleType("keras")
keras_module.__version__ = tf.keras.__version__    # __version__ 정의
keras_module.__dict__.update(tf.keras.__dict__)

# 필요한 서브모듈도 매핑
sys.modules["keras"] = keras_module
sys.modules["keras.layers"] = tf.keras.layers
sys.modules["keras.models"] = tf.keras.models
sys.modules["keras.backend"] = tf.keras.backend

import sys, os
# LipNet 코드가 있는 폴더를 파이썬 경로에 추가
# project root
proj_root = os.path.dirname(__file__)
# the folder that contains the `lipnet` package
repo_root = os.path.join(proj_root, "LipNet")
sys.path.insert(0, repo_root)

from lipnet.model2 import LipNet
from keras.models import Model

# 사전학습 LipNet 로드
lipnet_full = LipNet(
    img_c=3, img_w=100, img_h=50,
    frames_n=75, absolute_max_string_len=32,
    output_size=28
)
lipnet_full.model.load_weights("LipNet/evaluation/models/unseen_weights.h5")

# Encoder(3D-CNN→Bi-GRU)만 분리
encoder = Model(
    inputs=lipnet_full.input_data,
    outputs=lipnet_full.model.get_layer("gru2").output
)

# 확인용 summary
encoder.summary()
