"""全局配置：路径、属性命名映射、教程阈值、随机种子。
Global config: paths, attribute-name maps, tutorial thresholds, random seed.

属性命名陷阱（写代码前必须记住）：
Attribute-name trap (must remember before coding):
- mask 文件名用 milia_like_cyst（单数）
  mask filenames use milia_like_cyst (singular)
- JSON 里用 milia_like_cysts（复数）
  JSON uses milia_like_cysts (plural)
- 其余 4 个属性两边名字一致
  the other 4 attributes share the same name on both sides
"""
import os

PROJ_DIR = r'F:\Desktop\IC\project'
DATA_DIR = os.path.join(PROJ_DIR, 'summer_school_project_train', 'train')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
TASK1_GT_DIR = os.path.join(DATA_DIR, 'task1_gt')
TASK2_GT_DIR = os.path.join(DATA_DIR, 'task2_gt')
EXAMPLE_DIR = os.path.join(PROJ_DIR, 'example_result')

PRETRAINED = {
    'b0': os.path.join(PROJ_DIR, 'pretrained', 'segformer-b0-ade'),
    'b2': os.path.join(PROJ_DIR, 'pretrained', 'segformer-b2-ade'),
    'sam': os.path.join(PROJ_DIR, 'pretrained', 'sam-vit-base'),
}

OUT_DIR = os.path.join(PROJ_DIR, 'outputs')
CACHE_DIR = os.path.join(OUT_DIR, 'preprocessed')  # 预处理缓存（DullRazor+ShadesGray 跑一次存盘）/ preprocess cache

SEED = 42
IMG_SIZE = 512  # SegFormer-B2 训练分辨率；OOM 时降到 384 或换 b0
               # SegFormer-B2 training resolution; lower to 384 or switch to b0 if OOM

# JSON 里的属性顺序（example schema 固定顺序）
# Attribute order in JSON (fixed by the example schema)
ATTRS_JSON = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cysts', 'globules']
# mask 文件名里的属性名（milia 单数）
# Attribute names in mask filenames (milia singular)
ATTRS_FILE = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
ATTR_JSON_TO_FILE = dict(zip(ATTRS_JSON, ATTRS_FILE))
ATTR_FILE_TO_JSON = dict(zip(ATTRS_FILE, ATTRS_JSON))

# tutorial p40 的阈值，直接硬编码，不自调
# Tutorial p40 thresholds, hardcoded as-is, not self-tuned
STATUS_HI = 0.60   # p_attr >= 0.60 -> present
STATUS_LO = 0.40   # p_attr <= 0.40 -> absent；中间 / in between -> uncertain
BORDER_IRREG = 1.60  # border_irregularity >= 1.60 -> irregular
SIZE_SMALL = 0.08    # lesion_area_ratio < 0.08 -> small
SIZE_LARGE = 0.25    # > 0.25 -> large；中间 / in between -> moderate
