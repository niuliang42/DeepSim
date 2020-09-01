from os.path import join, exists, isdir, abspath
from packaging import version
import torch

ROOT_DIR = abspath('.')

DATA_DIR    = join(ROOT_DIR, 'dataset/')
MODEL_DIR   = join(ROOT_DIR, 'models/')
RESULT_DIR  = join(ROOT_DIR, 'result/')
MID_PRODUCT = join(ROOT_DIR, 'mid_product/')
for d in [DATA_DIR, MODEL_DIR, RESULT_DIR, MID_PRODUCT]:
    assert exists(d) and isdir(d), d+' does not exist. Please run "mkdir '+d+'" or download the dataset.'

######  for preprocessing only ######
GDRIVE_DIR  = '???'
KUNSHAN_1_RAW = join(GDRIVE_DIR, 'Kunshan1')
PARIS_1_RAW = join(GDRIVE_DIR, 'Paris-Le Bourget')
SHENNONGJIA_1_RAW = join(GDRIVE_DIR, 'Shennongjia')
SUZHOU_1_RAW = join(GDRIVE_DIR, 'Suzhoudata1')
SUZHOU_2_RAW = join(GDRIVE_DIR, 'Suzhoudata2')
SUZHOU_3_RAW = join(GDRIVE_DIR, 'Suzhoudata3')
SUZHOU_4_RAW = join(GDRIVE_DIR, 'Suzhoudata4')
SWISS_1_RAW = join(GDRIVE_DIR, 'Swissdata1-Merlischachen')
SWISS_2_RAW = join(GDRIVE_DIR, 'Swissdata2-Renens')
SWISS_3_RAW = join(GDRIVE_DIR, 'Swissdata3-Lausanne')
WEIHAI_1_RAW  = join(GDRIVE_DIR, 'Weihai')
WUXI_1_RAW  = join(GDRIVE_DIR, 'Wuxi')
ENGLAND_RAW_ROOT = join(DATA_DIR, 'England_raw')
ENGLAND_BIRMINGHAM_RAW = join(ENGLAND_RAW_ROOT, 'Birmingham')
ENGLAND_COVENTRY_RAW   = join(ENGLAND_RAW_ROOT, 'Coventry')
ENGLAND_LIVERPOOL_RAW  = join(ENGLAND_RAW_ROOT, 'Liverpool')
ENGLAND_PEAK_RAW       = join(ENGLAND_RAW_ROOT, 'PEAK')
######        end          #####

ENGLAND_DATA    = join(DATA_DIR, 'England')
ENGLAND_960x720 = join(DATA_DIR, 'England_960x720')

SWISS_DATA     = join(DATA_DIR, 'swiss_data/')
SWISS_1280x720  = join(DATA_DIR, 'swiss_1280x720/')
SWISS_960x720  = join(DATA_DIR, 'swiss_960x720/')

SUZHOU_DATA    = join(DATA_DIR, 'suzhou_data/')
SUZHOU_1280x720 = join(DATA_DIR, 'suzhou_1280x720/')
SUZHOU_960x720 = join(DATA_DIR, 'suzhou_960x720/')

FULL_DATA      = join(DATA_DIR, 'full_data/')
FULL_RESIZED   = join(DATA_DIR, 'full_resized/')
MIX_960x720   = join(DATA_DIR, 'scaled_down_dataset/')
FULL_960x720   = join(DATA_DIR, 'full_960x720/')
FULL_AUG_960x720   = join(DATA_DIR, 'full_aug_960x720/')

ERROR_TOLERANCE = join(DATA_DIR, 'error_tolerance/satellitemap')

FULL_RESIZED_FEATURE        = join(MID_PRODUCT, 'features_full_res34_eval.h5')
FULL_RESIZED_FEATURE_NOEVAL = join(MID_PRODUCT, 'features_full_res34_noeval.h5')

FULL_960x720_FEATURE_RES34  = join(MID_PRODUCT, 'features_full_960x720_res34.h5')
FULL_960x720_FEATURE_RES50  = join(MID_PRODUCT, 'features_full_960x720_res50.h5')

_NewResNet = False # Please change this line according to your wish

SQUEEZE_960x720_SHAPE       = (44, 59)
if version.parse(torch.__version__) >= version.parse('1.0') and _NewResNet:
    RESNET_POOLING = ['adaptive', 'fixed'] [0]
    RES34_960x720_SHAPE     = (512,)
    RES34_1280x720_SHAPE    = (512,)
    RES50_960x720_SHAPE     = (2048,)
else:
    RESNET_POOLING = ['adaptive', 'fixed'] [1]
    RES34_960x720_SHAPE     = (512, 17, 24)
    RES34_1280x720_SHAPE    = (512, 17, 34)
    RES50_960x720_SHAPE     = (2048, 17, 24)

