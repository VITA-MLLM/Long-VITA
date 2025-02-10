

import logging
logger = logging.getLogger(__name__)


if True:
    IMG_TAG_TOKEN = "<image>"
    VID_TAG_TOKEN = "<video>"
    AUD_TAG_TOKEN = "<audio>"

    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    VID_CONTEXT_TOKEN = '<VID_CONTEXT>'
    VID_START_TOKEN = '<vid>'
    VID_END_TOKEN = '</vid>'

    PATCH_CONTEXT_TOKEN = '<PATCH_CONTEXT>'
    PATCH_START_TOKEN = '<patch>'
    PATCH_END_TOKEN = '</patch>'

    AUD_START_TOKEN = '<|begin_of_audio|>'
    AUD_END_TOKEN = '<|end_of_audio|>'

    QUAD_START_TOKEN = '<quad>'
    QUAD_END_TOKEN = '</quad>'
    REF_START_TOKEN = '<ref>'
    REF_END_TOKEN = '</ref>'
    BOX_START_TOKEN = '<box>'
    BOX_END_TOKEN = '</box>'


if False:
    IMG_TAG_TOKEN = "<|image|>"
    VID_TAG_TOKEN = "<|video|>"
    AUD_TAG_TOKEN = "<|audio|>"

    IMG_CONTEXT_TOKEN = '<|context_of_image|>'
    IMG_START_TOKEN = '<|begin_of_image|>'
    IMG_END_TOKEN = '<|end_of_image|>'

    VID_CONTEXT_TOKEN = '<|context_of_video|>'
    VID_START_TOKEN = '<|begin_of_video|>'
    VID_END_TOKEN = '<|end_of_video|>'

    PATCH_CONTEXT_TOKEN = '<|context_of_patch|>'
    PATCH_START_TOKEN = '<|begin_of_patch|>'
    PATCH_END_TOKEN = '<|end_of_patch|>'

    AUD_START_TOKEN = '<|begin_of_audio|>'
    AUD_END_TOKEN = '<|end_of_audio|>'

    QUAD_START_TOKEN = '<|begin_of_quad|>'
    QUAD_END_TOKEN = '<|end_of_quad|>'
    REF_START_TOKEN = '<|begin_of_ref|>'
    REF_END_TOKEN = '<|end_of_ref|>'
    BOX_START_TOKEN = '<|begin_of_box|>'
    BOX_END_TOKEN = '<|end_of_box|>'

logger.info(f"IMG_TAG_TOKEN {IMG_TAG_TOKEN}")
logger.info(f"VID_TAG_TOKEN {VID_TAG_TOKEN}")
logger.info(f"AUD_TAG_TOKEN {AUD_TAG_TOKEN}")
logger.info(f"IMG_CONTEXT_TOKEN {IMG_CONTEXT_TOKEN}")
logger.info(f"IMG_START_TOKEN {IMG_START_TOKEN}")
logger.info(f"IMG_END_TOKEN {IMG_END_TOKEN}")
logger.info(f"VID_CONTEXT_TOKEN {VID_CONTEXT_TOKEN}")
logger.info(f"VID_START_TOKEN {VID_START_TOKEN}")
logger.info(f"VID_END_TOKEN {VID_END_TOKEN}")
logger.info(f"PATCH_CONTEXT_TOKEN {PATCH_CONTEXT_TOKEN}")
logger.info(f"PATCH_START_TOKEN {PATCH_START_TOKEN}")
logger.info(f"PATCH_END_TOKEN {PATCH_END_TOKEN}")
logger.info(f"AUD_START_TOKEN {AUD_START_TOKEN}")
logger.info(f"AUD_END_TOKEN {AUD_END_TOKEN}")

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
# CLIP_STD = (0.2686295, 0.2613025, 0.2757711)

# SIGLIP_MEAN = (0.5, 0.5, 0.5)
# SIGLIP_STD = (0.5, 0.5, 0.5)


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]



# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = IMG_CONTEXT_TOKEN
DEFAULT_IMAGE_PATCH_TOKEN = PATCH_CONTEXT_TOKEN
DEFAULT_IM_START_TOKEN = IMG_START_TOKEN
DEFAULT_IM_END_TOKEN = IMG_END_TOKEN




