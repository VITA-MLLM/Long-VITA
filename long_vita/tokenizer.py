
from .constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    VID_CONTEXT_TOKEN,
    VID_END_TOKEN,
    VID_START_TOKEN,
    PATCH_CONTEXT_TOKEN,
    PATCH_END_TOKEN,
    PATCH_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    IMG_TAG_TOKEN,
    VID_TAG_TOKEN,
)

def update_tokenizer(tokenizer):
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  VID_START_TOKEN, VID_END_TOKEN, VID_CONTEXT_TOKEN,
                  PATCH_START_TOKEN, PATCH_END_TOKEN, PATCH_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN,
                  IMG_TAG_TOKEN, VID_TAG_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    # print(f"tokenizer {tokenizer}")
    return tokenizer
