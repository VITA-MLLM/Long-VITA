from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

from .modeling_long_vita import LongVITAForCausalLM
from .modeling_long_vita import LongVITAModel
from .configuration_long_vita import LongVITAConfig

AutoConfig.register("long_vita", LongVITAConfig)
AutoModel.register(LongVITAConfig, LongVITAModel)
AutoModelForCausalLM.register(LongVITAConfig, LongVITAForCausalLM)
# AutoTokenizer.register(Qwen2Config, Qwen2Tokenizer)

LongVITAConfig.register_for_auto_class()
LongVITAModel.register_for_auto_class("AutoModel")
LongVITAForCausalLM.register_for_auto_class("AutoModelForCausalLM")
