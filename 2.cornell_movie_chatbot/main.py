from train import encoder, decoder
from eval import evaluateInput
from model import Attn, LuongAttnDecoderRNN, EncoderRNN, GreedySearchDecoder
from dataset import voc

# 将dropout layers设置为eval模式
encoder.eval()
decoder.eval()

# 初始化探索模块
searcher = GreedySearchDecoder(encoder, decoder)

# 开始聊天（取消注释并运行以下行开始）
evaluateInput(encoder, decoder, searcher, voc)
