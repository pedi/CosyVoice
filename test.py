import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

print(cosyvoice.sample_rate)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
# prompt_speech_16k = load_wav('./asset/trimmed.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '其实没有所谓的说，跟上潮流还是怎么样。因为，潮流你永远跟不上。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
# speech_chunks = []
# for i, j in enumerate(cosyvoice.inference_zero_shot("""
# 您好，欢迎收听今天的本地新闻播客！在这儿，我们将为您带来一些有趣且重要的本地新闻。这些新闻不仅将在早晨唤醒您的注意力，[laughter]，更会让您了解周围发生的重要事件。<breath>

# 开始我们航行的第一站，是白沙—榜鹅地区。在这里，[breath]一场特别的社区应变能力日活动如火如荼地举行着。超过两千名居民聚集在榜鹅综合社区中心，学习心肺复苏术以及如何使用自动体外除颤器和灭火器。这不是仅仅为了自救，而是为了互助的精神！国务资政张志贤也在现场，他强调社区的韧性就是要通过邻里间的互信建立起来。想象一下，邻里之间互相帮忙买东西或照看孩子，这样的小事构建了我们在危急时刻互助的基石！[laughter]

# 接着，我们来到了兀兰。在这里，居民迎来了967号新环线巴士，这条新线路连接了兀兰、马西岭多个地铁站与医疗设施。它极大地方便了居民们，尤其是年长者的生活。我们当中有位家庭主妇对这个新路线非常满意，她说："这种时候[breath]多一条方便的巴士线真是救星啊！"[laughter] 许多巴士发烧友也前来体验首趟运行，加入到这份兴奋与欢乐之中。

# 我们继续前行，这次来到了社会服务领域。35岁的公务员陈伟烈，凭借他对科技的热爱，开发了可供查询社会援助计划的应用——schemes.sg。通过这一跨界的创新项目，他让社会工作者和公众能够更容易地获取各种援助信息。<strong>这真的为社会服务领域做出了巨大的贡献！</strong> 陈伟烈的努力展示了现代技术如何服务于社会福利。[laughter]

# 最后，我们来到带着新春气息的同安会馆。在这里，迎春展的展出已经迈入了第二年，并且规模比以往更大。今年增加的"福灯迎春暖"特别感人，借助书法家之手，为慈善机构送去满满的祝福。整个活动期间，公众不仅可以欣赏到来自书法家及学生的作品，还有诸多社群活动让我们感受传统文化的魅力和氛围。这种社区活动不仅让我们感受到年节的快乐更是中华文化的独特魅力同时凝聚了社群的力量。[breath]

# 今天的新闻播客就到这里。希望这些有趣的故事和信息可以为您的生活增添色彩！感谢您的收听，我们下次再见！[laughter]
# """, '其实没有所谓的说，跟上潮流还是怎么样。因为，潮流你永远跟不上。', prompt_speech_16k, stream=False)):
#     speech_chunks.append(j['tts_speech'])

# # Concatenate all chunks along time dimension
# full_speech = torch.cat(speech_chunks, dim=1)
# torchaudio.save('instruct_full.wav', full_speech, cosyvoice.sample_rate)