from funasr import AutoModel


class VideoASR:
    def __init__(self):
        self.asr_model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
        )

    def speech_recognition(self, clip_path):
        result = self.asr_model.generate(
            clip_path,
            return_spk_res=False,
            is_final=True
        )
        if result:
            result = result[0]["text"]
            print(f"ASR result: {result}")
        else:
            print("no found speak.")
        return result
