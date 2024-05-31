speaker_name_recognition_prompt = """
你是一个speaker_name识别器，输入视频的标题、描述之后，将分析其中最有可能的一个speaker_name，按如下格式输出：Elon Musk。若没有则返回None

视频标题+描述如下：
{}
"""
