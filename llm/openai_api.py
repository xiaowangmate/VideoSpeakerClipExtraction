import ast
import logging
import time

from openai import OpenAI


class GPT:
    def __init__(self, api_key):
        self.api_key = api_key

    def openai_call(self, user_content, system_content=None, model="gpt-4"):
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://gptproxy.llmpaas.tencent.com/v1"
        )
        if system_content is not None and len(system_content.strip()):
            messages = [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content}
            ]
        else:
            messages = [
                {'role': 'user', 'content': user_content}
            ]
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model
            )
            print(chat_completion)
            content = chat_completion.choices[0].message.content
        except Exception as e:
            logging.info(f"Openai model inference error: {str(e)}.")
            time.sleep(10)
            content = self.openai_call(user_content)
        logging.info("Openai model inference done.")
        return content


if __name__ == '__main__':
    import json
    with open("../config/config.json", mode="r", encoding="utf-8") as c:
        config = json.loads(c.read())
        gpt = GPT(config["api_key"]["gpt"])
    from speaker_name_recognition_prompt import speaker_name_recognition_prompt

    # speaker_name = openai_call(speaker_name_recognition_prompt.format("Kelly Clarkson's Mom Is Still Shocked Her Daughter Has A Talk Show  | The Kelly Clarkson Show"))
    speaker_name = gpt.openai_call(
        "说出十个当过世界首富的人以及他们的成为世界首富的时间、行业、学历、父母背景。并告诉我你发现了他们有哪些共同点",
        model="gpt-4")
    print(speaker_name)
