import ast
import logging
from openai import OpenAI


def openai_call(user_content, system_content=None, model="gpt-4"):
    client = OpenAI(
        # This is the default and can be omitted
        api_key="xgjxaKzPQzQEEQkIycX6koWRofQ1M2Oc",
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
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    logging.info("Openai model inference done.")
    return chat_completion.choices[0].message.content


if __name__ == '__main__':
    from speaker_name_recognition_prompt import speaker_name_recognition_prompt
    speaker_name = openai_call(speaker_name_recognition_prompt.format("Kelly Clarkson's Mom Is Still Shocked Her Daughter Has A Talk Show  | The Kelly Clarkson Show"))
    print(speaker_name)
