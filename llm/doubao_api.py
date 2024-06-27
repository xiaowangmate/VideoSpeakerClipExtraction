from volcenginesdkarkruntime import Ark


class Doubao:
    def __init__(self, api_key):
        self.client = Ark(api_key=api_key)

    def standard_request(self, messages, model):
        # Non-streaming:
        # print("----- standard request -----")
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        # print(completion)
        content = completion.choices[0].message.content
        print(content)
        return content

    def streaming_request(self, messages, model):
        # Streaming:
        # print("----- streaming request -----")
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        # print(stream)
        for chunk in stream:
            if not chunk.choices:
                continue
            print(chunk.choices[0].delta.content, end="")
        print()
        content = stream.choices[0].message.content
        return content


if __name__ == '__main__':
    pass
