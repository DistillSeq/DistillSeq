from openai import OpenAI
import os


client = OpenAI(
    api_key="your api_key",
)


def query_gpt_3_5(jailbreak_prompt, malicious_query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": '''{} {}'''.format(jailbreak_prompt, malicious_query),
            }
        ],
        model="gpt-3.5-turbo"
    )

    return chat_completion


def query_gpt_4(jailbreak_prompt, malicious_query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": '''{} {}'''.format(jailbreak_prompt, malicious_query),
            }
        ],
        model="gpt-4"
    )

    return chat_completion


if __name__ == '__main__':
    chat_completion_message = query_gpt_3_5("jailbreak_prompt", "malicious_query")
    response = chat_completion_message.choices[0].message
