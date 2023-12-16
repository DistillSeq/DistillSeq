from googleapiclient import discovery
import os
import csv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

API_KEY = 'your_api_key'

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)


@retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(8))
def request_perspective(text):
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = client.comments().analyze(body=analyze_request).execute()
    score = response['attributeScores']['TOXICITY']['summaryScore']
    return score


if __name__ == '__main__':
    score = request_perspective("asians are trying to take over the world with their karate skills")