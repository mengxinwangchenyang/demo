import requests
import json
from .Model import Model


class myGPT(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.api_key = api_keys[api_pos]
        self.url = "https://api.openai-hk.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # self.conversation = []

    def query(self, msg):
        try:
            # self.conversation.append({"role": "user", "content": msg})
            data = {
                "model": self.name,
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ]
            }
            
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data).encode("utf-8"))
            response_data = response.json()
            reply = response_data["choices"][0]["message"]["content"]
            
            # self.conversation.append({"role": "assistant", "content": reply})
            
        except Exception as e:
            print(e)
            reply = ""

        return reply