from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="Bo5RYMsuSxkWyFxKP644"
)

result = CLIENT.infer(your_image.jpg, model_id="american-sign-language-letters-gxpdm/4")