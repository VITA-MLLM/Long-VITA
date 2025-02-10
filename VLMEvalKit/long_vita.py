from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp.vlm import encode_image_file_to_base64


class LongVITAWrapper(BaseAPI):

    allowed_types = ['text', 'image', 'video']

    is_api: bool = True

    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(self,
                 model: str,
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.default_params = {
            # 'top_p': 0.6,
            # 'top_k': 2,
            # 'temperature': 0.8,
            # 'repetition_penalty': 1.1,
            # 'best_of': 1,
            # 'do_sample': True,
            # 'stream': False,
            # 'tokens_to_generate': max_tokens,
            'tokens_to_generate': 1024,
        }
        # if key is None:
        #     key = os.environ.get('GLMV_API_KEY', None)
        # assert key is not None, (
        #     'Please set the API Key (obtain it here: '
        #     'https://open.bigmodel.cn/dev/howuse/introduction)'
        # )
        self.key = key
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def image_to_base64(self, image_path):
        import base64
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    def build_msgs(self, msgs_raw, system_prompt=None, dataset=None):
        msgs = cp.deepcopy(msgs_raw)
        image_list = []
        image_path_list = []
        video_path_list = []
        text = ""
        image_count = 1
        for i, msg in enumerate(msgs):
            if msg['type'] == 'text':
                text += msg['value']
            elif msg['type'] == 'image':
                # image_str = encode_image_file_to_base64(msg['value'])
                # image_list.append(image_str)
                image_path_list.append(msg['value'])
                if dataset == "Video-MME":
                    if image_count == 1:
                        text += f"<video>"
                    else:
                        text += f"<video>"
                else:
                    text += f"<image>\n"
                image_count += 1

            elif msg['type'] == 'video':
                video_path_list.append(msg['value'])
                text += f"<video>"
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")

        # VideoMME
        text = text.replace("\nAnswer: ", "\n")

        if dataset == "OCRBench":
            # text += "Offer a very short reply."
            text += "\nAnswer this question using the text in the image directly without any other context."
            # text += "Answer the question using a single word or phrase."
            # text += "Answer this question using the text in the image directly."

        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST', "MMStar"]:
            text = text.replace("Please select the correct answer from the options above.", "").strip() + "\n"
            # text += "Answer with the letter."
            text += "Answer with the option's letter from the given choices directly."
            # text += "Answer with the given options directly."
            # text += "Answer the question by selecting only one option from the given options."

        elif dataset in ['MVBench',]:
            text = text.replace("Only give the best option.Best option:(", "")
            text += "Answer with the letter."
            # text += "Answer with the option's letter from the given choices directly."

        elif dataset in ['MMVet',]:
            pass

        elif dataset in ['MathVista_MINI',]:
            text += "\nAnswer the question using a single word or phrase."

        elif dataset is not None and DATASET_TYPE(dataset) in ['Y/N',]:
            text = text.replace("Answer the question with Yes or No.", "").strip() + "\n"
            text += "Answer yes or no."
            # text += "\nAnswer the question using a single word or phrase.\n [/INST]"
            # text = text.replace("Please answer yes or no.", "").strip() + "\n"

        elif dataset is not None and DATASET_TYPE(dataset) in ['MCQ',]:
            text = text.replace("Please select the correct answer from the options above.", "").strip() + "\n"
            text += "Answer with the letter."
            # text += "Answer with the option's letter from the given choices directly."
            # text += "Answer with the given options directly."
            # text += "Answer the question by selecting only one option from the given options."

        elif dataset is not None and DATASET_TYPE(dataset) in ['VQA',]:
            pass

        elif dataset is not None and DATASET_TYPE(dataset) in ['Video-MCQ',]:
            text += "Offer a very short reply."
            # text += "Please respond with only the letter of the correct answer."
            # text += "Answer the question using few words or phrase."
            # text += "Please provide your answer by stating the letter followed by the full option."

        else:
            text = text.replace("Answer the question using a single word or phrase.", "").strip() + "\n"
            # text += "Offer a very short reply."
            text += "Answer the question using a single word or phrase."
            # text += "Answer the question with a short phrase."
            # text += Answer this question using the text in the image directly without any other context.
            # text += Answer this question using the text in the image directly.

        return text, image_list, image_path_list, video_path_list

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs

        print("-" * 100)
        print("inputs", inputs)

        messages, image_list, image_path_list, video_path_list = self.build_msgs(msgs_raw=inputs, dataset=kwargs.get('dataset', None))
        print("kwargs", kwargs)
        print("messages", messages)

        url = os.environ.get('LongVITA_URL', default='http://127.0.0.1:5001/api')
        max_num_frame = os.environ.get('MAX_NUM_FRAME', default=None)
        if max_num_frame is not None:
            self.default_params["max_num_frame"] = int(max_num_frame)
        headers = {
            'Content-Type': 'application/json',
            # 'Request-Id': 'remote-test',
            # 'Authorization': f'Bearer {self.key}'
        }
        payload = {
            # 'model': self.model,
            'prompts': [messages],
            # 'image_list': image_list,
            'image_path_list': image_path_list if len(image_path_list) > 0 else None,
            'video_path_list': video_path_list if len(video_path_list) > 0 else None,
            **self.default_params,
        }
        response = requests.put(url, headers=headers, data=json.dumps(payload), verify=False)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
            return -1, self.fail_msg, ''
        else:
            answer = response.json()['text'][0]
            print(f"answer {answer}")
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

            return 0, answer, 'Succeeded! '


class LongVITAAPI(LongVITAWrapper):

    def generate(self, message, dataset=None):
        return super(LongVITAAPI, self).generate(message, dataset=dataset)
