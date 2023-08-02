from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
import argparse
#获取选项        
def add_code_generation_args(parser):
    group = parser.add_argument_group(title="CodeGeeX2 DEMO")
    group.add_argument(
        "--model-path",
        type=str,
        default="THUDM/codegeex2-6b",
    )
    group.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
    )
    group.add_argument(
        "--port",
        type=int,
        default=7860,
    )
    group.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    group.add_argument(                      
        "--cpu",
        action="store_true",
    )
    group.add_argument(                      
        "--half",
        action="store_true",
    )
    return parser


app = FastAPI()
def device():
    if not args.cpu:
        if not args.half:
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).cuda()
        else:
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).cuda().half()
    else:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)

    return model

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    top_k = json_post_list.get('top_k')
    response = model.chat(tokenizer,
                                   prompt,
                                   max_length=max_length if max_length else 128,
                                   top_p=top_p if top_p else 0.95,
                                   top_k=top_k if top_k else 0,
                                   temperature=temperature if temperature else 0.2)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)

    return answer


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()
    model = device()
    model.eval()
    uvicorn.run(app, host=args.listen, port=args.port, workers=args.workers)
