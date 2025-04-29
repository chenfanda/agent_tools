import os
DEVICE_ID = os.environ['CUDA_ID']
os.environ["CUDA_VISIBLE_DEVICES"]=DEVICE_ID
os.environ["CUDA_LAUNCH_BLOCKING"]=DEVICE_ID
from fastapi import FastAPI,Request 
import uvicorn ,json ,datetime
import torch
import gc
from FlagEmbedding import FlagReranker,FlagLLMreranker,LayerWiseFlagLLMReranker

reranker = "reranker"

def torch_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def decribe_model(x):
    match x:
        case FlagReranker():
            return 1
        case FlagLLMReranker():
            return 2
        case LayerWiseFlagLLMReranker():
            return 3
        case _:
            return 0

app = FastAPI


@app.post("/reranker/prediction")
async def create_item(request:Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    sentence_pairs = json_post_list.get('sentence_pairs')
    model = json_post_list.get('model')

    batch_size = json_post_list.get('batch_size')
    max_length = json_post_list.get('max_length')
    normalize = json_post_list.get('normalize')
    cutoff_layers = json_post_list.get('cutoff_layers')
    use_fp16 = json_post_list.get('use_fp16')

    global reranker

    if batch_size:
        pass
    else:
        batch_size = 256

    if max_length:
        pass
    else:
        max_length = 512
    if normalize:
        normalize =True
    else:
        normalize =False

    if cutoff_layers:
        pass
    else:
        cutoff_layers = [28]
    if use_fp16:
        use_fp16 = True
    else:
        use_fp16 = False
    model_type = decribe_model(reranker)

    if model=="m3":
        if model_type==1:
            inputs_dict={
                'sentence_pairs':sentence_pairs,
                'batch_size':batch_size,
                'max_length':max_length,
                'normalize':normalize,
            }
        else:
            del reranker
            gc.collect()
            torch.cuda.empty.cache()
            torch.cuda.ipc_collect()
            reranker = FlagReranker('/home/app/model/bge-reranker-v2-m3/',use_fp16 = use_fp16)
            
            inputs_dict={
                'sentence_pairs':sentence_pairs,
                'batch_size':batch_size,
                'max_length':max_length,
                'normalize':normalize,
            }
    elif model=="gemma":
        if model_type==2:
            inputs_dict={
                'sentence_pairs':sentence_pairs,
                'batch_size':batch_size,
                'max_length':max_length,
                'normalize':normalize,
            }
        else:
            del reranker
            gc.collect()
            torch.cuda.empty.cache()
            torch.cuda.ipc_collect()
            reranker = FlagLLMReranker('/home/app/model/bge-reranker-v2-gemma/',use_fp16 = use_fp16)
            
            inputs_dict={
                'sentence_pairs':sentence_pairs,
                'batch_size':batch_size,
                'max_length':max_length,
                'normalize':normalize,
            }
    else:
         if model_type==3:
            inputs_dict={
                'sentence_pairs':sentence_pairs,
                'batch_size':batch_size,
                'max_length':max_length,
                'normalize':normalize,
            }
        else:
            del reranker
            gc.collect()
            torch.cuda.empty.cache()
            torch.cuda.ipc_collect()
            reranker = LayerWiseFlagLLMReranker('/home/app/model/bge-reranker-v2-minicpm-layerwise/',use_fp16 = use_fp16)
            
            inputs_dict={
                'sentence_pairs':sentence_pairs,
                'batch_size':batch_size,
                'max_length':max_length,
                'normalize':normalize,
            }
    try:
        score = reranker.compute_score(**inputs_dict)
    except:
        torch_gc()
        return None
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    if model=="cpm":
        answer = {
            "score":score[0],
            "model":model,
            "status":200,
            "time":time
        }
    else:
        answer = {
            "score": score,
            "model": model,
            "status":200,
            "time":time
        }
    return answer