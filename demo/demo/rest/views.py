from django.shortcuts import render
from django.http import HttpResponse
import json
from django.views.decorators.csrf import csrf_exempt
# Create your views here.


def test_hello(request):
    return HttpResponse("Hello testing")


def construct_response(data=None):
    if not data:
        data = [{"question":"How is IR course?","score":0.9},{"question":"Contents of IR course?","score":0.7}]
    resp = HttpResponse(json.dumps(data))
    return resp
@csrf_exempt
def generate_prediction(request):
    algo_id = request.GET.get('algo','xgboost')
    body = request.body.decode("utf-8")
    json_body = json.loads(body)
    data = ""
    if "query" in json_body:
        data = json_body['query']
    print("Data parsed{}".format(data))
    print('Algo id{}'.format(algo_id))

    ## call algorithm below

    ## responese
    
    return construct_response()
    




