from django.shortcuts import render
from django.http import JsonResponse
import json

from app.model.isDeadCluster import isClusterDead
from app.model.isSplitBrain import isSplitBrain
from app.model.neuralModel import predict_neural_model

def index(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            nodes = data.get("nodes", [])
            matrix = data.get("matrix", [])
            print(nodes)
            print(matrix)
            if not matrix or not nodes:
                return JsonResponse({"error": "Матриця або вузли не можуть бути порожніми."}, status=400)

            if isClusterDead(nodes, matrix):
                return JsonResponse({"probability": -1})

            if isSplitBrain(nodes, matrix):
                return JsonResponse({"probability": 100})

            probability = predict_neural_model(nodes, matrix)
            probability = round(probability * 100, 2)
            return JsonResponse({"probability": probability})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Невірний формат JSON."}, status=400)

    return render(request, 'index.html')


def cluster(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            nodes = data.get("nodes", [])
            matrix = data.get("matrix", [])
            print(nodes)
            print(matrix)
            if not matrix or not nodes:
                return JsonResponse({"error": "Матриця або вузли не можуть бути порожніми."}, status=400)

            if isClusterDead(nodes, matrix):
                return JsonResponse({"probability": -1})

            if isSplitBrain(nodes, matrix):
                return JsonResponse({"probability": 100})

            probability = predict_neural_model(nodes, matrix)
            probability = round(probability * 100, 2)
            return JsonResponse({"probability": probability})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Невірний формат JSON."}, status=400)

    return render(request, 'cluster.html')

def info(request):
    return render(request, 'info.html')

def teach(request):
    return render(request, 'teaching.html')