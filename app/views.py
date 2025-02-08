from django.shortcuts import render
from django.http import JsonResponse
import json

from app.model.gradientBoosting import predict_gb
from app.model.isDeadCluster import isClusterDead
from app.model.isSplitBrain import isSplitBrain
from app.model.neuralModel import predict_neural_model, teach_neural_model
from app.model.randomForest import predict_rf


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
                print("100%")
                return JsonResponse({"probability": 100})

            probability = predict_rf(nodes, matrix)
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

            probability = predict_gb(nodes, matrix)
            probability = round(probability * 100, 2)
            return JsonResponse({"probability": probability})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Невірний формат JSON."}, status=400)

    return render(request, 'cluster.html')

def info(request):
    return render(request, 'info.html')

def teach(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            nodes = data.get("nodes", [])
            matrix = data.get("matrix", [])
            split_brain = data.get("split_brain", 0)
            print(nodes)
            print(matrix)
            print(split_brain)

            if isClusterDead(nodes, matrix):
                return JsonResponse({"probability": -1})

            if isSplitBrain(nodes, matrix):
                if split_brain == 0:
                    return JsonResponse({
                        "probability": -2
                    })
            else:
                if split_brain == 1:
                    return JsonResponse({
                        "probability": -2
                    })

            probability = teach_neural_model(nodes, matrix)
            probability = round(probability, 7)
            return JsonResponse({"probability": probability})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Невірний формат JSON."}, status=400)

    return render(request, 'teaching.html')