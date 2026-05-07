from django.shortcuts import render
from django.http import JsonResponse
import json

from app.model.DataPreparation import isSplitBrain, isClusterDead
from app.model.catBoost import predict_cb, teach_cb
from app.model.gradientBoosting import predict_gb, teach_gb
from app.model.randomForest import predict_rf, teach_rf


def index(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(request.body)
            nodes = data.get("nodes", [])
            matrix = data.get("matrix", [])
            model = data.get("model", "gb")

            if not matrix or not nodes:
                return JsonResponse({"error": "Матриця або вузли не можуть бути порожніми."}, status=400)

            if isClusterDead(nodes, matrix):
                return JsonResponse({"probability": -1})

            if isSplitBrain(nodes, matrix):
                return JsonResponse({"probability": 100})

            if model == "rf":
                probability = predict_rf(nodes, matrix)
            elif model == "gb":
                probability = predict_gb(nodes, matrix)
            else:
                probability = predict_cb(nodes, matrix)

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
                return JsonResponse({
                    "probability_rf": -1,
                    "probability_gb": -1,
                    "probability_cb": -1,
                })

            if isSplitBrain(nodes, matrix):
                return JsonResponse({
                    "probability_rf": 100,
                    "probability_gb": 100,
                    "probability_cb": 100,
                })

            probability_rf = round(predict_rf(nodes, matrix) * 100, 2)
            probability_gb = round(predict_gb(nodes, matrix) * 100, 2)
            probability_cb = round(predict_cb(nodes, matrix) * 100, 2)

            return JsonResponse({
                "probability_rf": probability_rf,
                "probability_gb": probability_gb,
                "probability_cb": probability_cb,
            })

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
            model_name = data.get("model", "gb")

            print("Nodes:", nodes)
            print("Matrix:", matrix)
            print("Split Brain Flag:", split_brain)
            print("Model:", model_name)

            if isClusterDead(nodes, matrix):
                return JsonResponse({"probability": -1})

            actual_split_brain = isSplitBrain(nodes, matrix)
            if actual_split_brain != bool(split_brain):
                return JsonResponse({"probability": -2})

            if model_name == "rf":
                probability = teach_rf(nodes, matrix)
            elif model_name == "cb":
                probability = teach_cb(nodes, matrix)
            else:
                probability = teach_gb(nodes, matrix)

            probability = round(probability, 7)
            return JsonResponse({"probability": probability})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Невірний формат JSON."}, status=400)

    return render(request, 'teaching.html')