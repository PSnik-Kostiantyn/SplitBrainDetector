from django.shortcuts import render
from django.http import JsonResponse
import json

from app.model.DataPreparation import isSplitBrain, isClusterDead
from app.model.catBoost import predict_cb, teach_cb
from app.model.gradientBoosting import predict_gb, teach_gb
from app.model.randomForest import predict_rf, teach_rf


def _get_ensemble(key: str):
    from ensemble import get_ensemble
    return get_ensemble(key, calibrated=False)


def index(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            nodes = data.get("nodes", [])
            matrix = data.get("matrix", [])
            model_key = data.get("model", "cb")

            if not matrix or not nodes:
                return JsonResponse({"error": "Empty input."}, status=400)
            if isClusterDead(nodes, matrix):
                return JsonResponse({"probability": -1})
            if isSplitBrain(nodes, matrix):
                return JsonResponse({"probability": 100})

            p = (predict_rf(nodes, matrix) if model_key == "rf" else
                 predict_gb(nodes, matrix) if model_key == "gb" else
                 predict_cb(nodes, matrix))
            return JsonResponse({"probability": round(p * 100, 2)})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)
    return render(request, "index.html")


def cluster(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            nodes = data.get("nodes", [])
            matrix = data.get("matrix", [])
            mode = data.get("mode", "base")

            if not matrix or not nodes:
                return JsonResponse({"error": "Empty input."}, status=400)
            if isClusterDead(nodes, matrix):
                return JsonResponse({"dead": True})
            if isSplitBrain(nodes, matrix):
                return JsonResponse({"split_brain": True})

            if mode == "ensemble":
                return JsonResponse({
                    "mode": "ensemble",
                    "e1": round(_get_ensemble("e1").predict(nodes, matrix) * 100, 2),
                    "e2": round(_get_ensemble("e2").predict(nodes, matrix) * 100, 2),
                    "e3": round(_get_ensemble("e3").predict(nodes, matrix) * 100, 2),
                })
            return JsonResponse({
                "mode": "base",
                "probability_rf": round(predict_rf(nodes, matrix) * 100, 2),
                "probability_gb": round(predict_gb(nodes, matrix) * 100, 2),
                "probability_cb": round(predict_cb(nodes, matrix) * 100, 2),
            })

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)
    return render(request, "cluster.html")


def info(request):
    return render(request, "info.html")


def teach(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            nodes = data.get("nodes", [])
            matrix = data.get("matrix", [])
            split_brain = data.get("split_brain", 0)
            model_name = data.get("model", "gb")

            if isClusterDead(nodes, matrix):
                return JsonResponse({"probability": -1})
            if isSplitBrain(nodes, matrix) != bool(split_brain):
                return JsonResponse({"probability": -2})

            p = (teach_rf(nodes, matrix) if model_name == "rf" else
                 teach_cb(nodes, matrix) if model_name == "cb" else
                 teach_gb(nodes, matrix))
            return JsonResponse({"probability": round(p, 7)})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)
    return render(request, "teaching.html")
