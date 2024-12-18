from django.shortcuts import render
from django.http import JsonResponse
import json

from app.model.isDeadCluster import isClusterDead

def dummy_neural_model(nodes, matrix):
    return [{"node": node, "probability": 50.0} for node in nodes]

def index(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            nodes = data.get("nodes", [])
            matrix = data.get("matrix", [])

            if not matrix or not nodes:
                return JsonResponse({"error": "Матриця або вузли не можуть бути порожніми."}, status=400)

            if isClusterDead(nodes, matrix):
                print("Dead")
                return JsonResponse({"data": [{"node": node, "probability": -1.0} for node in nodes]})
            response = dummy_neural_model(nodes, matrix)
            return JsonResponse({"data": response})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Невірний формат JSON."}, status=400)

    return render(request, 'index.html')

def cluster(request):
    return render(request, 'cluster.html')

def info(request):
    return render(request, 'info.html')
