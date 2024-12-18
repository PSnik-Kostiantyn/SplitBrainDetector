from django.shortcuts import render
from django.http import JsonResponse
import json

def dummy_neural_model(nodes, matrix):
    # Приклад обробки даних, що повертає ймовірності у %
    return [{"node": node, "probability": 75.0} for node in nodes]


# Головна сторінка
def index(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            nodes = data.get("nodes", [])
            matrix = data.get("matrix", [])

            if not matrix or not nodes:
                return JsonResponse({"error": "Матриця або вузли не можуть бути порожніми."}, status=400)

            response = dummy_neural_model(nodes, matrix)
            return JsonResponse({"data": response})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Невірний формат JSON."}, status=400)

    return render(request, 'index.html')



# Графічна сторінка
def cluster(request):
    return render(request, 'cluster.html')


# Інформаційна сторінка
def info(request):
    return render(request, 'info.html')
