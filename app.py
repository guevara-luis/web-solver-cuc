from flask import Flask, render_template, request, jsonify, redirect
import json
import os
from datetime import datetime
from sympy import symbols, diff, sympify, lambdify
from scipy.optimize import root_scalar

app = Flask(__name__)
x = symbols("x")
history_file = "history.json"

# Guardar historial

def save_to_history(data):
    if not os.path.exists(history_file):
        with open(history_file, "w") as f:
            json.dump([], f)

    with open(history_file, "r") as f:
        history = json.load(f)

    history.insert(0, data)

    with open(history_file, "w") as f:
        json.dump(history[:30], f, indent=2)

# Cargar historial

def load_history():
    if not os.path.exists(history_file):
        return []
    with open(history_file, "r") as f:
        return json.load(f)

# Evaluar funcion desde string

def evaluate_function(expr, val):
    try:
        func = lambdify(x, sympify(expr), "numpy")
        return func(val)
    except Exception as e:
        raise ValueError("Error evaluando la función: " + str(e))

# Método de Newton-Raphson

def newton_raphson(expr, x0, tol):
    fx = sympify(expr)
    dfx = diff(fx, x)
    f = lambdify(x, fx, "numpy")
    df = lambdify(x, dfx, "numpy")
    
    xi = x0
    pasos = []
    for i in range(100):
        fxi = f(xi)
        dfxi = df(xi)
        if dfxi == 0:
            raise ZeroDivisionError("Derivada cero. El método no converge.")
        xi1 = xi - fxi / dfxi
        error = abs((xi1 - xi) / xi1)
        pasos.append({"x": round(float(xi), 6), "fx": round(float(fxi), 6), "error": round(error, 6)})
        if error < tol:
            break
        xi = xi1
    return xi1, pasos

# Método de Bisección

def bisection_method(expr, x0, tol):
    f = lambdify(x, sympify(expr), "numpy")
    a = x0
    b = x0 + 1
    while f(a) * f(b) > 0:
        a -= 1
        b += 1
        if a < -100 or b > 100:
            raise ValueError("No se encontró un intervalo adecuado.")
    resultado = root_scalar(f, method='brentq', bracket=[a, b], xtol=tol)
    if not resultado.converged:
        raise ValueError("El método de bisección no converge.")
    raiz = resultado.root
    pasos = []
    c = (a + b) / 2
    for i in range(100):
        fc = f(c)
        error = abs((b - a) / 2)
        pasos.append({"x": round(float(c), 6), "fx": round(float(fc), 6), "error": round(error, 6)})
        if error < tol or f(c) == 0:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2
    return raiz, pasos

# Ruta principal

@app.route("/")
def index():
    return render_template("index.html")

# Resolver

@app.route("/solve", methods=["POST"])
def solve():
    try:
        data = request.get_json()
        expr = data["funcion"]
        x0 = float(data["x0"])
        tol = float(data["tolerancia"])
        metodo = data["metodo"]
        comparar = data.get("comparar", False)

        if metodo == "Newton-Raphson":
            raiz, pasos = newton_raphson(expr, x0, tol)
            metodo_usado = "newton"
        else:
            raiz, pasos = bisection_method(expr, x0, tol)
            metodo_usado = "bisection"

        resultado = {
            "raiz": round(float(raiz), 6),
            "pasos": pasos,
            "iteraciones": len(pasos),
            "metodo": metodo_usado
        }

        if comparar:
            f = lambdify(x, sympify(expr), "numpy")
            a = x0
            b = x0 + 1
            while f(a) * f(b) > 0:
                a -= 1
                b += 1
                if a < -100 or b > 100:
                    break
            comp = root_scalar(f, method='brentq', bracket=[a, b], xtol=tol)
            if comp.converged:
                resultado["comparacion"] = {"raiz": round(float(comp.root), 6)}

        save_to_history({
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "funcion": expr,
            "metodo": metodo_usado,
            "raiz": round(float(raiz), 6)
        })

        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Mostrar historial

@app.route("/historial")
def historial():
    registros = load_history()
    return render_template("historial.html", registros=registros)

# Borrar historial

@app.route("/borrar_historial", methods=["POST"])
def borrar_historial():
    if os.path.exists(history_file):
        with open(history_file, "w") as f:
            json.dump([], f)
    return render_template("historial.html", registros=[])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto asignado por Render o 5000 por defecto
    app.run(host="0.0.0.0", port=port)
