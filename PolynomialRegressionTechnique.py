import numpy as np

class Regresion:
  
  def __init__(self, x, y):
    self.x = x
    self.y = y
    
  # Function to generate the matrix(Generacion de Matriz)
  def generar_matriz_diseno(self, grado):
    matriz_diseno = np.ones((self.x.shape[0], grado + 1))
    for i in range(1, grado + 1):
      matriz_diseno[:, i] = self.x ** i
    return matriz_diseno

  # Funtion to calculate the model coefficients(Calcula coeficientes)
  def calcular_coeficientes(self, grado):
    matriz_diseno = self.generar_matriz_diseno(grado)
    return np.linalg.inv(matriz_diseno.T @ matriz_diseno) @ matriz_diseno.T @ self.y

  # Funtion to calculate the estimated efficiency (Calcula eficiencia estimada)
  def calcular_eficiencia_estimada(self, coeficientes):
    matriz_diseno = self.generar_matriz_diseno(coeficientes.shape[0] - 1)
    return matriz_diseno @ coeficientes

  # Funtion to calculate the coefficient of determination R^2 (Calcula coeficiente de determinacion)
  def calcular_r2(self, y_estimada):
    ss_res = np.sum((self.y - y_estimada) ** 2)
    ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
    return 1 - (ss_res / ss_tot)

  # Function for calculating the correlation coefficient (Calcula coeficiente de correlacion)
  def calcular_coeficiente_correlacion(self, y_estimada):
    return np.corrcoef(self.y, y_estimada)[0, 1]

# Data definition
tamanos_lote = np.array([108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89])
#
eficiencia_maquina = np.array([95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93])
#
# Create an instance of the Regresion object
regresion = Regresion(tamanos_lote, eficiencia_maquina)

# Calculation of the coefficients for each grade
coeficientes_lineal = regresion.calcular_coeficientes(1)
coeficientes_cuadratica = regresion.calcular_coeficientes(2)
coeficientes_cubica = regresion.calcular_coeficientes(3)

# Calculation of the estimated efficiency for each grade
eficiencia_estimada_lineal = regresion.calcular_eficiencia_estimada(coeficientes_lineal)
eficiencia_estimada_cuadratica = regresion.calcular_eficiencia_estimada(coeficientes_cuadratica)
eficiencia_estimada_cubica = regresion.calcular_eficiencia_estimada(coeficientes_cubica)

# Calculation of the coefficients of determination R^2 for each degree
r2_lineal = regresion.calcular_r2(eficiencia_estimada_lineal)
r2_cuadratica = regresion.calcular_r2(eficiencia_estimada_cuadratica)
r2_cubica = regresion.calcular_r2(eficiencia_estimada_cubica)

# Calculation of correlation coefficients for each degree
coeficiente_correlacion_lineal = regresion.calcular_coeficiente_correlacion(eficiencia_estimada_lineal)
coeficiente_correlacion_cuadratica = regresion.calcular_coeficiente_correlacion(eficiencia_estimada_cuadratica)
coeficiente_correlacion_cubica = regresion.calcular_coeficiente_correlacion(eficiencia_estimada_cubica)


# Printing of Regression Equations, R^2 Determination Coefficients, and Correlation Coefficients
print("----------------------------------------------------------------")

print("Ecuación de regresión lineal:")
print("Y = {:.4f} + {:.4f}X".format(coeficientes_lineal[0], coeficientes_lineal[1]))
print("Coeficiente de determinación R^2: {:.4f}".format(r2_lineal))
print("Coeficiente de correlación: {:.4f}".format(coeficiente_correlacion_lineal))

print("----------------------------------------------------------------")

print("Ecuación de regresión cuadrática:")
print("Y = {:.4f} + {:.4f}X + {:.4f}X^2".format(coeficientes_cuadratica[0], coeficientes_cuadratica[1], coeficientes_cuadratica[2]))
print("Coeficiente de determinación R^2: {:.4f}".format(r2_cuadratica))
print("Coeficiente de correlación: {:.4f}".format(coeficiente_correlacion_cuadratica))

print("----------------------------------------------------------------")

print("Ecuación de regresión cúbica:")
print("Y = {:.4f} + {:.4f}X + {:.4f}X^2 + {:.4f}X^3".format(coeficientes_cubica[0], coeficientes_cubica[1], coeficientes_cubica[2], coeficientes_cubica[3]))
print("Coeficiente de determinación R^2: {:.4f}".format(r2_cubica))
print("Coeficiente de correlación: {:.4f}".format(coeficiente_correlacion_cubica))

print("----------------------------------------------------------------")