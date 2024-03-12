class SimpleLinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mean_x = sum(self.x) / len(self.x)
        self.mean_y = sum(self.y) / len(self.y)
        self.calculate_coefficients()

    def calculate_coefficients(self):
        numerator_beta1 = len(self.x)*sum([xi*yi for xi,yi in zip(self.x, self.y)]) - sum(self.x)*sum(self.y)
        denominator_beta1 = len(self.x)*sum([xi**2 for xi in self.x]) - sum(self.x)**2
        self.beta1 = numerator_beta1 / denominator_beta1
        self.beta0 = (sum(self.y) - self.beta1 * sum(self.x)) / len(self.x)

    def predict(self, new_x):
        return [self.beta0 + self.beta1*xi for xi in new_x]

    def calculate_correlation_and_determination(self):
        numerator_r = sum([xi*yi for xi,yi in zip(self.x, self.y)]) - len(self.x)*self.mean_x*self.mean_y
        denominator_r = ((sum([xi**2 for xi in self.x]) - len(self.x)*self.mean_x**2) * (sum([yi**2 for yi in self.y]) - len(self.y)*self.mean_y**2))**0.5
        self.r = numerator_r / denominator_r
        self.r_squared = self.r**2

# Sales (Millon Euro)
y = [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]

# Advertising (Millon Euro)
x = [23, 26, 30, 34, 43, 48, 52, 57, 58]

# Create a SimpleLinearRegression object
slr = SimpleLinearRegression(x, y)

# Print the coefficients
print(f"Los coeficientes de la regresión lineal son beta0 = {slr.beta0} y beta1 = {slr.beta1}")

# Calculate and print the correlation coefficient and the coefficient of determination
slr.calculate_correlation_and_determination()
print(f"El coeficiente de correlación (r) es {slr.r}")
print(f"El coeficiente de determinación (r cuadrado) es {slr.r_squared}")

# Make predictions for new x values and print them
new_x = [60, 65, 70, 75, 80]
new_y_pred = slr.predict(new_x)
for xi, yi in zip(new_x, new_y_pred):
    print(f"La predicción de ventas para una inversión en publicidad de {xi} millones de euros es {yi} millones de euros")