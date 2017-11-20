import numpy as np
import pylab
import matplotlib.pyplot as plt
import math

learn_rate = 0.3

# Passo
def step(x):
    if (x > 0):
        return 1
    else:
        return -1;   

# Dados
data = np.array([[-1,-1,1],
                   [-1,1,1],
                   [1,-1,1],
                   [1,1,1] ])
				   
# Dados de saída
output = np.array([[-1,1,1,1]]).T

np.random.seed(1)

# Inicialização dos pesos
weights = 2 * np.random.random((3,1)) - 1
print ("Pesos antes do treinamento", weights)

# Listagem de erros
erros = []

# Treinamento
for iterator in range(100):

    for item, intended in zip(data, output):
               	
        output_adaline = (item[0] * weights[0]) + (item[1] * weights[1]) + (item[2] * weights[2])

        output_adaline = step(output_adaline)

        error = intended - output_adaline                
        erros.append(error)
                
		# Calculo dos pesos
        weights[0] = weights[0] + learn_rate * error * item[0]
        weights[1] = weights[1] + learn_rate * error * item[1]
        weights[2] = weights[2] + learn_rate * error * item[2]

print ("Pesos depois do treinamento", weights)

for item, intended in zip(data, output):    
    
	output_adaline = (item[0] * weights[0]) + (item[1] * weights[1]) + (item[2] * weights[2])   
    
	output_adaline = step(output_adaline)

	print ("Atual ", output_adaline, "Desejado ", intended)

# Plotar erros
plt.plot(erros, c = '#bbaaff', label = 'ERROR')
plt.title("Erros Adaline (2,-2)")
plt.legend()
pylab.xlabel('Erro')
pylab.ylabel('Valor')
plt.show()

os.system("Pause")