import numpy as np

def nonlinear(x, deriv = False):
  if(deriv==True):
   return x*(1-x)

  return 1/(1+np.exp(-x))

def predict(x_test, y_test, ss):
  prediction = np.around(nonlinear(np.dot(x_test,ss)))
  error = np.mean(np.abs(y_test - prediction))
  print("\nP:", prediction, "\n", "\nE:", error)

x = np.array([[1,0,1],
           [0,1,1],
           [0,1,0],
           [1,1,1]])

y = np.array([[1],
           [0],
           [0],
           [0]])

x_test = np.array([[1,0,0],
               [1,0,1],
               [0,1,1],
               [0,1,0]])

y_test = np.array([[1],
               [1],
               [0],
               [0]])
np.random.seed(1)


file = open("syn.txt", "r")
append = ""
for i in range(3):
    syn0 = file.readline()
    if (i == 2):
        append += syn0
    else:
        append += syn0[:-1] + ","
syn0 = np.array(append.replace(" ", ""))
print(syn0)
file.close()

#syn0 = 2*np.random.random((3,1)) - 1 

for _ in range(100000): 

  l0 = x
  l1 = nonlinear(np.dot(l0, syn0))

  l1_error = y - l1

  l1_delta = l1_error * nonlinear(l1, deriv = True)

  syn0 += l0.T.dot(l1_delta)

  if (_%1000 == 0):
   print("Iteration no.:", _)
   print("Error:", np.mean(np.abs(l1_error)))

  if (_ == 99999):
   file = open("syn.txt","w")
   file.write(str(syn0))
   file.close()  

predict(x_test,y_test,syn0)

"""for _ in range(10000):
  prediction = nonlinear(np.dot(x_test,syn0))
  error = np.mean(np.abs(y_test - prediction))
  error_delta = error * nonlinear(prediction, deriv = True)
  syn0 += x_test.T.dot(error_delta)
"""
  