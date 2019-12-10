from tkinter import *
from tkinter import ttk

from tkinter import messagebox
import NNtask1
import matplotlib.pyplot as plt
def getinfo():
  Numofhiddenlayers= e2.get()
  learningrate = e1.get()
  numberofneurons=e4.get()
  epochs=e3.get()
  #Feature1 = feature1.get()
 # Feature2=feature2.get()
  ActivationFunction=Activationfunction.get()
  #class2=str(Classes).split('&')
  #class1=class2[0]
  #Class2=class2[1]
  bais=var1.get()
  #ActivationFunction=var2.get()

  program(Numofhiddenlayers, learningrate, bais, ActivationFunction,epochs,numberofneurons)

def program(Numofhiddenlayers, learningrate, bais, ActivationFunction,epochs,numberofneurons):
  NNtask1.main(int(Numofhiddenlayers), int(learningrate), bais, ActivationFunction,int(epochs),int(numberofneurons))



top =Tk()
top.geometry("300x300")
top.title('Info')
Label(top, text="LearningRate", width=20).grid(row=0)
Label(top, text="Numoflayers", width=20).grid(row=3)
Label(top, text="Numofepochs", width=20).grid(row=6)
Label(top, text="Numofneurons", width=20).grid(row=9)


e1 = Entry(top, width=20)
e2 = Entry(top, width=20)
e3 = Entry(top, width=20)
e4=Entry(top, width=20)
e1.grid(row=0, column=1)
e2.grid(row=3, column=1)
e3.grid(row=6, column=1)
e4.grid(row=9, column=1)


Activationfunction = StringVar()
Label(top, text="ActivationFunction", width=20).grid(row=12)
combo = ttk.Combobox(top, width=20, textvariable=Activationfunction)
combo['values'] = ("tanh", "sigmoid")
combo.grid(row=12, column=1)
var1 = IntVar()
check=Checkbutton(top, text="bias", onvalue=1, offvalue=0,variable=var1).grid(row=15, column=1, sticky=W)
#var2 = IntVar()
#check=Checkbutton(top, text="ActivationFunction", onvalue=1, offvalue=0,variable=var1).grid(row=18, column=1, sticky=W)

B=Button(top,text="Submit",command= getinfo)
B.grid(row=30,column=1)
top.mainloop()


