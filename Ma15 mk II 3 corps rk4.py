"""
Projet de Ma15 du 05/2019 par Tissinier Axel, Marcelino Sarah, Tello Adam Esteban, 1T3
Ipsa Toulouse
Résolution par la méthode de Runge-Kutta d'ordre 4 d'un problème à trois corps (avec les mouvements du soleil négligés)
"""

from tkinter import *
import numpy as np
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Fonctions utiles pour l'interface
#Message d'erreur
def Error():
    messagebox.showerror('Error', 'Entrez un nombre entier.')
#Donne le nombre de jour dans un mois donné pour une année donnée
def NbJourMois(m,year):
    if m==3:
        if year%4==0:
            return 29
        else:
            return 28
    elif m in [1,3,5,7,8,10,12]:
        return 31
    else:
        return 30
#Ajoute à une date un nombre de jour
def Ajout(date,jour):
    spl = date.split("/")
    day = int(spl[0])
    month = int(spl[1])
    year = int(spl[2])
    day = day + jour
    m = NbJourMois(month,year)
    while day>m:
        month = month +1
        day = day - m
        if month==13:
            month = 1
            year = year + 1
        m = NbJourMois(month,year)
    if day<10:
        day = "0" + str(day)
    else:
        day = str(day)
    if month<10:
        month = "0" + str(month)
    else:
        month = str(month)
    year = str(year)
    date = "/".join([day,month,year])
    return date

#Définitions des variables et données
h = 1 #Pas (en jour)
date = "10/06/1971" #Variable qui donnera la date
t = 0 #compteur de temps (au cas où le pas h serai inférieur à un jour)

#masses en masse solaire
masse_S = 1
masse_T = 3.0024*10**-6
masse_M = 3.228*10**-7
masse_L = 3.695*10**-8

#Données du 10 juin 1971 (origine = barycentre du soleil) récupérées sur https://ssd.jpl.nasa.gov/horizons.cgi#top)
#Distances Soleil-Terre en ua   
X_T = -1.93269*10**-1 
Y_T = -9.92244*10**-1
Z_T = -9.3011*10**-5
#vitesses Terre en ua/day
Vx_T = 1.65889*10**-2 
Vy_T = -3.37815*10**-3 
Vz_T = -4.7297*10**-7
#distances Soleil-Mars en ua
X_M = 2.5006*10**-1
Y_M = -1.40956
Z_M = -3.57275*10**-2
#vitesses Mars en ua/day
Vx_M = 1.43106*10**-2
Vy_M = 3.61649*10**-3
Vz_M = -2.77001*10**-4
#distances Soleil-Lune en ua
X_L = -1.93221*10**-1
Y_L = -9.94799*10**-1
Z_L = -2.54734*10**-4
#vitesses Lune en ua/day
Vx_L = 1.71819*10**-2
Vy_L = -3.34539*10**-3
Vz_L = 3.80163*10**-5

#positions Soleil (réferentiel à partir du barycentre du soleil)  
X_S,Y_S,Z_S = 0,0,0

#Regroupement des données dans des listes pour un traitement moins lourd
position_Terre = [X_T,Y_T,Z_T]
position_Mars = [X_M,Y_M,Z_M]
position_Lune = [X_L,Y_L,Z_L]
position_Soleil = [X_S,Y_S,Z_S]
info_Terre = [X_T,Y_T,Z_T,Vx_T,Vy_T,Vz_T]
info_Mars =  [X_M,Y_M,Z_M,Vx_M,Vy_M,Vz_M]
info_Lune = [X_L,Y_L,Z_L,Vx_L,Vy_L,Vz_L]

#calcul de l'accéleration à partir des positions et masses
def fonction_grav(z_1, z_2, z_3, mz_2, mz_3):
    #Constante Gravitationnelle en masse solaire.unité astronomique.jour^-2
    G = 2.959130713485796*10**-4
    #Calculs des distances entre objets et mise en forme vectorielle (liste)
    x = z_1[0] - z_2[0]
    y = z_1[1] - z_2[1]
    z = z_1[2] - z_2[2]
    x2 = z_1[0] - z_3[0]
    y2 = z_1[1] - z_3[1]
    z2 = z_1[2] - z_3[2] 
    u = [x,y,z]
    u2 = [x2, y2 ,z2]

    aX = mz_2*(-G* x * 1/(np.linalg.norm(u))**3) + mz_3*(-G* x2 * 1/(np.linalg.norm(u2))**3)
    aY = mz_2*(-G* y * 1/(np.linalg.norm(u))**3) + mz_3*(-G* y2 * 1/(np.linalg.norm(u2))**3)
    aZ = mz_2*(-G* z * 1/(np.linalg.norm(u))**3) + mz_3*(-G* z2 * 1/(np.linalg.norm(u2))**3)

    acceleration = [aX, aY, aZ]
    return acceleration

#changement de variable afin de se ramener à un problème de Cauchy
def fchangement(infos, z_2, z_3, mz_2, mz_3): 
    changement = []
    positions = []

    positions.append(infos[0])
    positions.append(infos[1])
    positions.append(infos[2])
    
    changement.extend((infos[3], infos[4], infos[5]))
    changement.extend(fonction_grav(positions, z_2, z_3, mz_2, mz_3))

    return changement

#Fonction qui renvoie la position à n+1 en suivant la méthode de Runge-Kutta d'ordre 4
def RK4(infos, z_2, z_3, mz_2, mz_3):
        global h
        
        k1p = []
        k2p = []
        k3p = []
        k2 = []
        k3 = []
        k4 = []
        
        Zn = infos
    
        k1 = fchangement(infos, z_2, z_3, mz_2, mz_3)
        for i in range(len(k1)):
            k1p.append(Zn[i]+ k1[i] * (h/2))

        k2 = fchangement(k1p,z_2, z_3, mz_2, mz_3)
        for i in range(len(k2)):
            k2p.append(Zn[i]+k2[i]*(h/2))

        k3 = fchangement(k2p,z_2, z_3, mz_2, mz_3)
        for i in range(len(k2)):
            k3p.append(Zn[i]+k3[i]*h)
        
        k4 = fchangement(k3p,z_2, z_3, mz_2, mz_3)

        for i in range(len(k1)):
            k1[i] *= h/6

        for i in range(len(k2)):
            k2[i] *= h/3

        for i in range(len(k3)):
            k3[i] *= h/3

        for i in range(len(k4)):
            k4[i] *= h/6

        Znplus1 = []

        for i in range(len(Zn)):
            a = Zn[i] + k1[i] + k2[i] + k3[i] + k4[i]
            Znplus1.append(a)

        return Znplus1


#Fonction d'animation 
def move():
    global t, h, date, flag, info_Terre, info_Mars, position_Soleil, position_Terre, position_Mars, masse_S, masse_T, masse_M
   
    info_Mars = RK4(info_Mars, position_Soleil, position_Terre, masse_S, masse_T)
    info_Terre = RK4(info_Terre, position_Soleil, position_Mars, masse_S, masse_M)

    x1=int(info_Terre[0]*150+300)
    y1=int(info_Terre[1]*150+300)
    x2=int(info_Mars[0]*150+300)
    y2=int(info_Mars[1]*150+300)
     
    can1.coords(oval1,x1-10,y1-10,x1+10,y1+10)
    can1.coords(oval2,x2-10,y2-10,x2+10,y2+10)
    lab.configure(text=date)
    
    position_Terre = info_Terre[0:3]
    position_Mars = info_Mars[0:3]
    
    t += h
    if t>1:
        day = t//1
        t = t%1
        date = Ajout(date,day)
    #NB : Si on veux que l'animation soit plus rapide, réduire ce nombre, attention au performances
    if flag >0:
        fen1.after(10,move)
     

#Démarage/arrêt de l'animation
def start(): 
    global flag
#Test si l'animation est en cours ou non
    if flag == 0:
        flag = 1
        move()
        bou2.configure(text="Stop")
    else: #arrêt de l'animation
        flag = 0
        bou2.configure(text="Start")

 
#Variable pour l'animation
flag =0

def graph_launch():
    global flag
    if flag == 1:
        flag = 0
        bou2.configure(text="Start")
    try:
        value = int(variable.get())
        graph()
    except:
        Error()

#Fonction de création de graphique
def graph():
    global t, h, date, variable, info_Mars, info_Terre, position_Soleil, position_Terre, position_Mars, masse_S, masse_T, masse_M
    
    #Récupération du nombre de jour
    value = variable.get()
    iterations = int(value)
    iterations = int(iterations/h)
    iteration = str(iterations*h)
    
    XMars = []
    YMars = []
    XTerre = []
    YTerre = []
    
    for i in range(iterations):

        #Pour raison inconnue, le Yterre se retrouve à l'envers sur Matplotlib, on l'inverse alors ici
        XMars.append(info_Mars[0])
        YMars.append(-info_Mars[1])
        XTerre.append(info_Terre[0])
        YTerre.append(-info_Terre[1])

        info_Mars = RK4(info_Mars, position_Soleil, position_Terre, masse_S, masse_T)
        info_Terre = RK4(info_Terre, position_Soleil, position_Mars, masse_S, masse_M)

        position_Terre = info_Terre[0:3]
        position_Mars = info_Mars[0:3]

        t += h
    
    #mise à jour de la fenêtre graphique pour la cohérence
    t=int(t)
    date_init = date
    date = Ajout(date,t)
    x1=int(info_Terre[0]*150+300)
    y1=int(info_Terre[1]*150+300)
    x2=int(info_Mars[0]*150+300)
    y2=int(info_Mars[1]*150+300) 
    can1.coords(oval1,x1-10,y1-10,x1+10,y1+10)
    can1.coords(oval2,x2-10,y2-10,x2+10,y2+10)
    lab.configure(text=date)
    
    root = Tk()
    root.title("Graphe des trajectoires du "+date_init+" au "+date+" (" + iteration + " jours)")
    f = Figure(figsize=(5,5), dpi=100)
    a = f.add_subplot(111)

    #Dessin de la trajectoire sur mathplotlib
    a.plot(XTerre,YTerre,'b-') #trajectoire de la Terre en bleu
    a.plot(XMars, YMars, 'r-') #trajectoire de Mars en rouge
    a.plot(0,0,'yo') #position du soleil, considéré totalement immobile dans ce problème
    a.axis('scaled')
    graph = FigureCanvasTkAgg(f, root)

    canvas = graph.get_tk_widget()
    canvas.grid(row=0, column=0)

    root.mainloop()


#Données de départ
x1=int(info_Terre[0]*150+300)
y1=int(info_Terre[1]*150+300)
x2=int(info_Mars[0]*150+300)
y2=int(info_Mars[1]*150+300)


#Création de la fenêtre principale
fen1 = Tk()
fen1.title("Animation des trajectoires de Mars et de la Terre (à partir du 10/06/1971)")

# création des objets
can1 = Canvas(fen1,bg='black',height=600, width=600)
can1.pack(side=LEFT, padx =5, pady =5)

oval1 = can1.create_oval(x1-10, y1-10, x1+10, y1+10, width=2, fill='#3069cb')
oval2 = can1.create_oval(x2-10, y2-10, x2+10, y2+10, width=2, fill='#df5003')
oval3 = can1.create_oval(270, 270, 330, 330, width=2, fill="#f9df08")

bou1 = Button(fen1,text='Quitter', width =10, command=fen1.destroy)
bou1.pack(side=BOTTOM)

lab0 = Label(fen1,text="Animation", width =10, font=("Arial", 10, "bold"))
lab0.pack()
bou2 = Button(fen1, text='Start', width =10, command=start)
bou2.pack()

lab1 = Label(fen1, text="", width =10)
lab1.pack()
lab2 = Label(fen1, text="Années passées", width =12)
lab2.pack()
lab = Label(fen1, text=date, width =10)
lab.pack()
lab3 = Label(fen1, text="", width =10)
lab3.pack()

variable = StringVar()
lab4 = Label(fen1, text="Trajectoire", width =10, font=("Arial", 10, "bold"))
lab4.pack()
lab5 = Label(fen1, text="Nb de jours", width =10)
lab5.pack()
entry = Entry(fen1, width =10, textvariable=variable)
entry.pack()

bou4 = Button(fen1, text='Créer Graph', width =10, command=graph_launch)
bou4.pack()

fen1.mainloop()
