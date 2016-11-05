setwd("C:/Users/Jos� Manuel/Documents/ICD/proyecto-kaggle-titanic-josemalvarezg1")

# ------- Pre-procesamiento ------- 

#Se lee el dataset de training.
train = read.csv(file = "data/train.csv", header = T)
#Se lee el dataset de testing.
test = read.csv(file = "data/test.csv", header = T)

#Se identifican las columnas de ambos datasets.
colnames(train) <- c("ID", "Sobrevivi�", "Clase", "Nombre", "Sexo", "Edad", "Hermanos/C�nyuges", "Padres/Ni�os", "Ticket", "Tarifa", "Cabina", "Embarcaci�n")
colnames(test) <- c("ID", "Clase", "Nombre", "Sexo", "Edad", "Hermanos/C�nyuges", "Padres/Ni�os", "Ticket", "Tarifa", "Cabina", "Embarcaci�n")

#Se elimina la primera columna de los datasets.
train = train[,-1]
test = test[,-1]

#Se calcula la edad promedio y es colocada a los N/A.
train[,5][is.na(train[, 5])] <- 0
train[,5][train[, 5] == 0] <- ceiling(mean(train[["Edad"]]))

#Cambiar sexo a num�rico. Male = 0, Female = 1.
train$Sexo = (train$Sexo=="female")*1

# ------- K-Medias ------- 

#Se trabajar� s�lo con la columna "Edad".
trainK = train[,c(5)]

#Se aplica el Codo de Jambu para obtener el K m�s adecuado al m�todo de K-Medias.
InerciaIC = rep(0,50)
for (k in 1:50) {
  grupos = kmeans(trainK, k)
  InerciaIC[k] = grupos$tot.withinss
}

#Se grafica la Inercia Inter-Clases.
plot(InerciaIC, col = "blue", type = "b")
#Se puede observar que cambia muy poco a partir de K=4 y K=5.

#Se calcula K-Medias con K=5 y 100 iteraciones.
clusters <- kmeans(trainK, 5, iter.max = 100) 

#Se grafica el dataset inicial.
plot(trainK, pch = 20)

#Se colorean los cinco grupos en el gr�fico.
plot(trainK, col = clusters$cluster)

# ------- Clasificaci�n Jer�rquica ------- 

#Se trabaja el dataset como una matriz.
datos = as.matrix(trainK)
#Se calcula la matriz de distancia.
distancia = dist(datos)

#Se aplica el M�todo Complete.
cluster = hclust(distancia, method = "complete")
plot(cluster)
#Se determina la altura requerida con k clusters, cortando el dendograma con k clases:
corteD = cutree(cluster, k = 5)
#Observamos la cantidad de clusters.
unique(corteD)
#Graficamos los clusters.
plot(trainK, col = corteD, main = "COMPLETE")

#Se aplica el M�todo Single.
cluster = hclust(distancia, method = "single")
plot(cluster)
#Se determina la altura requerida con k clusters, cortando el dendograma con k clases:
corteD = cutree(cluster, k = 5)
#Observamos la cantidad de clusters
unique(corteD)
#Graficamos los clusters.
plot(trainK, col = corteD, main = "SINGLE")

#Se aplica el M�todo Average.
cluster = hclust(distancia, method = "average")
plot(cluster)
#Se determina la altura requerida con k clusters, cortando el dendograma con k clases:
corteD = cutree(cluster, k = 5)
#Observamos la cantidad de clusters.
unique(corteD)
#Graficamos los clusters.
plot(trainK, col = corteD, main = "AVERAGE")

#Se aplica el M�todo Ward.
cluster = hclust(distancia, method = "ward.D")
plot(cluster)
#Se determina la altura requerida con k clusters, cortando el dendograma con k clases:
corteD = cutree(cluster, k = 5)
#Observamos la cantidad de clusters.
unique(corteD)
#Graficamos los clusters.
plot(trainK, col = corteD, main = "WARD")

# ------- Reglas de Asociaci�n ------- 

library(arules)
library(arulesViz)

#Se lee el dataset Titanic Raw.
load("data/titanic.raw.Rdata")
str(titanic.raw)

#Se transforma dataframe en transaccional.
trans <- as(titanic.raw, "transactions")

#Se generan las reglas.
reglas <- apriori(trans)
summary(reglas)

#Se muestran las las 10 transacciones con mayor frecuencia en el dataset.
itemFrequencyPlot(trans,topN=10,type="absolute")

#Se ordenan las reglas por confianza
confianzaAlta <-sort(reglas, by="confidence", decreasing=TRUE)
inspect(head(confianzaAlta))

#Se ordenan las reglas por soporte
supportAlto <-sort(reglas, by="support", decreasing=TRUE)
inspect(head(supportAlto))

#Se ordenan las reglas por lift
liftAlto <-sort(reglas, by="lift", decreasing=TRUE)
inspect(head(liftAlto))

# ------- �rboles de Decisi�n ------- 

library("rpart")
library("rpart.plot")

#Se trabajar� s�lo con las columnas "Sobrevivi�", "Clase", "Sexo", "Edad", "Hermanos/C�nyugues", "Padres/Ni�os" y "Tarifa".
trainT <- subset(train, select = c(1,2,4,5,6,7,9))

#Se obtiene el �rbol de decisi�n.
tree <- rpart(Sobrevivi� ~ ., trainT, method = "class")

#Se grafica el �rbol de decisi�n.
rpart.plot(tree)

#Se muestran las predicciones.
p <- predict(tree, trainT, type="class")
table(trainT[,1], p)

# ------- Curvas ROC ------- 

library("ROCR")

#Se trabajar� s�lo con las columnas "Sobrevivi�", "Clase", "Edad", "Hermanos/C�nyugues", "Padres/Ni�os" y "Tarifa".
trainT <- subset(train, select = c(1,2,5,6,7,9))
#Se trabajar� s�lo con las columnas "Clase", "Edad", "Hermanos/C�nyugues", "Padres/Ni�os" y "Tarifa".
testT <- subset(test, select = c(2,5,6,7,9))

set.seed(1)
#Se obtiene el �rbol de decisi�n nuevamente. OJO.
tree <- rpart(Sobrevivi� ~ ., trainT, method = "class")
prob <- predict(tree, testT, type = "prob")[,1]
prob_tree <- prob
pred <- prediction(prob,testT$Sobrevivi�)

#Se obtiene la tasa de verdaderos y falsos positivos
perf <- performance(pred,"tpr","fpr")

#Se grafica la tasa anterior
plot(perf)

#Se obtiene el �rea bajo la curva
set.seed(1)
tree <- rpart(Sobrevivi� ~ ., trainT, method = "class")
prob <- predict(tree, testT, type = "prob")[,2]
prob_curve <- prob
pred <- prediction(prob,testT$Sobrevivi�)

perf <- performance(pred,"auc")
#Se obtiene el porcentaje de precisi�n
perf@y.values[[1]] * 100

#Se comparan los m�todos
pred_tree <- prediction(prob_tree,testT$Sobrevivi�)
pred_curve <- prediction(prob_curve,testT$Sobrevivi�)

perf_tree <- performance(pred_tree,"tpr","fpr")
perf_curve <- performance(pred_curve,"tpr","fpr")

#Se grafica el desempe�o de ambos m�todos
plot(perf_tree)
plot(perf_curve)

# ------- M�quinas de Soporte Vectorial ------- 

library("e1071")

#Se trabajar� s�lo con las columnas "Sobrevivi�", "Clase", "Sexo", "Edad", "Hermanos/C�nyugues", "Padres/Ni�os" y "Tarifa".
trainSVM <- subset(train, select = c(1,2,4,5,6,7,9))
#Se trabajar� s�lo con las columnas "Clase", "Sexo", "Edad", "Hermanos/C�nyugues", "Padres/Ni�os" y "Tarifa".
trainSVM2 <- subset(train, select = c(2,4,5,6,7,9))

#Sigmoid

#Se entrena el modelo con el kernel "sigmoid".
svm_model <- svm(Sobrevivi� ~ .,kernel="sigmoid",data = trainSVM)
pred <- predict(svm_model,trainSVM2)
pred <- replace(pred, pred < 0.5, 0)
pred <- replace(pred, pred >= 0.5, 1)

#Se muestra la matriz de confusi�n.
table(pred, trainSVM$Sobrevivi�)

#Radial

#Se entrena el modelo con el kernel "radial".
svm_model <- svm(Sobrevivi� ~ .,kernel="radial",data = trainSVM)
pred <- predict(svm_model,trainSVM2)
pred <- replace(pred, pred < 0.5, 0)
pred <- replace(pred, pred >= 0.5, 1)

#Se muestra la matriz de confusi�n.
table(pred, trainSVM$Sobrevivi�)

#Polynomial

#Se entrena el modelo con el kernel "polynomial".
svm_model <- svm(Sobrevivi� ~ .,kernel="polynomial",data = trainSVM)
pred <- predict(svm_model,trainSVM2)
pred <- replace(pred, pred < 0.5, 0)
pred <- replace(pred, pred >= 0.5, 1)

#Se muestra la matriz de confusi�n.
table(pred, trainSVM$Sobrevivi�)

#Linear

#Se entrena el modelo con el kernel "linear".
svm_model <- svm(Sobrevivi� ~ .,kernel="linear",data = trainSVM)
pred <- predict(svm_model,trainSVM2)
pred <- replace(pred, pred < 0.5, 0)
pred <- replace(pred, pred >= 0.5, 1)

#Se muestra la matriz de confusi�n.
table(pred, trainSVM$Sobrevivi�)

#Se afina el modelo considerando el kernel "radial"
svm_tune <- 0
svm_tune <- tune(svm, train.x=trainSVM, train.y=trainSVM2, kernel="radial", ranges= list(cost = c(0.1, 1, 10, 100, 1000), gamma = c(0.5, 1, 2, 3, 4)))
print(svm_tune)

#Se eval�a el modelo luego de afinarlo y conseguir el cost y gamma �ptimos. OJO.
svm_model_after_tune <- svm(Sobrevivi� ~ ., kernel="radial", data = trainSVM, cost=1, gamma=0.5)
summary(svm_model_after_tune)
pred <- predict(svm_model_after_tune,trainSVM2)
pred <- replace(pred, pred < 0.5, 0)
pred <- replace(pred, pred >= 0.5, 1)

#Se muestra la matriz de confusi�n.
table(pred,trainSVM$Sobrevivi�)




