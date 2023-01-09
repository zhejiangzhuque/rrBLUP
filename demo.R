library(rrBLUP)

train <- read.csv('data/protein.train.nan-101.csv')
test <- read.csv('data/protein.test.nan-101.csv')

train_x <- as.matrix(train[-1])
train_y <- as.matrix(train['label'])
test_x <- as.matrix(test[-1])
test_y <- as.matrix(test['label'])

# A.mat
start_time <- Sys.time()
Amat <- A.mat(train_x)
print(Amat)
print(Sys.time()-start_time)

# EM impute method
start_time <- Sys.time()
Amat <- A.mat(train_x, impute.method='EM')
print(Amat)
print(Sys.time()-start_time)

# mean impute method, return imputed 
start_time <- Sys.time()
result <- A.mat(train_x, impute.method='mean', return.imputed=TRUE)
Amat <- result$A
train_x_imp <- result$imputed
print(Amat)
print(train_x_imp)
print(Sys.time()-start_time)

# EJ shrink method
start_time <- Sys.time()
Amat <- A.mat(train_x, shrink=TRUE)
print(Amat)
print(Sys.time()-start_time)

# REG shrink method
start_time <- Sys.time()
Amat <- A.mat(train_x, shrink=list(method="REG",n.qtl=100,n.iter=5))
print(Amat)
print(Sys.time()-start_time)


train <- read.csv('data/protein.train.csv')
test <- read.csv('data/protein.test.csv')

train_x <- as.matrix(train[-1])
train_y <- as.matrix(train['label'])
test_x <- as.matrix(test[-1])
test_y <- as.matrix(test['label'])

# mixed.solve using Matrix Z
start_time <- Sys.time()
result <- mixed.solve(y=train_y, Z=train_x)
print(result)
print(Sys.time()-start_time)
test_pred <- (test_x %*% as.matrix(result$u)) + as.vector(result$beta)
print(test_y)
print(test_pred)

# mixed.solve using Matrix K
start_time <- Sys.time()
result <- mixed.solve(y=train_y, K=A.mat(train_x))
print(result)
print(Sys.time()-start_time)
train_pred <- as.matrix(result$u) + as.vector(result$beta)
print(train_y)
print(train_pred)