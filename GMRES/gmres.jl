module gmres
export criteria, restGmres, hessenberg

function criteria(v2,v1)
    w = v2 - v1
    dot(w,w)/dot(v2,v2)
end

function init(A, x, b)
  r = b - A*x
  beta = norm(r)
  v1 = r/beta
  return r, beta, v1
end

function solveU!(U, b)
  n = length(b)
  x = b
  for i in n:-1:1
    for j in n:-1:i+1
      x[i] -= U[i,j]*x[j]
    end
    x[i] /= U[i,i]
  end
  return x
end

function least_squares!(Hess, beta)
  m = length(Hess[1,:])
  b = zeros(m+1)
  b[1] = beta
  #Givens Rotations
  for i in 1:m
    c = Hess[i,i]/sqrt(Hess[i,i]^2 + Hess[i+1,i]^2)
    s = Hess[i+1,i]/sqrt(Hess[i,i]^2 + Hess[i+1,i]^2)
    #Right Side
    b[i], b[i+1] = c*b[i], -s*b[i]
    #Left Side
    for j in i:m
      Hess[i,j], Hess[i+1,j] = c*Hess[i,j]+s*Hess[i+1,j], -s*Hess[i,j]+c*Hess[i+1,j]
    end
    Hess[i+1,i] = 0
  end
  return solveU!(Hess[1:m,1:m],b[1:m])
end

function hessenberg(A,v1,n,m)
  Hess = zeros(m+1,m)
  Vm = zeros(n,m)
  Vm[:,1] = v1
  for j in 1:m
    w = A*Vm[:,j]
    for i in 1:j
      Hess[i,j] = dot(w, Vm[:,i])
      w -= Hess[i,j]*Vm[:,i]
    end
    Hess[j+1,j] = norm(w)
    if Hess[j+1,j] == 0 || j == m
      return Hess[1:j+1,1:j], Vm[:,1:j]
    end
    Vm[:,j+1] = w/Hess[j+1,j]
  end
end

function restGmres(
    A, b;
    x = zeros(length(b)),
    restart::Int=min(20,length(b)),
    tolerance = 1.e-9,
    maxIter = 5
  )

  n = length(b)
  for iter in 1:maxIter
    for m in 1:restart
      r, beta, v1 = init(A, x, b)
      Hess, Vm = hessenberg(A,v1,n,m)
      y = least_squares!(Hess,beta)
      x = x + Vm*y
      if (norm(b - A*x) < tolerance)
        return x
      end
    end
  end
  return x
end

end
