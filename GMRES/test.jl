include("gmres.jl")
using gmres


#for tol in [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21, 1e-24, 1e-27, 1e-30]

res = Float64[]
for i in 1:20
    a = eye(100,100)*10
    v = rand(100)*10
    a = lufact(a)
    push!(res, criteria(restGmres(a[:L],v, maxIter=1),a[:L]\v))
    push!(res, criteria(restGmres(a[:U],v, maxIter=1),a[:U]\v))
end
println('\t', mean(res))
#end
