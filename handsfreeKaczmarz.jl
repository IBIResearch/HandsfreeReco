using RegularizedLeastSquares, SparseArrays, VectorizationBase, ProgressMeter
import Base: length, iterate
using VectorizationBase: shufflevector, zstridedpointer

###                                ###
### Struct handsfreeKaczmarzSolver ###
###                                ###
mutable struct handsfreeKaczmarz{matT,T,U} 
    S::matT
    u::Vector{T}
    reg::Vector{Regularization}
    denom::Vector{U}
    rowindex::Vector{Int64}
    rowIndexCycle::Vector{Int64}
    cl::Vector{T}
    vl::Vector{T}
    εw::Vector{T}
    τl::T
    αl::T
    weights::Vector{U}
    norm_cl::Vector{U}
    resid::Vector{U}
    curvature::Vector{U}
    iterbounds::Tuple{Int64,Int64}
    stoppingParas::Tuple{Float64,Float64}
    expected_iters::Int64
    wanted_iters::Int64
    minFreq::Float64
    finalfreq::Matrix{Float64}
    recChannels::Vector{Int64}
    startλ::Float64
    SNRbounds::Tuple{Float64,Float64}
    regMatrix::Union{Nothing,Vector{U}} # Tikhonov regularization matrix
end

###                                 ###
### BuildUp handsfreeKaczmarzSolver ###
###                                 ###
function handsfreeKaczmarz(S, finalfreq;
    b=nothing,
    reg = nothing,
    weights=nothing,
    shuffleRows::Bool=false,
    iterbounds::Tuple{Int64,Int64}=(1,50),
    stoppingParas::Tuple{Float64,Float64}=(0.25,2.0),
    expected_iters::Int64=0,
    wanted_iters::Int64=0,
    recChannels::Vector{Int64}=[1,2,3],
    startλ::Float64=5.0,
    SNRbounds::Tuple{Float64,Float64}=(60.0,1.5),
    regMatrix=nothing,
    minFreq = 80e3, 
    kargs...)

    T = real(eltype(S))
    min_iter,max_iter = iterbounds
    M,N = size(S)

    λl = [T(startλ ./ (1 .+ (0.2 .* l .- 0.2).^5)) for l=1:max_iter]
    
    trace = MPIReco.calculateTraceOfNormalMatrix(S,nothing)
    λl .= λl .* (trace / N)
    
    if reg == nothing
        reg = RegularizedLeastSquares.Regularization(["L2" for i=1:max_iter], λl)
    end
    
    if regMatrix != nothing
        regMatrix = T.(regMatrix) # make sure regMatrix has the same element type as S 
        S = transpose(1 ./ sqrt.(regMatrix)) .* S # apply Tikhonov regularization to system matrix
    end

    # make sure weights are not empty
    w = (weights!=nothing ? weights : ones(T,size(S,1)))
    
    # setup denom and rowindex
    denom, rowindex = handsfreeinitkaczmarz(S, λl[1], w)
    rowIndexCycle = collect(1:length(rowindex))
    
    if b != nothing
      u = b
    else
      u = zeros(eltype(S),M)
    end
    cl = zeros(eltype(S),N)
    vl = zeros(eltype(S),M)
    εw = zeros(eltype(S),length(rowindex))
    τl = zero(eltype(S))
    αl = zero(eltype(S))
    norm_cl = zeros(eltype(T),max_iter)
    resid = zeros(eltype(T),max_iter)
    curvature = zeros(eltype(T),max_iter)
  
    return handsfreeKaczmarz(S,u,reg,denom,rowindex,rowIndexCycle,cl,vl,εw,τl,αl,T.(w),norm_cl,resid,curvature,iterbounds,stoppingParas,expected_iters,wanted_iters,minFreq,finalfreq,recChannels,startλ,SNRbounds,regMatrix)    
end

###                                        ###
### Initialization handsfreeKaczmarzSolver ###
###                                        ###
function init!(solver::handsfreeKaczmarz
  ; S::matT=solver.S
  , u::Vector{T}=T[]
  , cl::Vector{T}=T[]
  , weights::Vector{R}=solver.weights) where {T,matT,R}

  if S != solver.S
      solver.denom, solver.rowindex = handsfreeinitkaczmarz(S,solver.reg[1].λ,weights)
      solver.rowIndexCycle = collect(1:length(solver.rowindex))
  end

  solver.u[:] .= u
  solver.weights=weights

  # start vector
  if isempty(cl)
      solver.cl[:] .= zero(T)
  else
      solver.cl[:] .= cl
  end
  solver.vl[:] .= zero(T)

  solver.resid[:] .= zero(T)
  solver.norm_cl[:] .= zero(T)
  solver.curvature[:] .= zero(T)

  for i=1:length(solver.rowindex)
      j = solver.rowindex[i]
      solver.ɛw[i] = sqrt(solver.reg[1].λ) / weights[j]
  end
end

###                                        ###
### Solve-Function handsfreeKaczmarzSolver ###
###                                        ###
function solve(solver::handsfreeKaczmarz, u::Vector{T};
  S::matT=solver.S, startVector::Vector{T}=eltype(S)[]
  , weights::Vector=solver.weights,solverInfo=nothing, kargs...) where {T,matT}

  # initialize solver parameters
  init!(solver; S=S, u=u, cl=startVector, weights=weights)

  # perform Kaczmarz iterations
  i=1
  for item in solver
      solverInfo != nothing && RegularizedLeastSquares.storeInfo(solverInfo,solver.cl,real(item))
      i+=1
  end
  
  # backtransformation of solution with Tikhonov matrix
  if solver.regMatrix != nothing
      solver.cl = solver.cl .* (1 ./ sqrt.(solver.regMatrix))
  end

  return (solver.cl,i,solver.curvature)
end

###                                          ###
### Iterate-Function handsfreeKaczmarzSolver ###
###                                          ###
function iterate(solver::handsfreeKaczmarz, iteration::Int=1)
    
  T = typeof(real(solver.S[1]))

  ## determine frequency selection with SNRtresh of this iteration 
  threshl = [j > solver.SNRbounds[2] ? T(j) : T(solver.SNRbounds[2]) for j in (solver.SNRbounds[1]) / (1 + (0.28 * iteration - 0.28)^2)][1]

### Fettes Problem!!! TODO: filterfreqs brauch MPIFile. besser: SNRs direkt dabei haben und hier aussortieren  

  #freql = filterFrequencies(bSF, minFreq=solver.minFreq, SNRThresh=threshl, recChannels=solver.recChannels)
   # indexl as a subset of freq (base frequencies with SNRthresh = 1.5)
  indexl = sort(sortperm(solver.finalfreq[:,2])[findfirst(f->f>threshl,sort(solver.finalfreq[:,2])):end])
#[findfirst(f->f==freql[i],solver.finalfreq[:,1]) for i=1:length(freql)]

  ## get kaczmarz denomenetor and weighted lambda for lambda and SNRthresh of this iteration
  denoml,ɛwl = getkaczmarz_stuff(solver,indexl,iteration)

  ## perform kaczmarz iteration on all frequencies in indexl
  for i = 1:length(indexl)
    j = indexl[i]
    solver.τl = RegularizedLeastSquares.dot_with_matrix_row(solver.S,solver.cl,j)
    solver.αl = denoml[i]*(solver.u[j]-solver.τl-ɛwl[i]*solver.vl[j])
    handsfreekaczmarz_update!(solver.S,solver.cl,j,solver.αl)
    solver.vl[j] += solver.αl*ɛwl[i]
  end

  # invoke constraints
  RegularizedLeastSquares.applyConstraints(solver.cl, nothing, false, false, nothing)

  solver.norm_cl[iteration] = norm(real(solver.cl))
  solver.resid[iteration] = norm(- sqrt(solver.reg[iteration].λ) * solver.vl[indexl]) / length(indexl)
  
  if iteration == 1
      solver.curvature[iteration] = 0
  elseif iteration == 2
      dcdr = (solver.norm_cl[iteration]-solver.norm_cl[iteration-1]) / (solver.resid[iteration]-solver.resid[iteration-1])
      solver.curvature[iteration] = ( (dcdr - 0) / (solver.resid[iteration]-solver.resid[iteration-1]) ) / ((1 + dcdr^2)^(3/2))
  else
      dcdr_old = (solver.norm_cl[iteration-1] - solver.norm_cl[iteration-2]) / (solver.resid[iteration-1] - solver.resid[iteration-2])
      dcdr = (solver.norm_cl[iteration] - solver.norm_cl[iteration-1]) / (solver.resid[iteration] - solver.resid[iteration-1])
      solver.curvature[iteration] = ( (dcdr - dcdr_old) / (solver.resid[iteration]-solver.resid[iteration-1]) ) / ((1 + dcdr^2)^(3/2))
  end

  if done(solver,iteration)
       return nothing
  else
      return solver.resid[iteration], iteration+1
  end
end

###                                                ###
### StopCriterion-Function handsfreeKaczmarzSolver ###
###                                                ###
function done(solver::handsfreeKaczmarz,iteration::Int)
  if solver.expected_iters != 0 && iteration >= round(Int,solver.expected_iters*3/2)    
      tmp=solver.expected_iters
      @info "Would like to go more than $iteration iterations, but expect $tmp iterations."
      if !(solver.wanted_iters < solver.expected_iters)
          solver.wanted_iters = iteration+2
      end
      return true
  elseif iteration >= 2 && iteration >= solver.iterbounds[1] &&  solver.curvature[iteration] > solver.stoppingParas[1] * solver.norm_cl[1] && abs(solver.curvature[iteration-1]*solver.stoppingParas[2]) < abs(solver.curvature[iteration])
      if iteration >= round(Int,solver.expected_iters*1/2)
          tmp=solver.expected_iters
          @info "Stopped after $iteration iterations. Expected $tmp iterations."
          solver.wanted_iters = iteration
          return true
      else
          tmp = solver.expected_iters
          tmp2 = round(Int,solver.expected_iters*2/5)
          solver.expected_iters = tmp2
          solver.wanted_iters = iteration
          @info "Would like to stop after $iteration iterations, but expect $tmp iterations. Update expected iterations for this reco to $tmp2."
          return false
      end
  elseif iteration >= solver.iterbounds[2]
      tmp = solver.expected_iters
      @info "Stopped at the max iter-bound of $iteration iterations. Expected $tmp iterations."
      solver.wanted_iters = iteration
      return true
  else
      return false
  end
end

# some helper-functions...

function getkaczmarz_stuff(solver::handsfreeKaczmarz,indexl,iteration::Int)
  S = solver.S
  λ = solver.reg[iteration].λ
  weights = solver.weights

  T = typeof(real(S[1]))
  denoml = T[]
  ɛwl = T[]

  for j in indexl
    s² = rownorm²(S,j)*weights[j]^2
    if s²>0
      push!(denoml,weights[j]^2/(s²+λ))
      push!(ɛwl,sqrt(λ) / weights[j])
    end
  end
  return denoml,ɛwl
end

function handsfreeinitkaczmarz(S::AbstractMatrix,λ,weights::Vector)
  T = typeof(real(S[1]))
  denom = T[]
  rowindex = Int64[]

  for i=1:size(S,1)
    s² = rownorm²(S,i)*weights[i]^2
    if s²>0
      push!(denom,weights[i]^2/(s²+λ))
      push!(rowindex,i)
    end
  end
  denom, rowindex
end

function handsfreekaczmarz_update!(A::DenseMatrix{T}, x::Vector, k::Integer, beta) where T
  @simd for n=1:size(A,2)
    @inbounds x[n] += beta*conj(A[k,n])
  end
end

function handsfreekaczmarz_update!(B::Transpose{T,S}, x::Vector,
  k::Integer, beta) where {T,S<:DenseMatrix}
  A = B.parent
  @inbounds @simd for n=1:size(A,1)
      x[n] += beta*conj(A[n,k])
  end
end

function handsfreekaczmarz_update!(B::Transpose{T,S}, x::Vector,
  k::Integer, beta) where {T,S<:SparseMatrixCSC}
  A = B.parent
  N = A.colptr[k+1]-A.colptr[k]
  for n=A.colptr[k]:N-1+A.colptr[k]
      @inbounds x[A.rowval[n]] += beta*conj(A.nzval[n])
  end
end
